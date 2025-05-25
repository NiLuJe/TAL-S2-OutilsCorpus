#!/usr/bin/env python3
#
# Based on https://levelup.gitconnected.com/text-classification-in-the-era-of-transformers-2e40babe8024
#
##
from time import time

from datasets import Dataset, DatasetDict
from loguru import logger
import numpy as np
import polars as pl
from polars_splitters import split_into_train_eval
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm.rich import tqdm
import typer

from outils_corpus.config import FIGURES_DIR, FULL_DATASET, MODELS_DIR, SETFIT_DIR

app = typer.Typer()


def create_dataset() -> tuple[Dataset, list[str]]:
	"""
	Wrangle our data in the Dataset format

	Returns
	-------
	ds : DatasetDict
		The full dataset in train/val/test splits
	labels: list[str]
		The list of gold labels
	"""

	# load our full dataset
	df = pl.read_parquet(FULL_DATASET)

	labels = sorted(df.select("century").unique().to_series().to_list())
	labels = [str(label) for label in labels]

	# Do an 80/20 train/eval stratified split
	logger.info("Doing a 80/20 stratified train/eval split")
	df_train, df_eval = split_into_train_eval(
		df,
		eval_rel_size=0.2,
		stratify_by="century",
		shuffle=True,
		seed=42,
		validate=True,
		as_lazy=False,
		rel_size_deviation_tolerance=0.1,
	)

	# Do a further 75/25 val/test stratified split on eval
	logger.info("Doing a 75/25 stratified val/test split")
	df_val, df_test = split_into_train_eval(
		df_eval,
		eval_rel_size=0.25,
		stratify_by="century",
		shuffle=True,
		seed=42,
		validate=True,
		as_lazy=False,
		rel_size_deviation_tolerance=0.1,
	)
	# NOTE: df_val.group_by("century").len() to check the class distribution,
	#       because we have some dangerously low popcounts in a few classes...

	# Convert that to a Dataset
	logger.info("Convert to a DatasetDict")
	train_ds = Dataset.from_polars(
		df_train.select(pl.col("text"), pl.col("century").alias("label")).cast({"label": pl.String})
	)
	# Encode the class labels as such (ClassLabel)
	train_ds = train_ds.class_encode_column("label")

	# Make sure we have full class representation so the ClassLabel encoding is consistent across the splits...
	df_val.extend(
		pl.DataFrame(
			{
				"pg_num": [0, 0, 0],
				"title": ["", "", ""],
				"author": ["", "", ""],
				# These three have extremely low pop count in df_eval (specifically, 1 ;)).
				"century": [1200, 1300, 1400],
				"text": ["", "", ""],
			},
			schema={"pg_num": pl.UInt16, "title": None, "author": None, "century": pl.UInt16, "text": None},
		)
	)
	val_ds = Dataset.from_polars(
		df_val.select(pl.col("text"), pl.col("century").alias("label")).cast({"label": pl.String})
	)
	val_ds = val_ds.class_encode_column("label")

	test_ds = Dataset.from_polars(
		df_test.select(pl.col("text"), pl.col("century").alias("label")).cast({"label": pl.String})
	)
	test_ds = test_ds.class_encode_column("label")
	ds = DatasetDict(
		{
			"train": train_ds,
			"val": val_ds,
			"test": test_ds,
		}
	)

	return ds, labels


def train_setfit(dataset: DatasetDict, labels: list[str]) -> SetFitModel:
	"""
	Fine-tune a SentenceTransformer checkpoint via SetFit

	Parameters
	----------
	dataset : DatasetDict
		Input dataset in train/val/test splits, as returned by `create_dataset`
	labels : list[str]
		List of gold labels, as returned by `create_dataset`

	Returns
	-------
	model : SetFitModel
		A trained SetFitModel, fine-tuned on the input dataset
	"""

	train_dataset = dataset["train"]
	eval_dataset = dataset["val"]

	# Test on a small subset of data, because this is a hungry caterpillar...
	# NOTE: Besides the training time,
	#       there's a step right before training that temporarily gobbles up a ginormous amount of RAM...
	#       A larger sample size would require > 64GB of RAM.
	tmp_train_dataset = train_dataset.select(range(100)).shuffle()
	tmp_eval_dataset = eval_dataset.select(range(33)).shuffle()

	# NOTE: Make sure to use a checkpoint that was actually trained on French ;o)
	checkpoint = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

	# Load a SetFit model from Hub
	logger.info("Loading a SentenceTransformer checkpoint")
	model = SetFitModel.from_pretrained(
		checkpoint, use_differentiable_head=True, head_params={"out_features": len(labels)}, labels=labels
	)

	# Team Red over here, and ROCm doesn't really seem to help, so this is likely a noop on my machine...
	model.to("cuda")

	args = TrainingArguments(
		batch_size=4, num_epochs=2, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True
	)

	logger.info("Training a SetFit on our data . . .")
	trainer = Trainer(
		model=model,
		args=args,
		train_dataset=tmp_train_dataset,
		eval_dataset=tmp_eval_dataset,
		metric="accuracy",
		column_mapping={"text": "text", "label": "label"},
	)

	# Finetune the model
	trainer.train()

	# Save the model
	SETFIT_DIR.mkdir(exist_ok=True)
	trainer.model._save_pretrained(MODELS_DIR / "setfit-trained")

	return trainer.model


def calculate_f1_score(y_true: np.array, y_pred: np.array) -> dict[str, str]:
	"""
	Calculates micro and macro F1-scores given the predicted and actual labels

	Parameters
	----------
	y_true : np.array
		Actual labels
	y_pred : np.array
		Predicted labels

	Returns
	-------
	dict :
		A dictionary containing micro f1 and macro f1 scores.
	"""

	# Generate a classification report to compute detailed metrics
	clf_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
	# Also dump it to text
	report = classification_report(y_true, y_pred, zero_division=0)
	(FIGURES_DIR / "SetFit-classification-report.txt").write_text(report)

	return {"micro f1": clf_dict["weighted avg"]["f1-score"], "macro f1": clf_dict["macro avg"]["f1-score"]}


def load_setfit() -> SetFitModel:
	"""
	Load our fine-tuned model from disk
	"""

	return SetFitModel._from_pretrained(SETFIT_DIR.as_posix())


def inference(model: SetFitModel, test_dataset: Dataset):
	"""
	Run inference on test data via our fine-tuned model

	Parameters
	----------
	model : SetFitModel
		A trained SetFitModel
	test_dataset : Dataset
		The data to run inference on
	"""

	# DataLoader for batching
	logger.info("Loading data for inference")
	batch_size = 4
	dataloader = DataLoader(test_dataset, batch_size=batch_size)

	predicted_labels = []
	# Expand our ClassLabels to their string representation
	actual_labels = [test_dataset.features["label"].int2str(sample["label"]) for sample in test_dataset]

	# Generate predictions in batches
	logger.info("Predicting . . .")
	start_time = time()
	for i, inputs in enumerate(tqdm(dataloader)):
		predictions = model.predict(inputs["text"])
		# NOTE: Not using a GPU, so, no need to jump through extra hoops here
		# predicted_labels.extend(list(tmp) for tmp in predictions.detach().cpu().numpy())
		predicted_labels.extend(predictions)
	end_time = time()

	print(end_time - start_time)

	# Quick evaluation
	logger.info("Report:")
	report = calculate_f1_score(actual_labels, predicted_labels)
	print(report)


@app.command()
def main() -> None:
	# Wrangle our parquet dataset in a DatasetDict
	ds, labels = create_dataset()
	test_dataset = ds["test"]

	# Fine-tune a SetFit model on our data, from a sentence-transformer checkpoint
	model = train_setfit(ds, labels)
	# Or load our saved model from an earlier training run
	# model = load_setfit()

	# Run inference
	inference(model, test_dataset)


if __name__ == "__main__":
	app()
