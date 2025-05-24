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

from outils_corpus.config import FIGURES_DIR, FULL_DATASET

app = typer.Typer()


def create_dataset() -> tuple[Dataset, np.array]:
	"""
	Wrangle our data in the Dataset format
	"""

	# load our full dataset
	df = pl.read_parquet(FULL_DATASET)

	labels = np.sort(df.select("century").unique().to_series().to_numpy())

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
		eval_rel_size=0.2,
		stratify_by="century",
		shuffle=True,
		seed=42,
		validate=True,
		as_lazy=False,
		rel_size_deviation_tolerance=0.1,
	)

	# Convert that to a Dataset
	logger.info("Convert to a DatasetDict")
	ds = DatasetDict(
		{
			"train": Dataset.from_polars(df_train.select(pl.col("text"), pl.col("century").alias("label"))),
			"val": Dataset.from_polars(df_val.select(pl.col("text"), pl.col("century").alias("label"))),
			"test": Dataset.from_polars(df_test.select(pl.col("text"), pl.col("century").alias("label"))),
		}
	)

	return ds, labels


def train_setfit(dataset: Dataset, labels: np.array):
	"""
	Fine-tune a SentenceTransformer model via SetFit
	"""

	train_dataset = dataset["train"]
	eval_dataset = dataset["val"]
	test_dataset = dataset["test"]

	# Test on a small subset of data, because this is a hungry caterpillar...
	tmp_train_dataset = train_dataset.select(range(75)).shuffle()
	tmp_eval_dataset = eval_dataset.select(range(25)).shuffle()

	# labels=labels

	# NOTE: Make sure to use a checkpoint that was actually trained on French ;o)
	checkpoint = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

	# Load a SetFit model from Hub
	logger.info("Loading a SentenceTransformer checkpoint")
	model = SetFitModel.from_pretrained(
		checkpoint, use_differentiable_head=True, head_params={"out_features": len(labels)}, labels=labels
	)

	# Team Red over here, and apparently my numpy build is borked, so can't test rocm ;'(
	# model.to("cuda")

	args = TrainingArguments(
		batch_size=4, num_epochs=2, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True
	)

	logger.info("Training a SetFit on our data")
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

	return trainer.model, test_dataset


def calculate_f1_score(y_true, y_pred):
	"""
	Calculates micro and macro F1-scores given the predicted and actual labels

	Parameters
	----------
	y_true (numpy array) :
		Actual labels
	y_pred (numpy array) :
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
	(FIGURES_DIR / "SimFit-classification-report.txt").write_text(report)

	return {"micro f1": clf_dict["micro avg"]["f1-score"], "macro f1": clf_dict["macro avg"]["f1-score"]}


def inference(model, test_dataset: Dataset, labels: np.array):
	"""
	Run inference via our fine-tuned model
	"""

	# DataLoader for batching
	logger.info("Loading data for inference")
	batch_size = 4
	dataloader = DataLoader(test_dataset, batch_size=batch_size)

	predicted_labels = []
	actual_labels = [sample["label"] for sample in test_dataset]

	# Generate predictions in batches
	logger.info("Predicting...")
	start_time = time.time()
	for i, inputs in tqdm(enumerate(dataloader)):
		predictions = model.predict(inputs["text"])
		predicted_labels.extend(list(tmp) for tmp in predictions.detach().cpu().numpy())
	end_time = time.time()

	print(end_time - start_time)

	# Eval
	logger.info("Report:")
	report = calculate_f1_score(actual_labels, predicted_labels)
	print(report)


@app.command()
def main() -> None:
	ds, labels = create_dataset()
	model, test_dataset = train_setfit(ds, labels)
	inference(model, labels, test_dataset)


if __name__ == "__main__":
	app()
