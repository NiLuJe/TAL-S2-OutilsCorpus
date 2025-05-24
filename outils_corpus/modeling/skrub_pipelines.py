#!/usr/bin/env python3

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from polars_splitters import split_into_train_eval
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from skrub import GapEncoder, MinHashEncoder, StringEncoder, TableVectorizer, TextEncoder
import typer

from outils_corpus.config import FIGURES_DIR, FULL_DATASET

app = typer.Typer()


def plot_gap_feature_importance(X_trans):
	x_samples = X_trans.pop("text")

	# We slightly format the topics and labels for them to fit on the plot
	topic_labels = [x.replace("text: ", "") for x in X_trans.columns]
	labels = x_samples.str[:50].values + "..."

	# We clip large outliers to make activations more visible.
	X_trans = np.clip(X_trans, a_min=None, a_max=200)

	plt.figure(figsize=(10, 10), dpi=200)
	plt.imshow(X_trans.T)

	plt.yticks(
		range(len(topic_labels)),
		labels=topic_labels,
		ha="right",
		size=12,
	)
	plt.xticks(range(len(labels)), labels=labels, size=12, rotation=50, ha="right")

	plt.colorbar().set_label(label="Topic activations", size=13)
	plt.ylabel("Latent topics", size=14)
	plt.xlabel("Data entries", size=14)
	plt.tight_layout()
	plt.savefig(FIGURES_DIR / "skrub-gap-feature-importance.png")


def plot_box_results(named_results):
	fig, ax = plt.subplots()
	names, scores = zip(*[(name, result["test_score"]) for name, result in named_results])
	ax.boxplot(scores)
	ax.set_xticks(range(1, len(names) + 1), labels=list(names), size=12)
	ax.set_ylabel("ROC AUC", size=14)
	plt.title(
		"AUC distribution across folds (higher is better)",
		size=14,
	)
	plt.savefig(FIGURES_DIR / "skrub-roc-auc.png")


def plot_performance_tradeoff(results):
	fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
	markers = ["s", "o", "^", "x"]
	for idx, (name, result) in enumerate(results):
		ax.scatter(
			result["fit_time"],
			result["test_score"],
			label=name,
			marker=markers[idx],
		)
		mean_fit_time = np.mean(result["fit_time"])
		mean_score = np.mean(result["test_score"])
		ax.scatter(
			mean_fit_time,
			mean_score,
			color="k",
			marker=markers[idx],
		)
		std_fit_time = np.std(result["fit_time"])
		std_score = np.std(result["test_score"])
		ax.errorbar(
			x=mean_fit_time,
			y=mean_score,
			yerr=std_score,
			fmt="none",
			c="k",
			capsize=2,
		)
		ax.errorbar(
			x=mean_fit_time,
			y=mean_score,
			xerr=std_fit_time,
			fmt="none",
			c="k",
			capsize=2,
		)
		ax.set_xscale("log")

		ax.set_xlabel("Time to fit (seconds)")
		ax.set_ylabel("ROC AUC")
		ax.set_title("Prediction performance / training time trade-off")

	ax.annotate(
		"",
		xy=(1.5, 0.98),
		xytext=(8.5, 0.90),
		arrowprops=dict(arrowstyle="->", mutation_scale=15),
	)
	ax.text(5.8, 0.86, "Best time / \nperformance trade-off")
	ax.legend(bbox_to_anchor=(1.02, 0.3))
	plt.savefig(FIGURES_DIR / "skrub-time-vs-perf.png")


def skrub_pipeline():
	# load our full dataset
	df = pl.read_parquet(FULL_DATASET)

	# Test on a small sample
	df_train, df_test = split_into_train_eval(
		df,
		eval_rel_size=0.1,
		stratify_by="century",
		shuffle=True,
		seed=42,
		validate=True,
		as_lazy=False,
		rel_size_deviation_tolerance=0.1,
	)

	df_test = df_test.to_pandas()

	X = df_test.select("text", "century")
	y = df_test.select("century")

	# GapEncoder
	logger.info("Initial exploratory GapEncoder fit")
	gap = GapEncoder(n_components=30)
	X_trans = gap.fit_transform(X["text"])
	# Add the original text as a column
	X_trans = X_trans.with_columns(text=X["text"])

	plot_gap_feature_importance(X_trans.head())

	results = []

	logger.info("Creating a GapEncoder pipeline")
	gap_pipe = make_pipeline(
		TableVectorizer(high_cardinality=GapEncoder(n_components=30)),
		SGDClassifier(loss="log_loss", alpha=1e-4, n_iter_no_change=3, early_stopping=True),
	)
	logger.info("Cross-validating GapEncoder pipeline")
	gap_results = cross_validate(gap_pipe, X, y, scoring="roc_auc")
	results.append(("GapEncoder", gap_results))

	logger.info("Creating a MinHashEncoder pipeline")
	minhash_pipe = make_pipeline(
		TableVectorizer(high_cardinality=MinHashEncoder(n_components=30)),
		SGDClassifier(loss="log_loss", alpha=1e-4, n_iter_no_change=3, early_stopping=True),
	)
	logger.info("Cross-validating MinHashEncoder pipeline")
	minhash_results = cross_validate(minhash_pipe, X, y, scoring="roc_auc")
	results.append(("MinHashEncoder", minhash_results))

	logger.info("Creating a TextEncoder pipeline")
	text_encoder = TextEncoder(
		"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
		device="cpu",
	)

	text_encoder_pipe = make_pipeline(
		TableVectorizer(high_cardinality=text_encoder),
		SGDClassifier(loss="log_loss", alpha=1e-4, n_iter_no_change=3, early_stopping=True),
	)
	logger.info("Cross-validating TextEncoder pipeline")
	text_encoder_results = cross_validate(text_encoder_pipe, X, y, scoring="roc_auc")
	results.append(("TextEncoder", text_encoder_results))

	logger.info("Creating a StringEncoder pipeline")
	string_encoder = StringEncoder(ngram_range=(3, 4), analyzer="char_wb")

	string_encoder_pipe = make_pipeline(
		TableVectorizer(high_cardinality=string_encoder),
		SGDClassifier(loss="log_loss", alpha=1e-4, n_iter_no_change=3, early_stopping=True),
	)

	logger.info("Cross-validating StringEncoder pipeline")
	string_encoder_results = cross_validate(string_encoder_pipe, X, y, scoring="roc_auc")
	results.append(("StringEncoder", string_encoder_results))

	plot_box_results(results)


@app.command()
def main() -> None:
	skrub_pipeline()


if __name__ == "__main__":
	app()
