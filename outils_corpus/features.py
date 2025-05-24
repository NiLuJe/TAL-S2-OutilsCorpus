#!/usr/bin/env python3

from time import time

from loguru import logger
import numpy as np
import polars as pl
from polars_splitters import split_into_train_eval
from rich.pretty import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
import typer

from outils_corpus.config import FULL_DATASET

app = typer.Typer()


def extract_features():
	"""
	Perform feature extraction (TfidfVectorizer) on our full dataset, in an 80/20 train/test stratified split

	Returns
	-------
	X_train: CSR matrix
	X_test: CSR matrix
	y_train: np.array
	y_test: np.array
	feature_names: np.array
	target_names: np.array
	"""

	# load our full dataset
	df = pl.read_parquet(FULL_DATASET)

	# Do an 80/20 train/test stratified split
	logger.info("Doing a 80/20 stratified train/test split")
	# NOTE: sklearn's train_test_split seems to be having a sad (segfault \o/)
	#       when using stratify and a Polars DataFrame as input, so, use polars-splitters instead...
	# X_train, X_test, y_train, y_test = train_test_split(
	# 	df, df.select("century"), test_size=0.2, random_state=42, shuffle=True, stratify=df.select("century").unique()
	# )
	df_train, df_test = split_into_train_eval(
		df,
		eval_rel_size=0.2,
		stratify_by="century",
		shuffle=True,
		seed=42,
		validate=True,
		as_lazy=False,
		rel_size_deviation_tolerance=0.1,
	)
	y_train = df_train.select("century").to_series().to_numpy()
	y_test = df_test.select("century").to_series().to_numpy()

	# Extracting features from the training data using a sparse vectorizer
	logger.info("Extracting features from the train set")
	t0 = time()
	# NOTE: A higher max_df doesn't help (in fact, it often leads to very slightly lower accuracy)
	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, stop_words=list(fr_stop))
	X_train = vectorizer.fit_transform(df_train.select("text").to_series().to_numpy())
	duration_train = time() - t0

	# Extracting features from the test data using the same vectorizer
	logger.info("Extracting features from the test set")
	t0 = time()
	X_test = vectorizer.transform(df_test.select("text").to_series().to_numpy())
	duration_test = time() - t0

	feature_names = vectorizer.get_feature_names_out()

	# NOTE: Sort this, or the plots are confused as all hell...
	target_names = np.sort(df.select("century").unique().to_series().to_numpy())

	logger.info(f"{len(target_names)} categories")
	logger.info(f"vectorize training done in {duration_train:.3f}s ")
	logger.info(f"n_samples: {X_train.shape[0]}, n_features: {X_train.shape[1]}")
	logger.info(f"vectorize testing done in {duration_test:.3f}s ")
	logger.info(f"n_samples: {X_test.shape[0]}, n_features: {X_test.shape[1]}")

	# Let's see what we've got...
	logger.info("feature_names:")
	pprint(feature_names)
	logger.info("target_names:")
	pprint(target_names)
	logger.info("X_test:")
	pprint(X_test)

	return X_train, X_test, y_train, y_test, feature_names, target_names


@app.command()
def main() -> None:
	extract_features()


if __name__ == "__main__":
	app()
