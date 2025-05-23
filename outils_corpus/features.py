#!/usr/bin/env python3

from pathlib import Path

from loguru import logger
from sklearn.model_selection import train_test_split
import typer

from outils_corpus.config import FULL_DATASET

app = typer.Typer()


def feature_extraction():
	# load our full dataset
	df = pl.read_parquet(FULL_DATASET)

	# Do a 80/20 stratified split
	X_train, X_test, y_train, y_test = train_test_split(df, df.select("century"), test_size=0.2, random_state=42, shuffle=True, stratify=df.select("century").unique())


@app.command()
def main() -> None:
	foo()


if __name__ == "__main__":
	app()
