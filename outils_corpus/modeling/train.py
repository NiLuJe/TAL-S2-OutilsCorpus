#!/usr/bin/env python3


from loguru import logger
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import typer

from outils_corpus.config import FIGURES_DIR
from outils_corpus.features import extract_features

app = typer.Typer()


def train_models():
	"""
	Train a bunch of models
	"""

	# Load our embeddings
	X_train, X_test, y_train, y_test, feature_names, target_names = extract_features()

	# CLF
	logger.info("Training a CLF")
	clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)

	# Plot the resulting confusion matrix
	logger.info("Plotting its confusion matrix")
	fig, ax = plt.subplots(figsize=(10, 5))
	ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
	ax.xaxis.set_ticklabels(target_names)
	ax.yaxis.set_ticklabels(target_names)
	_ = ax.set_title(f"Confusion Matrix for {clf.__class__.__name__}\non the original documents")
	plt.save(FIGURES_DIR / "CLF-confusion-matrix.png")


@app.command()
def main() -> None:
	train_models()


if __name__ == "__main__":
	app()
