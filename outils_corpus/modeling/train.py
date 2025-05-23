#!/usr/bin/env python3


from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import typer

from outils_corpus.config import FIGURES_DIR
from outils_corpus.features import extract_features

app = typer.Typer()


def plot_clf_feature_effects(clf, X_train, target_names, feature_names) -> None:
	"""
	Plot the impact of specific features on the classifier's decision,
	c.f., https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#model-without-metadata-stripping
	"""

	# learned coefficients weighted by frequency of appearance
	average_feature_effects = clf.coef_ * np.asarray(X_train.mean(axis=0)).ravel()

	for i, label in enumerate(target_names):
		top5 = np.argsort(average_feature_effects[i])[-5:][::-1]
		if i == 0:
			top = pd.DataFrame(feature_names[top5], columns=[label])
			top_indices = top5
		else:
			top[label] = feature_names[top5]
			top_indices = np.concatenate((top_indices, top5), axis=None)
	top_indices = np.unique(top_indices)
	predictive_words = feature_names[top_indices]

	# plot feature effects
	bar_size = 0.25
	padding = 0.75
	y_locs = np.arange(len(top_indices)) * (4 * bar_size + padding)

	fig, ax = plt.subplots(figsize=(10, 8))
	for i, label in enumerate(target_names):
		ax.barh(
			y_locs + (i - 2) * bar_size,
			average_feature_effects[i, top_indices],
			height=bar_size,
			label=label,
		)
	ax.set(
		yticks=y_locs,
		yticklabels=predictive_words,
		ylim=[
			0 - 4 * bar_size,
			len(top_indices) * (4 * bar_size + padding) - 4 * bar_size,
		],
	)
	ax.legend(loc="lower right")

	print("top 5 keywords per class:")
	print(top)

	return ax


def train_models() -> None:
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
	_ = ax.set_title(f"Confusion Matrix for {clf.__class__.__name__}")
	plt.savefig(FIGURES_DIR / "CLF-confusion-matrix.png")

	_ = plot_clf_feature_effects(clf, X_train, target_names, feature_names).set_title("Average feature effect")
	plt.savefig(FIGURES_DIR / "CLF-feature-effect.png", dpi=450)


@app.command()
def main() -> None:
	train_models()


if __name__ == "__main__":
	app()
