#!/usr/bin/env python3

from time import time

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
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

	logger.info("top 5 keywords per class:")
	print(top)

	return ax


def benchmark_clf(clf, X_train, X_test, y_train, y_test, feature_names, target_names, custom_name=False):
	"""
	Benchmark a specific classifier
	(Implies training, inference, evaluation & plotting).
	"""

	print("_" * 80)
	logger.info("Training: ")
	logger.info(clf)
	t0 = time()
	clf.fit(X_train, y_train)
	train_time = time() - t0
	logger.info(f"train time: {train_time:.3}s")

	t0 = time()
	pred = clf.predict(X_test)
	test_time = time() - t0
	logger.info(f"test time:  {test_time:.3}s")

	score = metrics.accuracy_score(y_test, pred)
	logger.info(f"accuracy:   {score:.3}")

	if hasattr(clf, "coef_"):
		logger.info(f"dimensionality: {clf.coef_.shape[1]}")
		logger.info(f"density: {density(clf.coef_)}")
		print()

	print()
	if custom_name:
		clf_descr = str(custom_name)
	else:
		clf_descr = clf.__class__.__name__

	# Plot the resulting confusion matrix
	logger.info("Plotting its confusion matrix")
	fig, ax = plt.subplots(figsize=(10, 5))
	metrics.ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
	ax.xaxis.set_ticklabels(target_names)
	ax.yaxis.set_ticklabels(target_names)
	_ = ax.set_title(f"Confusion Matrix for {clf.__class__.__name__}")
	plt.savefig(FIGURES_DIR / f"{clf.__class__.__name__}-confusion-matrix.png")

	# Plot the feature effect, if possible
	if hasattr(clf, "coef_"):
		_ = plot_clf_feature_effects(clf, X_train, target_names, feature_names).set_title("Average feature effect")
		plt.savefig(FIGURES_DIR / f"{clf.__class__.__name__}-feature-effect.png", dpi=450)

	return clf_descr, score, train_time, test_time


def train_models() -> None:
	"""
	Train a bunch of models
	"""

	# Load our embeddings
	X_train, X_test, y_train, y_test, feature_names, target_names = extract_features()

	# Train, pred & eval all the things
	results = []
	for clf, name in (
		(LogisticRegression(C=5, max_iter=1000), "Logistic Regression"),
		(RidgeClassifier(alpha=1.0, solver="sparse_cg"), "Ridge Classifier"),
		(KNeighborsClassifier(n_neighbors=100), "kNN"),
		(RandomForestClassifier(), "Random Forest"),
		# L2 penalty Linear SVC
		(LinearSVC(C=0.1, dual=False, max_iter=1000), "Linear SVC"),
		# L2 penalty Linear SGD
		(
			SGDClassifier(loss="log_loss", alpha=1e-4, n_iter_no_change=3, early_stopping=True),
			"log-loss SGD",
		),
		# NearestCentroid (aka Rocchio classifier)
		(NearestCentroid(), "NearestCentroid"),
		# Sparse naive Bayes classifier
		(ComplementNB(alpha=0.1), "Complement naive Bayes"),
	):
		print("=" * 80)
		logger.info(name)
		results.append(benchmark_clf(clf, X_train, X_test, y_train, y_test, feature_names, target_names, name))


@app.command()
def main() -> None:
	train_models()


if __name__ == "__main__":
	app()
