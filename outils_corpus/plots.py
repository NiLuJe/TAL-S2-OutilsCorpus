#!/usr/bin/env python3


import altair as alt
from loguru import logger
import polars as pl
import typer

from outils_corpus.config import CATEGORIES_VIZ, FULL_DATASET

app = typer.Typer()


def plot_category_distribution() -> None:
	"""
	Class distribution plot (i.e., publication year, in 100 year intervals)
	"""

	logger.info("Generating categorical distribution plot from final data...")

	lf = pl.scan_parquet(FULL_DATASET)

	distrib = (
		# We don't need any other columns
		lf.select("century", "text")
		# Duh'
		.group_by("century")
		# Compute the amount of rows per group (i.e., documents)
		# Then the the amount of characters per group (each group is an aggregate of rows)
		.agg(pl.len().alias("documents"), pl.col("text").str.len_chars().sum().alias("characters"))
		.collect()
	)

	# Documents (i.e., rows) per individual category distribution
	chart = (
		alt.Chart(distrib)
		.encode(
			x="documents:Q",
			y="century:N",
			text="documents:Q",
		)
		.properties(
			title="Class distribution",
		)
	)
	chart.encoding.x.title = "Document count"
	chart.encoding.y.title = "Class"
	chart = chart.mark_bar(tooltip=True) + chart.mark_text(align="left", dx=2)
	chart.save(CATEGORIES_VIZ)

	# Maybe slightly more telling, *characters* per category
	chart = (
		alt.Chart(distrib)
		.encode(
			x="characters:Q",
			y="century:N",
			text="characters:Q",
		)
		.properties(
			title="Class distribution",
		)
	)
	chart.encoding.x.title = "Characters"
	chart.encoding.y.title = "Class"
	chart = chart.mark_bar(tooltip=True) + chart.mark_text(align="left", dx=2)
	chart.save(CATEGORIES_VIZ.with_stem(CATEGORIES_VIZ.stem + "-chars"))

	logger.success("Plot generation complete.")


@app.command()
def main() -> None:
	plot_category_distribution()


if __name__ == "__main__":
	app()
