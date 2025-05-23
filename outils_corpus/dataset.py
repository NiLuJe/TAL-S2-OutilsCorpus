#!/usr/bin/env python3

from contextlib import chdir
import subprocess

from loguru import logger
import niquests
import typer

from outils_corpus.config import PG_MIRROR, PG_RDF_TARBALL

app = typer.Typer()


def download_pg_listing():
	"""
	Download PG's listing of French TXT files
	"""
	# NOTE: PG frowns upon crawling, they instead offer tools to download specific subsets of their catalog,
	#       which we're going to mae use of.
	#       c.f., https://www.gutenberg.org/policy/robot_access.html for more details.

	# Work in a dedicated subfolder
	PG_MIRROR.mkdir(exist_ok=True)
	with chdir(PG_MIRROR):
		# We'll want TXT files, in French
		logger.info("Downloading listing of catalog subset")
		subprocess.run(
			"wget -w 2 -m -H http://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=fr".split(), check=True
		)

		# We're also going to need the metadata catalog
		logger.info("Downloading metadata catalog dump")
		r = niquests.get("https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2", stream=True)
		with open(PG_RDF_TARBALL, "wb") as fd:
			for chunk in r.iter_content():
				fd.write(chunk)


@app.command()
def main():
	# Download the filtered catalog
	download_pg_listing()


if __name__ == "__main__":
	app()
