from contextlib import chdir
from pathlib import Path
import os
import subprocess

from loguru import logger
from tqdm.rich import tqdm
import typer
import niquests

from outils_corpus.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, EXTERNAL_DATA_DIR, PG_MIRROR, PG_RDF_TARBALL

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
		subprocess.run("wget -w 2 -m -H 'http://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=fr'", check=True)

		# We're also going to need the metadata catalog
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
