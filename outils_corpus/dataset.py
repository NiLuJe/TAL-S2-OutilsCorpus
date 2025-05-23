#!/usr/bin/env python3

from contextlib import chdir
import subprocess

from bs4 import BeautifulSoup
from loguru import logger
from rich.pretty import pprint
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


def parse_pg_listing() -> list[str]:
	"""
	Parse the listing downloaded by `download_pg_listing` to get at the actual content.
	"""

	# It's just a bunch of links in a minimal HTML page, so, bs4 FTW
	PG_LISTING = PG_MIRROR / "www.gutenberg.org" / "robot"
	books = []
	for root, dirs, files in PG_LISTING.walk(top_down=False):
		for name in files:
			input_html = root / name
			logger.opt(colors=True).info(f"Parsing <blue>{input_html.name}</blue>")
			with open(input_html) as fp:
				soup = BeautifulSoup(fp, "lxml")
				for link in soup.find_all("a"):
					href = link.get("href")
					# Skip the navigation links
					if href.endswith(".zip"):
						books.append(href)

	print(f"Found {len(books)} books")
	pprint(books)
	return books


@app.command()
def main():
	# Download the filtered catalog
	# download_pg_listing()
	parse_pg_listing()


if __name__ == "__main__":
	app()
