#!/usr/bin/env python3

from contextlib import chdir
import io
import subprocess
import zipfile

from bs4 import BeautifulSoup
from loguru import logger
import niquests
from rich.pretty import pprint
from tqdm.rich import tqdm
import typer

from outils_corpus.config import PG_MIRROR, PG_RDF_TARBALL, RAW_DATA_DIR

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
		for name in tqdm(files):
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


def download_pg_books(books: str):
	"""
	Download the book archives, and extract their content (a single txt file)
	"""

	# Work in a session to reuse connections
	with niquests.Session() as s:
		# Might take a while, so, show pretty colors while we wait ;).
		for book in tqdm(books):
			filename = book.split("/")[-1]
			logger.opt(colors=True).info(f"Downloading <blue>{book}</blue>")
			r = s.get(book)
			# Bail on that URL if the request failed
			if r.status_code != niquests.codes.ok:
				logger.opt(colors=True).warning(f"Failed to download <red>{book}</red>")
				continue
			# Keep the zip data in memory, we only want to flush the archive's content to disk...
			logger.opt(colors=True).info(f"Extracting <green>{filename}</green>")
			try:
				with zipfile.ZipFile(io.BytesIO(r.content), "r") as zf:
					# We're getting at actual actionable data, chuck it in data/raw
					zf.extractall(path=RAW_DATA_DIR)
			except zipfile.BadZipFile:
				logger.opt(colors=True).warning(f"Failed to unpack <red>{filename}</red>")


@app.command()
def main():
	# Download the filtered catalog
	# download_pg_listing()
	# Parse the listing to extract the download links
	books = parse_pg_listing()
	# Download the books
	download_pg_books(books)


if __name__ == "__main__":
	app()
