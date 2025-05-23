#!/usr/bin/env python3

from contextlib import chdir
import io
import re
import subprocess
import tarfile
from typing import TypeAlias
import zipfile

from bs4 import BeautifulSoup
from loguru import logger
from lxml import etree
import niquests
import polars as pl
from rich.pretty import pprint
from tqdm.rich import tqdm
import typer

from outils_corpus.config import FULL_DATASET, PG_METADATA_DIR, PG_MIRROR, PG_RDF_TARBALL, RAW_DATA_DIR

# Use type aliases to keep the signatures sane...
Book: TypeAlias = dict[str, str]
Corpus: TypeAlias = dict[int, Book]

app = typer.Typer()


def download_pg_listing() -> None:
	"""
	Download PG's listing of French TXT files
	"""
	# NOTE: PG frowns upon crawling, they instead offer tools to download specific subsets of their catalog,
	#       which we're going to make use of.
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

	Returns
	-------
	list[str]
		A list of URLs to a PG zip archive
	"""

	# It's just a bunch of links to zip archives in a minimal HTML page, so, bs4 FTW
	PG_LISTING = PG_MIRROR / "www.gutenberg.org" / "robot"
	books = []
	for root, dirs, files in PG_LISTING.walk():
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


def download_pg_books(books: list[str]) -> None:
	"""
	Download the book archives, and extract their content (a single txt file)

	Parameters
	----------
	books: list[str]
		A list of URLs to a PG zip archive
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


def extract_pg_metadata() -> None:
	"""
	Extract RDF metadata for the subset of the PG catalog we downloaded
	"""

	# NOTE: We use a set, as we may have multiple copies of the same pg ebook number (in different encodings)
	metadata_files = set()
	for root, dirs, files in RAW_DATA_DIR.walk():
		for name in files:
			book = root / name
			if book.suffix == ".txt":
				# We need the PG ebook number, which is basically the first part of the filename's stem
				# (What comes after the dash denotes the character encoding)
				pg_num = book.stem.split("-")[0]
				# Match the RDF tarball directory layout
				metadata_files.add(f"cache/epub/{pg_num}/pg{pg_num}.rdf")
			else:
				logger.opt(colors=True).warning(f"Skipped <red>{book}</red>")

	# pprint(sorted(metadata_files))

	# We only care about our catalog subset
	def catalog_subset(members):
		for tarinfo in members:
			if tarinfo.isreg() and tarinfo.name in metadata_files:
				# Junk the path while we're here...
				yield tarinfo.replace(name=tarinfo.name.split("/")[-1])

	logger.info("Extracting RDF metadata. . .")
	with tarfile.open(PG_RDF_TARBALL) as tar:
		tar.extractall(path=PG_METADATA_DIR, members=catalog_subset(tar), filter="data")

	# Double-check that we didn't miss anything
	extracted_files = set()
	for root, dirs, files in PG_METADATA_DIR.walk():
		for name in files:
			filename = root / name
			extracted_files.add(filename.name)

	wanted_files = {f.split("/")[-1] for f in metadata_files}
	logger.info(f"Wanted {len(wanted_files)} files, got {len(extracted_files)}")
	# Print the difference
	# pprint(wanted_files - extracted_files)


def dedupe_pg_books() -> Corpus:
	"""
	Build a map of the "best" (in terms of character encoding) file for each PG book number we've got
	"""

	books = {}
	# Start from the metadata
	for root, dirs, files in PG_METADATA_DIR.walk():
		for name in files:
			filename = root / name
			# Skip the leading "pg"
			pg_num = filename.stem[2:]

			# Build a list of potential variants, in reverse order of priority
			variants = [
				("ascii", pg_num + ".txt"),
				("iso-8859-1", pg_num + "-8" + ".txt"),
				("utf-8", pg_num + "-0" + ".txt"),
			]
			# Use a more appropriate datatype for pg_num from here on out
			pg_num = int(pg_num)
			# Check 'em, and store that in a dict, so the best one "wins"
			for variant in variants:
				encoding, path = variant
				if (RAW_DATA_DIR / path).exists():
					books[pg_num] = {"encoding": encoding, "path": path}

			# Warn if we couldn't find a match
			if pg_num not in books:
				logger.opt(colors=True).warning(f"Couldn't find a file for book number <red>{pg_num}</red>")

	logger.info(f"Found {len(books)} unique books")
	# pprint(books)

	return books


def parse_pg_metadata(books: Corpus) -> Corpus:
	"""
	Parse PG metadata for our books, and extract authors & publication dates from it.

	Parameters
	----------
	books: Corpus
		A corpus as generated by `dedupe_pg_books`

	Returns
	-------
	Corpus
		The same corpus, enriched with author & publication date information (if any)
	"""

	# Compile te REs we'll need to sift through the summaries.
	# Thankfully, they're auto-generated, so the layout is pretty stable.
	summary_pattern = re.compile(r"(published|written) (.*?) (century|Middle Ages)\.")
	century_pattern = re.compile(r"(\d{1,2})(th|st|nd)")

	for pg_num, book in books.items():
		# Parse the matching RDF file
		metadata_file = PG_METADATA_DIR / ("pg" + str(pg_num) + ".rdf")
		logger.opt(colors=True).info(f"Parsing <blue>{metadata_file.name}</blue>")
		tree = etree.parse(metadata_file)

		# Pull the title
		title = tree.findtext(".//{http://purl.org/dc/terms/}title")
		book.update(title=title)

		# Find the authors and their dates
		authors = {}
		for creator in tree.iterfind(".//{http://purl.org/dc/terms/}creator"):
			for agent in creator.iterfind("{http://www.gutenberg.org/2009/pgterms/}agent"):
				author = agent.findtext("{http://www.gutenberg.org/2009/pgterms/}name")
				birth = agent.findtext("{http://www.gutenberg.org/2009/pgterms/}birthdate")
				death = agent.findtext("{http://www.gutenberg.org/2009/pgterms/}deathdate")

				# NOTE: PG doesn't provide the original publiucation date,
				#       as the content may be composited from *different* sources.
				#       Since *we* need a date that roughly matches the publication date,
				#       we'll use the date of the author's death (or it's birth + 15 barring that).
				if death:
					authors[author] = int(death)
				elif birth:
					authors[author] = int(birth) + 15
				else:
					authors[author] = 0
					logger.opt(colors=True).warning(f"No dates for author <cyan>{author}</cyan>")

		# If it's a translation, prefer the translator's dates, as we care about the actual language of *this* text
		translators = {}
		for trl in tree.iterfind(".//{http://id.loc.gov/vocabulary/relators/}trl"):
			for agent in trl.iterfind("{http://www.gutenberg.org/2009/pgterms/}agent"):
				translator = agent.findtext("{http://www.gutenberg.org/2009/pgterms/}name")
				birth = agent.findtext("{http://www.gutenberg.org/2009/pgterms/}birthdate")
				death = agent.findtext("{http://www.gutenberg.org/2009/pgterms/}deathdate")

				if death:
					translators[translator] = int(death)
				elif birth:
					translators[translator] = int(birth) + 15
				else:
					logger.opt(colors=True).warning(f"No dates for translator <cyan>{translator}</cyan>")

		# Keep the latest date
		if translators:
			translators = sorted(translators.items(), key=lambda item: -item[1])
			most_recent = translators[0]
			tl, year = most_recent
			# Use the (main, i.e., first) author name for our metadata, if any
			if authors:
				book.update(author=list(authors)[0], year=year)
			else:
				book.update(author=tl, year=year)
		elif authors:
			authors = sorted(authors.items(), key=lambda item: -item[1])
			most_recent = authors[0]
			auth, year = most_recent
			book.update(author=auth, year=year)
		else:
			logger.opt(colors=True).warning(f"No dates for book <red>{pg_num}</red>")

		# If there's an indication in the summary, trust that
		summary = tree.findtext(".//{http://www.gutenberg.org/2009/pgterms/}marc520")
		if summary:
			for m in re.finditer(summary_pattern, summary):
				if m.group(3) == "Middle Ages":
					# Easy enough ;p
					book.update(century=1400)
					logger.opt(colors=True).info(
						"Salvaged a publication century (<green>Middle Ages</green>) from the description"
					)
				else:
					cm = re.search(century_pattern, m.group(2))
					if cm:
						# Convert to a date
						century = (int(cm.group(1)) - 1) * 100
						book.update(century=century)
						logger.opt(colors=True).info(
							f"Salvaged a publication century (<green>{century}</green>) from the description"
						)

	pprint(books)
	return books


def build_full_dataset(books: Corpus) -> None:
	"""
	Finalize data extraction, and dump the full dataset to disk, in parquet.

	Parameters
	----------
	books: Corpus
		An enriched corpus as generated by `parse_pg_metadata`
	"""

	dataset = []
	for pg_num, book in tqdm(books.items()):
		logger.opt(colors=True).info(f"Processing book number <blue>{pg_num}</blue>")

		# If we do not have an author, the document wasn't a book (e.g., might have been a music sheet or something)
		if "author" not in book:
			logger.opt(colors=True).warning("Not a book, skipping it")
			continue

		# If we have a century already set, use that, otherwise, round the year down to the nearest multiple of a hundred
		century = book.get("century")
		if not century:
			century = book.get("year")
			if century:
				century = century // 100 * 100
		if not century:
			logger.opt(colors=True).warning("No dates, skipping it")
			continue

		# We'll need the basic metadata, too
		title = book.get("title")
		author = book.get("author")

		# Pull the full text
		txt_file = RAW_DATA_DIR / book.get("path")
		# NOTE: We assumed unsuffixed files used ASCII, but it's not always the case...
		if book.get("encoding") == "ascii":
			for encoding in ["ascii", "iso-8859-1", "utf-8"]:
				try:
					data = txt_file.read_text(encoding=encoding)
				except UnicodeDecodeError:
					pass
		elif book.get("encoding") == "utf-8":
			# Similarly, some books have wrong metadata...
			for encoding in ["utf-8", "iso-8859-1"]:
				try:
					data = txt_file.read_text(encoding=encoding)
				except UnicodeDecodeError:
					pass
		else:
			data = txt_file.read_text(encoding=book.get("encoding"))
		if not data:
			logger.opt(colors=True).warning("Failed to decode text file, skipping it")
			continue

		# Strip the PG metadata (it's not *completely* standard, so here be some kludges)
		potential_headers = [
			"*** START OF TH",
			"***START OF TH",
			"This etext was produced by",
			"This Etext is",
		]
		for header in potential_headers:
			try:
				start_idx = data.index(header)
				break
			except ValueError:
				continue
		start_idx = data.index("\n", start_idx) + 1

		potential_footers = [
			"*** END OF TH",
			"***END OF TH",
			"This etext was produced by",
			"This Etext is",
		]
		for footer in potential_footers:
			try:
				end_idx = data.rindex(footer)
				break
			except ValueError:
				continue
		end_idx -= 1

		data = data[start_idx:end_idx]

		# Build our dataset as a list of Book dicts in our final "schema"
		dataset.append(
			{
				"pg_num": pg_num,
				"title": title,
				"author": author,
				"century": century,
				"text": data,
			}
		)

	# Build a lazy Polars DataFrame out of that, and dump it to disk
	lf = pl.LazyFrame(
		dataset, schema={"pg_num": pl.UInt16, "title": None, "author": None, "century": pl.UInt16, "text": None}
	)
	lf.sink_parquet(FULL_DATASET)


@app.command()
def main():
	# Download the filtered catalog
	download_pg_listing()
	# Parse the listing to extract the download links
	urls = parse_pg_listing()
	# Download the books
	download_pg_books(urls)
	# Extract the RDF metadata for our downloaded books
	extract_pg_metadata()
	# Find the best file for each book number
	books = dedupe_pg_books()
	# Parse metadata for each book number
	books = parse_pg_metadata(books)
	# Finally, create the full, final parquet dataset
	build_full_dataset(books)


if __name__ == "__main__":
	app()
