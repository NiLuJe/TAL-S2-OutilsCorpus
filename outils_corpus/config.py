from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# PG Mirror
PG_MIRROR = EXTERNAL_DATA_DIR / "pg"
# PG RDF archive
PG_RDF_TARBALL = PG_MIRROR / "rdf-files.tar.bz2"
PG_METADATA_DIR = EXTERNAL_DATA_DIR / "metadata"

# Full dataset
FULL_DATASET = PROCESSED_DATA_DIR / "pg-fr-books-full.parquet"

# Visualizations
CATEGORIES_VIZ = FIGURES_DIR / "cat-distrib.html"

# SetFit
SETFIT_DIR = MODELS_DIR / "setfit-trained"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
	from tqdm.rich import tqdm

	logger.remove(0)
	logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
	pass
