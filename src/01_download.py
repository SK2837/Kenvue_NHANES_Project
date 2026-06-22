"""Download NHANES 2017-2018 .XPT files from CDC servers."""

import logging
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# CDC changed URL pattern; new format uses year and lowercase .xpt extension
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

MODULES = {
    "DEMO_J": "Demographics, survey weights, strata, PSU",
    "BMX_J": "BMI, waist circumference",
    "BIOPRO_J": "Full biochemistry panel including ALT (outcome)",
    "PBCD_J": "Blood lead, cadmium, mercury",
    "ALQ_J": "Alcohol use",
    "DIQ_J": "Diabetes diagnosis",
    "SMQ_J": "Smoking",
    "HEPB_S_J": "Hepatitis B serology",
    "HEPC_J": "Hepatitis C serology",
}


def download_xpt(module_code: str) -> Path:
    """Download a single NHANES .XPT file, skipping if already cached and valid."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{module_code}.XPT"

    if dest.exists():
        # Validate cached file is binary XPT, not an HTML soft-404
        with dest.open("rb") as f:
            header = f.read(8)
        if header[:5] == b"<html" or header[:9] == b"<!DOCTYPE":
            logger.warning("Corrupt cached file for %s — deleting and re-downloading", module_code)
            dest.unlink()
        else:
            size_kb = dest.stat().st_size / 1024
            logger.info("Cached  %s  (%.1f KB)", module_code, size_kb)
            return dest

    url = f"{BASE_URL}/{module_code}.xpt"
    logger.info("Downloading %s from %s ...", module_code, url)

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        raise ValueError(
            f"CDC returned HTML for {module_code} (soft 404). "
            f"URL may be wrong: {url}"
        )

    dest.write_bytes(response.content)
    size_kb = len(response.content) / 1024
    logger.info("Saved   %s  (%.1f KB)", module_code, size_kb)
    return dest


def main() -> None:
    logger.info("NHANES 2017-2018 download — saving to %s", RAW_DIR)
    failed = []

    for code, description in MODULES.items():
        try:
            download_xpt(code)
        except (requests.HTTPError, ValueError) as exc:
            logger.error("FAILED %s (%s): %s", code, description, exc)
            failed.append(code)

    if failed:
        logger.error("Failed modules: %s", failed)
        sys.exit(1)

    logger.info("All modules downloaded successfully.")


if __name__ == "__main__":
    main()
