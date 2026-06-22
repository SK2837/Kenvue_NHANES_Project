"""
Download NHANES 2021-2023 .XPT files from CDC servers.

This is the first complete post-COVID collection cycle (Aug 2021 – Aug 2023).
Files use the _L suffix. Data saved to data/raw_2021/ to avoid overwriting
the 2017-2018 (_J) files.

Outputs: data/raw_2021/*.XPT
"""

import logging
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles"
RAW_DIR  = Path(__file__).parent.parent / "data" / "raw_2021"

# ── Core modules (same components as 2017-2018 but _L suffix) ─────────────────
CORE_MODULES = {
    "DEMO_L":    "Demographics, survey weights (WTMEC2YR), strata, PSU",
    "BMX_L":     "Body measures — BMI, waist circumference",
    "BIOPRO_L":  "Biochemistry panel — ALT, AST, ALP, albumin, bilirubin, triglycerides, glucose",
    "PBCD_L":    "Blood lead (LBXBPB), cadmium (LBXBCD), total mercury (LBXTHG)",
    "ALQ_L":     "Alcohol use — drinking frequency and quantity",
    "DIQ_L":     "Diabetes diagnosis (DIQ010)",
    "SMQ_L":     "Smoking history (SMQ020)",
    "HEPB_S_L":  "Hepatitis B surface antigen (LBXHBS)",
    "HEPC_L":    "Hepatitis C antibody (LBXHCR)",
}

# ── COVID-era modules — new variables for Phase 3 hypothesis testing ───────────
COVID_ERA_MODULES = {
    "DPQ_L":  "Depression screener PHQ-9 (DPQ010–DPQ090) — lockdown mental health proxy",
    "PAQ_L":  "Physical activity — work + recreational MET-minutes (inactivity proxy)",
    "SLQ_L":  "Sleep disorders — hours per night (SLD012/SLD013)",
}

ALL_MODULES = {**CORE_MODULES, **COVID_ERA_MODULES}


def download_xpt(module_code: str, description: str) -> Path:
    """Download a single NHANES .XPT file; skip if already cached and valid."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{module_code}.XPT"

    if dest.exists():
        with dest.open("rb") as f:
            header = f.read(9)
        if header[:5] == b"<html" or header[:9] == b"<!DOCTYPE":
            logger.warning("Corrupt cached file for %s — re-downloading", module_code)
            dest.unlink()
        else:
            size_kb = dest.stat().st_size / 1024
            logger.info("Cached   %-14s  %.1f KB  |  %s", module_code, size_kb, description)
            return dest

    url = f"{BASE_URL}/{module_code}.xpt"
    logger.info("Downloading  %-14s  from %s", module_code, url)

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(f"HTTP {e.response.status_code} for {module_code}: {url}") from e

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        raise ValueError(
            f"CDC returned HTML for {module_code} (soft 404). "
            f"Check URL or file availability: {url}"
        )

    dest.write_bytes(response.content)
    size_kb = len(response.content) / 1024
    logger.info("Saved    %-14s  %.1f KB  |  %s", module_code, size_kb, description)
    return dest


def main() -> None:
    logger.info("NHANES 2021-2023 download — saving to %s", RAW_DIR)
    logger.info("Cycle: August 2021 – August 2023  |  File suffix: _L\n")

    failed  = []
    success = []

    logger.info("── Core modules (same as 2017-2018) ──────────────────────────")
    for code, desc in CORE_MODULES.items():
        try:
            download_xpt(code, desc)
            success.append(code)
        except Exception as exc:
            logger.error("FAILED  %-14s  %s", code, exc)
            failed.append((code, str(exc)))

    logger.info("\n── COVID-era modules (new for Phase 3) ────────────────────────")
    for code, desc in COVID_ERA_MODULES.items():
        try:
            download_xpt(code, desc)
            success.append(code)
        except Exception as exc:
            logger.error("FAILED  %-14s  %s", code, exc)
            failed.append((code, str(exc)))

    logger.info("\n── Summary ────────────────────────────────────────────────────")
    logger.info("Downloaded: %d / %d modules", len(success), len(ALL_MODULES))

    if failed:
        logger.error("Failed modules:")
        for code, reason in failed:
            logger.error("  %-14s  %s", code, reason)
        logger.error(
            "\nIf a module is 404, CDC may have not yet published it "
            "or the filename may differ. Check: "
            "https://wwwn.cdc.gov/Nchs/Nhanes/2021-2023/"
        )
        sys.exit(1)

    logger.info("All modules downloaded successfully.")
    logger.info("Next step: run  python src/12_build_dataset_2021.py")


if __name__ == "__main__":
    main()
