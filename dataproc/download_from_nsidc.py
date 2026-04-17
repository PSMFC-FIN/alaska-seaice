"""
Title: NOAA NSIDC Sea Ice Data Downloader
Author: Sunny Bak Hospital
Modified: April 16, 2026

Description:
    Downloads daily sea ice concentration NetCDF files from the NOAA/NSIDC
    HTTPS server and caches them locally. Run this script once (or to update)
    before running compute_annualized_timeseries.py.

    CDR  (1978-present, finalized): G02202_V6
    NRT  (current year): G10016_V4

    Directory layout created:
        data/
          cdr/    <- G02202_V6 daily files
          nrt/    <- G10016_V4 daily files

Usage:
    python download_nsidc.py
    python download_nsidc.py --cdr-only
    python download_nsidc.py --nrt-only
    python download_nsidc.py --start-year 2020           # from 2020 to product max
    python download_nsidc.py --start-year 2020 --end-year 2022  # 2020–2022 only

Dependencies:
    requests, tqdm
"""

import argparse
import os
import re
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# NSIDC server configuration
# ---------------------------------------------------------------------------

CDR_BASE = "https://noaadata.apps.nsidc.org/NOAA/G02202_V6/north/daily/"
NRT_BASE = "https://noaadata.apps.nsidc.org/NOAA/G10016_V4/north/daily/"

# Filename patterns (NSIDC naming conventions)
# CDR  : seaice_conc_daily_nh_YYYYMMDD_*.nc
# NRT  : seaice_conc_daily_nh_YYYYMMDD_*.nc  (same pattern, different product)  
FILE_PATTERN = re.compile(r'sic_psn25_\d{8}_[^"]+\.nc')   
# Year boundaries
CDR_YEAR_RANGE = (1984, 1984)   # CDR is finalized through 2024
# NRT_YEAR_RANGE = (2026, 2026)   # NRT covers the most recent / current year
LOCAL_DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_remote_files(base_url: str, year: int) -> list[str]:
    """
    Parses the NSIDC directory listing for a given year and returns a list
    of matching NetCDF filenames.

    Args:
        base_url (str): Base HTTPS directory URL.
        year (int)    : Year to fetch.

    Returns:
        list[str]: Filenames (not full URLs) available for that year.
    """
    url = f"{base_url}{year}/"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  WARNING: Could not list {url} — {e}")
        return []

    # Parse hrefs from the directory listing HTML
    return FILE_PATTERN.findall(resp.text)


def download_file(url: str, dest_path: Path, chunk_size: int = 1 << 20):
    """
    Downloads a single file from url to dest_path with a progress bar.
    Skips if the file already exists and has a non-zero size.

    Args:
        url (str)         : Full URL to the file.
        dest_path (Path)  : Local destination path.
        chunk_size (int)  : Download chunk size in bytes (default 1 MB).
    """
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return  # Already cached

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest_path, "wb") as fh, tqdm(
                desc=dest_path.name,
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    fh.write(chunk)
                    bar.update(len(chunk))
    except requests.RequestException as e:
        print(f"  ERROR downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()   # Remove partial file


def download_product(base_url: str, year_range: tuple, local_dir: Path,
                     start_year: int = None, end_year: int = None):
    """
    Downloads all monthly files for a product over a range of years.

    Args:
        base_url (str)    : NSIDC base URL for the product.
        year_range (tuple): (first_year, last_year) inclusive — product bounds.
        local_dir (Path)  : Local directory to save files.
        start_year (int)  : If set, clamp the lower bound to this year.
        end_year (int)    : If set, clamp the upper bound to this year.
                            Defaults to the product's last year.
    """
    first, last = year_range
    if start_year is not None:
        first = max(first, start_year)
    if end_year is not None:
        last = min(last, end_year)

    for year in range(first, last + 1):
        print(f"  Year {year} …", end=" ", flush=True)
        filenames = list_remote_files(base_url, year)

        if not filenames:
            print("no files found / skipped")
            continue

        year_dir = local_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        new_count = 0
        for fname in filenames:
            dest = year_dir / fname
            if not (dest.exists() and dest.stat().st_size > 0):
                download_file(f"{base_url}{year}/{fname}", dest)
                new_count += 1

        status = f"{len(filenames)} files" + (
            f" ({new_count} new)" if new_count else " (all cached)"
        )
        print(status)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Download NSIDC SIC monthly data.")
    p.add_argument("--cdr-only",   action="store_true",
                   help="Download CDR data only.")
    p.add_argument("--nrt-only",   action="store_true",
                   help="Download NRT data only.")
    p.add_argument("--start-year", type=int, default=None,
                   help="Download from this year onward (within each product's range).")
    p.add_argument("--end-year",   type=int, default=None,
                   help="Download up to and including this year (optional). "
                        "Defaults to the product's maximum year.")
    return p.parse_args()


def main():
    args = parse_args()

    # Validate year arguments
    if args.start_year and args.end_year and args.start_year > args.end_year:
        print(f"ERROR: --start-year ({args.start_year}) must be <= --end-year ({args.end_year})")
        sys.exit(1)

    do_cdr = not args.nrt_only
    do_nrt = not args.cdr_only

    if do_cdr:
        cdr_dir = LOCAL_DATA_DIR / "cdr"
        print(f"\nDownloading CDR (G02202_V6) → {cdr_dir}")
        download_product(CDR_BASE, CDR_YEAR_RANGE, cdr_dir,
                         start_year=args.start_year,
                         end_year=args.end_year)

    if do_nrt:
        nrt_dir = LOCAL_DATA_DIR / "nrt"
        print(f"\nDownloading NRT (G10016_V4) → {nrt_dir}")
        download_product(NRT_BASE, NRT_YEAR_RANGE, nrt_dir,
                         start_year=args.start_year,
                         end_year=args.end_year)

    print("\nDownload complete.")
    print(f"  CDR files : {LOCAL_DATA_DIR / 'cdr'}")
    print(f"  NRT files : {LOCAL_DATA_DIR / 'nrt'}")
    print("\nNext step: python compute_annualized_timeseries.py")


if __name__ == "__main__":
    main()