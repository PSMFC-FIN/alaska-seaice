"""
Title: Annualized Sea Ice Extent Computation for Alaska Regions
Author: Sunny Bak Hospital
Modified: April 16, 2026

Description:
    Computes annualized sea ice extent for multiple Alaska regions from 1985 to 2025,
    using locally cached NSIDC CDR daily NetCDF files (downloaded via download_nsidc.py)
    and the SIC25k class from the `sic` module.

    For efficiency, SIC25k is instantiated per annual window, opening only the
    two year-directories needed for that Sep-Aug period rather than loading
    all 40 years at once. This keeps memory usage low.

    Each year's result is written to CSV immediately after computation so that
    progress is never lost if the script crashes or is interrupted.
    If the output CSV already exists, the script resumes from where it left off,
    skipping years already present in the file.

    Data sources (local, downloaded by download_nsidc.py):
        CDR  1985-2025 : data/cdr/YYYY/   (G02202_V6 daily)
        Area           : resources/ref_files/NSIDC0771_CellArea_PS_N25km_v1.0.nc

    Annual window definition:
        Year N covers Sep 1 of (N-1) through Aug 31 of N.
        e.g.  Year 1985 = 1984-09-01 to 1985-08-31
              Year 2025 = 2024-09-01 to 2025-08-31

    The annualized extent is the mean of daily extents within the Sep-Aug window.

    Output: one CSV per region -> annualized_extent_<RegionName>.csv
            columns: region, year, extent_km2, n_days

Regions:
    AlaskanArctic, NorthernBering, EasternBering, SoutheasternBering

Dependencies:
    sic (local module), pandas, geopandas, gc

Usage:
    # Download data first (one-time / update step):
    python download_nsidc.py --cdr-only

    # Then compute:
    python compute_annualized_timeseries.py
"""

import gc
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

from sic import SIC25k

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CRS      = 'epsg:3411'          # Polar Stereographic North (Hughes 1980)
VAR_NAME = 'cdr_seaice_conc'    # Variable name in CDR daily files

CDR_DATA_DIR = Path("data/cdr")
AREA_NC_PATH = Path("resources/ref_files/NSIDC0771_CellArea_PS_N25km_v1.0.nc")
SHP_DIR      = Path("resources/akmarineeco")

FIRST_YEAR = 1985
LAST_YEAR  = 2025

REGIONS = {
    'AlaskanArctic':      'arctic_sf.shp',
    'NorthernBering':     'nbering_sf.shp',
    'EasternBering':      'ebering_sf.shp',
    'SoutheasternBering': 'se_bering_sf.shp',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_data_dirs(year: int) -> list[Path]:
    """
    Returns the two CDR year-directories covering Sep (year-1) to Aug (year).
    Warns if either directory does not exist.
    """
    dirs = [
        CDR_DATA_DIR / str(year - 1),   # Sep-Dec
        CDR_DATA_DIR / str(year),        # Jan-Aug
    ]
    existing = [d for d in dirs if d.exists()]
    missing  = [d for d in dirs if not d.exists()]
    if missing:
        print(f"    WARNING: missing directories: {[str(d) for d in missing]}")
    return existing


def get_completed_years(out_csv: Path) -> set:
    """
    Returns the set of years already written to the output CSV.
    Used to skip years when resuming an interrupted run.
    """
    if not out_csv.exists():
        return set()
    try:
        df = pd.read_csv(out_csv)
        return set(df['year'].tolist())
    except Exception:
        return set()


def validate_paths():
    errors = []
    if not CDR_DATA_DIR.exists():
        errors.append(f"CDR data directory not found: {CDR_DATA_DIR}")
    if not AREA_NC_PATH.exists():
        errors.append(f"Cell area file not found: {AREA_NC_PATH}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        print("\nRun `python download_nsidc.py --cdr-only` to download the data first.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    validate_paths()

    for region_name, shp_file in REGIONS.items():
        shp_path = SHP_DIR / shp_file

        if not shp_path.exists():
            print(f"WARNING: Shapefile not found for '{region_name}': {shp_path}. Skipping.")
            continue

        print(f"{'='*60}")
        print(f"Region: {region_name}")
        print(f"{'='*60}")

        alaska_shp_proj = gpd.read_file(shp_path).to_crs(CRS)

        out_csv = Path(f"annualized_extent_{region_name}.csv")

        # Check which years are already done so we can resume if interrupted
        completed_years = get_completed_years(out_csv)
        if completed_years:
            print(f"  Resuming — {len(completed_years)} years already in {out_csv}")

        # Write CSV header only if starting fresh
        write_header = not out_csv.exists()

        for year in range(FIRST_YEAR, LAST_YEAR + 1):

            if year in completed_years:
                print(f"  {year}  already computed, skipping.")
                continue

            date_start = f"{year - 1}-09-01"
            date_end   = f"{year}-08-31"

            dirs = get_data_dirs(year)
            if not dirs:
                print(f"  {year}: no data directories found, skipping.")
                continue

            print(f"  {year}  {date_start} -> {date_end}", end=" ... ")

            try:
                sic = SIC25k(
                    data_dirs=dirs,
                    varname=VAR_NAME,
                    crs=CRS,
                )
                sic.load_area_local(str(AREA_NC_PATH))

                ds, area = sic.subset_dim([date_start, date_end], alaska_shp_proj)

                if ds.time.size == 0:
                    print("no data in window, skipping.")
                    continue

                sic_bin     = sic.format_sic(ds, threshold=0.15)
                ext         = sic.compute_extent_km(sic_bin, area)
                annual_mean = float(ext.mean().values)

                # Write this year's result immediately
                row = pd.DataFrame([{
                    'region':     region_name,
                    'year':       year,
                    'extent_km2': annual_mean,
                    'n_days':     int(ds.time.size),
                }])
                row.to_csv(out_csv, mode='a', index=False, header=write_header)
                write_header = False  # header written once only

                print(f"extent = {annual_mean:,.0f} km2  ({ds.time.size} days)")

            except Exception as e:
                print(f"ERROR: {e}")

            finally:
                try:
                    del sic, ds, sic_bin, ext
                except NameError:
                    pass
                gc.collect()

        print(f"\nDone with {region_name} -> {out_csv}\n")

        del alaska_shp_proj
        gc.collect()

    print("All regions complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()