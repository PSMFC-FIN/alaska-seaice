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
    all 40 years at once. This keeps the Dask task graph small and memory usage low.

    Data sources (local, downloaded by download_nsidc.py):
        CDR  1985-2025 : data/cdr/YYYY/   (G02202_V6 daily)
        Area           : dataproc/resources/ref_files/NSIDC0771_CellArea_PS_N25km_v1.0.nc

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
    sic (local module), numpy, pandas, geopandas, dask, gc

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
import numpy as np
import pandas as pd
from dask.distributed import Client

from sic import SIC25k

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CRS      = 'epsg:3413'          # Polar Stereographic North
VAR_NAME = 'cdr_seaice_conc'    # Variable name in CDR daily files

# Local data directory (populated by download_nsidc.py --cdr-only)
CDR_DATA_DIR = Path("data/cdr") # G02202_V6 daily, organised as data/cdr/YYYY/
AREA_NC_PATH = Path("resources/ref_files/NSIDC0771_CellArea_PS_N25km_v1.0.nc")

# Shapefile directory
SHP_DIR = Path("resources/akmarineeco")

# Year range (inclusive)
FIRST_YEAR = 1985
LAST_YEAR  = 2025

# Regions: output_name -> shapefile name
REGIONS = {
    'AlaskanArctic':        'arctic_sf.shp',
    # 'NorthernBering':     'nbering_sf.shp',
    # 'EasternBering':      'ebering_sf.shp',
    # 'SoutheasternBering': 'se_bering_sf.shp',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_data_dirs(year: int) -> list[Path]:
    """
    Returns the two CDR year-directories that cover the Sep (year-1) to
    Aug (year) window. Warns if either directory does not exist.

    Args:
        year (int): Annual window end-year.

    Returns:
        list[Path]: Existing year-directories for this window.
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


def validate_paths():
    """Checks that required top-level data paths exist before running."""
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
    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    validate_paths()

    # ------------------------------------------------------------------
    # 2. Start Dask
    # ------------------------------------------------------------------
    client = Client()
    print(f"Dask dashboard: {client.dashboard_link}\n")

    # ------------------------------------------------------------------
    # 3. Loop over regions
    # ------------------------------------------------------------------
    for region_name, shp_file in REGIONS.items():
        shp_path = SHP_DIR / shp_file

        if not shp_path.exists():
            print(f"WARNING: Shapefile not found for '{region_name}': {shp_path}. Skipping.")
            continue

        print(f"{'='*60}")
        print(f"Region: {region_name}")
        print(f"{'='*60}")

        alaska_shp      = gpd.read_file(shp_path)
        alaska_shp_proj = alaska_shp.to_crs(CRS)

        extents = []

        # --------------------------------------------------------------
        # 4. Loop over years — one SIC25k object per annual window
        # --------------------------------------------------------------
        for year in range(FIRST_YEAR, LAST_YEAR + 1):
            date_start = f"{year - 1}-09-01"
            date_end   = f"{year}-08-31"

            dirs = get_data_dirs(year)

            if not dirs:
                print(f"  {year}: no data directories found, skipping.")
                continue

            print(f"  {year}  {date_start} -> {date_end}", end=" ... ")

            try:
                # Instantiate fresh SIC25k for this window only
                sic = SIC25k(
                    data_dirs=dirs,
                    varname=VAR_NAME,
                    crs=CRS,
                )
                sic.load_area_local(str(AREA_NC_PATH))

                # Clip to region and time window
                ds, area = sic.subset_dim([date_start, date_end], alaska_shp_proj)

                if ds.time.size == 0:
                    print("no data in window, skipping.")
                    continue

                # Binarise at 0.15 threshold then compute daily extent (km2)
                sic_bin = sic.format_sic(ds, threshold=0.15)
                ext     = sic.compute_extent_km(sic_bin, area)

                # Annualised = mean of daily extents over the Sep-Aug window
                annual_mean = float(ext.mean().values)

                extents.append({
                    'region':     region_name,
                    'year':       year,
                    'extent_km2': annual_mean,
                    'n_days':     int(ds.time.size),
                })

                print(f"extent = {annual_mean:,.0f} km2  ({ds.time.size} days)")

            except Exception as e:
                print(f"ERROR: {e}")

            finally:
                try:
                    del sic, ds, sic_bin, ext
                except NameError:
                    pass
                gc.collect()

        # ------------------------------------------------------------------
        # 5. Save results for this region
        # ------------------------------------------------------------------
        if extents:
            df      = pd.DataFrame(extents)
            out_csv = f"annualized_extent_{region_name}.csv"
            df.to_csv(out_csv, index=False)
            print(f"\nSaved: {out_csv}  ({len(df)} rows)\n")
        else:
            print(f"No results for {region_name}.\n")

        del alaska_shp_proj
        gc.collect()

    print("Done.")
    client.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()