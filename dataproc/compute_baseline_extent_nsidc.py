#!/usr/bin/env python
"""
Title: Baseline Sea Ice Extent Computation for Alaska Regions
Author: Sunny Bak Hospital
Modified: April 16, 2026

Description:
    Computes daily sea ice extent statistics (mean and std) over a baseline
    period for multiple Alaska regions, using locally cached NSIDC CDR daily
    NetCDF files and the SIC25k class from the `sic` module.

    Logic:
        For each year in the baseline period, daily SIC values are binarised
        at the 0.15 threshold and summed spatially to produce a daily extent
        value (km2). Once all years are collected, mean and std are computed
        grouped by month-day. This is the statistically valid approach: the
        threshold is applied to each observation before averaging, so extent
        statistics reflect real ice/no-ice variability across years.

    Memory efficiency:
        - Cell area is loaded once and clipped once per region, not reloaded
          every year.
        - SIC25k is instantiated per calendar year, opening only that year's
          files. After computation, spatial data is discarded and only the
          small 1-D daily extent time series is kept in memory.

    Data sources (local, downloaded by download_nsidc.py):
        CDR : data/cdr/YYYY/   (G02202_V6 daily)
        Area: resources/ref_files/NSIDC0771_CellArea_PS_N25km_v1.0.nc

    Output: one CSV per region -> bs_extent_<RegionName>.csv
            columns: month, day, seaice_extent_mean, seaice_extent_std, month_day

Regions:
    AlaskanArctic, NorthernBering, EasternBering, SoutheasternBering

Usage:
    python compute_baseline_extent_nsidc.py                           # defaults: 1985-2020
    python compute_baseline_extent_nsidc.py --start-year 1985 --end-year 2020
    python compute_baseline_extent_nsidc.py --start-year 1991         # end-year defaults to 2020
"""

import argparse
import gc
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr

from sic import SIC25k, clip_data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CRS      = 'epsg:3411'        # NSIDC Polar Stereographic North (Hughes 1980)
VAR_NAME = 'cdr_seaice_conc'  # !CHECK Variable name in CDR daily files

CDR_DATA_DIR = Path("data/cdr")
AREA_NC_PATH = Path("resources/ref_files/NSIDC0771_CellArea_PS_N25km_v1.0.nc")
RESOURCE_DIR = Path("resources/akmarineeco")

REGIONS = {
    'AlaskanArctic':        'arctic_sf.shp',
    # 'NorthernBering':       'nbering_sf.shp',
    # 'EasternBering':        'ebering_sf.shp',
    # 'SoutheasternBering':   'se_bering_sf.shp',
}

DEFAULT_START_YEAR = 1985
DEFAULT_END_YEAR   = 2020


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute baseline daily sea ice extent statistics."
    )
    p.add_argument(
        '--start-year', type=int, default=DEFAULT_START_YEAR,
        help=f"First year of the baseline period (default: {DEFAULT_START_YEAR})."
    )
    p.add_argument(
        '--end-year', type=int, default=DEFAULT_END_YEAR,
        help=f"Last year of the baseline period, inclusive (default: {DEFAULT_END_YEAR})."
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Data must be downloaded prior to running this script, 
# so check for expected paths and exit with a  message if not found.
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


def load_area(start_year: int, crs: str) -> xr.DataArray:
    """
    Loads the full grid-cell area DataArray once from disk.
    Uses the first available year directory just to initialise SIC25k,
    which is needed to set up the spatial metadata on the area file.

    Args:
        start_year (int): First year — used to find a valid data directory.
        crs (str)       : CRS string.

    Returns:
        xr.DataArray: Full (unclipped) grid-cell area DataArray.
    """
    year = start_year
    while year <= DEFAULT_END_YEAR:
        year_dir = CDR_DATA_DIR / str(year)
        if year_dir.exists():
            break
        year += 1
    else:
        print("ERROR: No valid CDR year directory found to initialise area.")
        sys.exit(1)

    sic_tmp = SIC25k(data_dirs=year_dir, varname=VAR_NAME, crs=crs)
    sic_tmp.load_area_local(str(AREA_NC_PATH)) # load grid area data from the local file
    area = sic_tmp.area
    del sic_tmp
    gc.collect() #garbage collector to free memory immediately
    return area


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    start_year = args.start_year
    end_year   = args.end_year

    if start_year > end_year:
        print(f"ERROR: --start-year ({start_year}) must be <= --end-year ({end_year})")
        sys.exit(1)

    validate_paths()

    print(f"Baseline period: {start_year} - {end_year}")
    print(f"Regions: {list(REGIONS.keys())}\n")

    # Load area once — it's the same grid for all regions and all years
    print("Loading grid-cell area ... ", end="")
    area_full = load_area(start_year, CRS)
    print("done.\n")

    # ------------------------------------------------------------------
    # Loop over regions
    # ------------------------------------------------------------------
    for region_name, shp_file in REGIONS.items():
        shp_path = RESOURCE_DIR / shp_file

        if not shp_path.exists():
            print(f"WARNING: Shapefile not found for '{region_name}': {shp_path}. Skipping.")
            continue

        print(f"{'='*60}")
        print(f"Region: {region_name}")
        print(f"{'='*60}")

        # Convert the shapefile to the same CRS as the SIC data
        alaska_shp_proj = gpd.read_file(shp_path).to_crs(CRS)

        # Clip area once per region — reused for every year
        area_clipped = clip_data(area_full, alaska_shp_proj)

        extents = []   # collects 1-D daily extent DataArrays across years

        # --------------------------------------------------------------
        # Loop over years — one SIC25k per calendar year
        # --------------------------------------------------------------
        for year in range(start_year, end_year + 1):
            year_dir = CDR_DATA_DIR / str(year)

            if not year_dir.exists():
                print(f"  {year}: directory not found ({year_dir}), skipping.")
                continue

            print(f"  {year}", end=" ... ")

            try:
                # Load data into SIC25k instance for this year only (opens only that year's files)
                sic = SIC25k(
                    data_dirs=year_dir,
                    varname=VAR_NAME,
                    crs=CRS,
                )

                # Clip SIC to region for this year
                ds = clip_data(
                    sic.ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31')),
                    alaska_shp_proj,
                )

                if ds.time.size == 0:
                    print("no data, skipping.")
                    continue

                # Convert to binary values per pixel based on threshold,
                # multiply by area, sum spatially
                # to obtain 1-D DataArray of daily extent values (km2)
                sic_bin = sic.format_sic(ds, threshold=0.15)
                ext     = sic.compute_extent_km(sic_bin, area_clipped)

                # Store only the small 1-D result; discard spatial data
                extents.append(ext)

                print(f"{ds.time.size} days.")

            except Exception as e:
                print(f"ERROR: {e}")

            finally:
                try:
                    del sic, ds, sic_bin, ext
                except NameError:
                    pass
                gc.collect()

        if not extents:
            print(f"No data collected for {region_name}, skipping.\n")
            del alaska_shp_proj, area_clipped
            continue

        # --------------------------------------------------------------
        # Combine all years into one continuous daily time series
        # --------------------------------------------------------------
        combined = xr.concat(extents, dim='time')

        ext_df = combined.to_dataframe().reset_index()

        if 'spatial_ref' in ext_df.columns:
            ext_df = ext_df.drop(columns=['spatial_ref'])

        ext_df['time']  = pd.to_datetime(ext_df['time'])
        ext_df['month'] = ext_df['time'].dt.month
        ext_df['day']   = ext_df['time'].dt.day
        ext_df = ext_df.drop(columns=['time'])

        # Evaluate the data before grouping to check for any anomalies
        print("\nSample of combined daily extent data before grouping:")
        ext_df.to_csv(f'check_ext_df_{region_name}_{year}.csv', index=False)
        # --------------------------------------------------------------
        # Group by month-day: mean and std across years
        # --------------------------------------------------------------
        stats = (ext_df
                 .groupby(['month', 'day'])['seaice_extent']
                 .agg(['mean', 'std'])
                 .reset_index()
                 .round(2)
                 .rename(columns={
                     'mean': 'seaice_extent_mean',
                     'std':  'seaice_extent_std',
                 }))

        stats['month_day'] = stats.apply(
            lambda r: f"{int(r['month']):02d}-{int(r['day']):02d}", axis=1
        )

        out_csv = f"bs_extent_{region_name}.csv"
        stats.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}  ({len(stats)} month-day rows)\n")

        del alaska_shp_proj, area_clipped, extents, combined, ext_df, stats
        gc.collect()

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()