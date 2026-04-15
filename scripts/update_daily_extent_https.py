import pandas as pd
import numpy as np
from datetime import date
import geopandas as gpd
from utils import *
import xarray as xr
import requests
from bs4 import BeautifulSoup
import re
import tempfile
import os
from pyproj import CRS

# Data source: NSIDC CDR SIC Near Real Time from https://nsidc.org/data/g10016/versions/4
# File pattern for versiion 4 CDR SIC files (e.g. sic_psn25_20260410_am2_icdr_v04r00.nc)
# UPDATED: data processed and available as of 04/10/2026
NSIDC_BASE = "https://noaadata.apps.nsidc.org/NOAA/G10016_V4/north/daily"
FILE_PATTERN = re.compile(r"sic_psn25_(\d{8})_\w+_icdr_v04r00\.nc")

# Data source: NSIDC CDR SIC Science Quality 
# File pattern for versiion 4 CDR SIC files (e.g. sic_psn25_20241204_F17_v06r00.nc  )
# UPDATED: data processed and available until 04/09/2026
NSIDC_BASE = "https://noaadata.apps.nsidc.org/NOAA/G02202_V6/north/daily"
FILE_PATTERN = re.compile(r"sic_psn25_(\d{8})_\w+_v06r00\.nc")




def list_nsidc_files_for_year(year: int) -> dict:
    """Scrape the NSIDC directory for a year → {YYYYMMDD: full_url}"""
    url = f"{NSIDC_BASE}/{year}/"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    files = {}
    for a in soup.find_all("a", href=True):
        m = FILE_PATTERN.match(os.path.basename(a["href"]))
        if m:
            files[m.group(1)] = url + os.path.basename(a["href"])
    return files


def get_var_data_nsidc(crs: str, var_name: str, date_range: list) -> xr.DataArray:
    """
    replacement for get_var_data() in Utils.py that fetches from NSIDC CDR SIC instead of ERDDAP.
    Returns an xr.DataArray with:
      - dims:  (time, ygrid, xgrid)
      - CRS written via rioxarray
      - values clipped to [0, 1]
      - spatial subset to broad Alaska region
    """
    start = pd.to_datetime(date_range[0])
    end   = pd.to_datetime(date_range[1])

    # Collect available files, one directory listing per year
    available = {}
    for yr in range(start.year, end.year + 1):
        try:
            print(f'Year {yr}: listing files from NSIDC ...')
            available.update(list_nsidc_files_for_year(yr))
        except requests.HTTPError as e:
            print(f"  Warning: could not list NSIDC files for {yr}: {e}")

    needed_dates = pd.date_range(start, end, freq="D")
    arrays = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for dt in needed_dates:
            date_key = dt.strftime("%Y%m%d")
            if date_key not in available:
                print(f"  Skipping {date_key}: no file found on NSIDC")
                continue

            url = available[date_key]
            local_path = os.path.join(tmpdir, os.path.basename(url))

            print(f"  Downloading {os.path.basename(url)} ...")
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)

            print(f"  Processing {os.path.basename(url)} ...")
            ds = xr.open_dataset(local_path)

            # Resolve variable name 
            if var_name in ds:
                da = ds[var_name]
            else:
                candidates = [v for v in ds.data_vars
                              if "seaice" in v.lower() or "sic" in v.lower()
                              or "conc" in v.lower()]
                if not candidates:
                    print(f"  Warning: no sea ice variable found in {date_key}, skipping")
                    ds.close()
                    continue
                print(f"  Note: '{var_name}' not found; using '{candidates[0]}'")
                da = ds[candidates[0]]

            #  Ensure time dim exists  
            if "time" not in da.dims:
                da = da.expand_dims(time=[dt])

            # Spatial subset to broad Alaska/Bering bbox  
            # x: Alaska broad region
            # y: descending in this file so slice high → low
            da = da.sel(
                x=slice(-3837500.0, -287500.0),
                y=slice(4337500.0,   -2337500.0),
            )

            # ── Rename x/y → xgrid/ygrid to match area .nc files  
            da = da.rename({"x": "xgrid", "y": "ygrid"})

            da = da.load()
            ds.close()
            arrays.append(da)

    if not arrays:
        return xr.DataArray()

    combined = xr.concat(arrays, dim="time")

    # ── Write native CRS (EPSG:3413) — no reprojection needed ───────────────
    combined.rio.set_spatial_dims(x_dim="xgrid", y_dim="ygrid", inplace=True)
    combined.rio.write_crs(crs, inplace=True)

    # ── Clip values to [0, 1] removing filled values
    combined = combined.clip(min=0, max=1)

    return combined

def main():

    crs      = 'epsg:3413'
    var_name = 'cdr_seaice_conc'

    regions = {
        'AlaskanArctic':      'arctic_sf.shp',
        'NorthernBering':     'nbering_sf.shp',
        'EasternBering':      'ebering_sf.shp',
        'SoutheasternBering': 'se_bering_sf.shp',
    }

    recent_dat = pd.read_csv("data/nrt_extent_NorthernBering.csv", parse_dates=['date'])
    start_date = (recent_dat['date'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    # start_date = '2026-04-11'  # For testing: hardcode start date to force processing of all available data
    end_date   = date.today().strftime('%Y-%m-%d')

    print(f"Fetching {start_date} → {end_date}")

    sic = get_var_data_nsidc(crs, var_name, [start_date, end_date])

    if "time" in sic.dims and sic.sizes["time"] > 0:

        for name, shp in regions.items():
            print(f'Processing: {name}')

            alaska_shp      = gpd.read_file(f'resources/alaska_shapefiles/{shp}')
            alaska_shp_proj = alaska_shp.to_crs(crs)
            ds_area         = get_area(name)
            clipped_sic     = clip_data(sic, alaska_shp_proj)
            ext             = compute_extent_km(clipped_sic, ds_area)

            ext_df = (ext
                      .to_dataframe()
                      .reset_index()
                      .drop(['spatial_ref'], axis='columns', errors='ignore')
                      .rename(columns={'time': 'date'})
                      [['date', 'seaice_extent']])
            try:
                ext_df.to_csv(f'data/nrt_extent_{name}.csv', mode='a', index=False, header=False)
                print(f'  Successfully updated nrt_extent_{name}.csv')
            except Exception as e:
                print(f'  Failed to update nrt_extent_{name}.csv: {e}')
    else:
        print("Processing Stopped: No new data available.")


if __name__ == "__main__":
    main()