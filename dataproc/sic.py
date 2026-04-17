"""
Title: Sea Ice Concentration Data Analysis
Author: Sunny Bak Hospital
Modified: April 16, 2026

Description:
    Provides tools for analyzing and processing sea ice concentration (SIC) data
    downloaded locally from NOAA/NSIDC. Loads from local NetCDF files via
    xarray.open_mfdataset().

    Designed for per-annual-window loading: pass two directory paths
    (e.g. ['data/cdr/2023', 'data/cdr/2024']) to cover a Sep–Aug window that
    crosses a calendar year boundary, keeping the Dask task graph small.

    Data sources (downloaded locally via download_nsidc.py):
    - CDR  : https://noaadata.apps.nsidc.org/NOAA/G02202_V6/north/daily/
    - NRT  : https://noaadata.apps.nsidc.org/NOAA/G10016_V4/north/daily/
    - Area : cell_area.nc (local file)

Modules:
    pandas, rioxarray, shapely, numpy, xarray, geopandas, dask, rasterio

Main Classes:
    IceData  : Base class for loading and manipulating local NetCDF SIC data.
    SIC25k   : Derived class for 25 km resolution SIC processing.

Helper Functions:
    clip_data : Spatial clipping using a GeoDataFrame shape.
"""

import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray                        # noqa: F401  (activates .rio accessor)
import xarray as xr
from shapely.geometry import mapping
from typing import Tuple, Union


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class IceData:
    def __init__(self,
                 data_dirs: Union[str, Path, list],
                 varname: str,
                 crs: str,
                 file_pattern: str = "*.nc",
                 grids: dict = None,
                 shape: gpd.GeoDataFrame = None):
        """
        Initializes IceData, loads sea ice data from one or more local NetCDF
        directories, and optionally clips to a shape.

        Args:
            data_dirs (str | Path | list): One directory path or a list of paths.
                Pass two paths to span a Sep-Aug window that crosses a calendar
                year boundary (e.g. ['data/cdr/2023', 'data/cdr/2024']).
            varname (str)      : Variable name to extract (e.g. 'cdr_seaice_conc').
            crs (str)          : CRS as EPSG string (e.g. 'epsg:3413').
            file_pattern (str) : Glob pattern for NetCDF files. Default '*.nc'.
            grids (dict)       : Spatial dim names, e.g. {'x': 'x', 'y': 'y'}.
            shape (GeoDataFrame, optional): Pre-projected clip geometry.
        """
        # Normalise to a list of Path objects
        if isinstance(data_dirs, (str, Path)):
            self.data_dirs = [Path(data_dirs)]
        else:
            self.data_dirs = [Path(d) for d in data_dirs]

        self.varname      = varname
        self.crs          = crs
        self.file_pattern = file_pattern
        self.grids        = grids or {'x': 'x', 'y': 'y'}
        self.shape        = shape

        try:
            if shape is not None:
                ds = self.load_data()
                self.ds = clip_data(ds, shape)
            else:
                self.ds = self.load_data()
        except Exception as e:
            print(f"Unable to load data: {e}")

    def __str__(self):
        return (f"IceData:\n  data_dirs={self.data_dirs}\n  varname={self.varname}\n"
                f"  crs={self.crs}\n  grids={self.grids}\n  shape={self.shape}\n"
                f"  dataset={self.ds}")

    def load_data(self) -> xr.DataArray:
        """
        Loads all NetCDF files matching file_pattern from each directory in
        data_dirs into a single time-concatenated xarray DataArray.
        Assigns CRS and clips values to [0, 1].

        Returns:
            xr.DataArray: Time-concatenated SIC DataArray.

        Raises:
            FileNotFoundError : If no matching files are found across all dirs.
            ValueError        : If varname is not present in the dataset.
        """
        files = []
        for d in self.data_dirs:
            files.extend(glob.glob(str(d / self.file_pattern)))

        if not files:
            raise FileNotFoundError(
                f"No files matching '{self.file_pattern}' found in: "
                f"{[str(d) for d in self.data_dirs]}"
            )

        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            chunks={"time": "auto"},
            parallel=True,
        )

        if self.varname not in ds:
            raise ValueError(
                f"Variable '{self.varname}' not found. "
                f"Available: {list(ds.data_vars)}"
            )

        da = ds[self.varname]
        da.rio.set_spatial_dims(
            x_dim=self.grids['x'], y_dim=self.grids['y'], inplace=True
        )
        da.rio.write_crs(self.crs, inplace=True)
        da = da.clip(min=0, max=1)

        return da

    def compute_clim(self, year_range: list, frequency: str) -> xr.DataArray:
        """
        Computes a long-term climatological mean for a given year range and
        temporal frequency.

        Args:
            year_range (list) : [start_year, end_year] (inclusive).
            frequency (str)   : One of '15D', 'W', 'D', 'M', 'Q'.

        Returns:
            xr.DataArray: Climatology averaged at the specified frequency.

        Raises:
            ValueError: If year_range is out of bounds or frequency is invalid.
        """
        start_year = pd.Timestamp(self.ds['time'].min().values).year
        end_year   = pd.Timestamp(self.ds['time'].max().values).year

        if year_range[0] < start_year or year_range[1] > end_year:
            raise ValueError(
                f"year_range {year_range} is outside dataset bounds "
                f"[{start_year}, {end_year}]."
            )

        ds_sel = self.ds.sel(
            time=slice(f"{year_range[0]}-01-01", f"{year_range[1]}-12-31")
        ).chunk({'x': 'auto', 'y': 'auto'})

        if frequency == "15D":
            ds_clim = ds_sel.groupby(
                custom_15day_interval(ds_sel.time)
            ).mean("time")
        elif frequency in ["W", "D", "M", "Q"]:
            ds_clim = ds_sel.resample(time=frequency).mean()
        else:
            raise ValueError(
                "frequency must be one of ['15D', 'W', 'D', 'M', 'Q']"
            )

        return ds_clim


# ---------------------------------------------------------------------------
# SIC25k  - 25 km resolution sea ice concentration
# ---------------------------------------------------------------------------

class SIC25k(IceData):
    """
    Handles NSIDC 25 km resolution sea ice concentration data loaded from
    local NetCDF files (CDR G02202_V6 or NRT G10016_V4).

    Designed for per-annual-window use: instantiate with the one or two
    year directories that cover the Sep-Aug window of interest, compute,
    then discard. This keeps memory and Dask task graphs small.

    Inherits from IceData.
    """

    def __init__(self,
                 data_dirs: Union[str, Path, list],
                 varname: str,
                 crs: str,
                 file_pattern: str = "*.nc",
                 shape: gpd.GeoDataFrame = None):
        """
        Args:
            data_dirs (str | Path | list): One or two local directories
                containing daily NetCDF files for the window of interest.
            varname (str)      : Variable name (e.g. 'cdr_seaice_conc').
            crs (str)          : CRS as EPSG string (e.g. 'epsg:3413').
            file_pattern (str) : Glob pattern. Default '*.nc'.
            shape (GeoDataFrame, optional): Clip geometry (in dataset CRS).
        """
        super().__init__(
            data_dirs=data_dirs,
            varname=varname,
            crs=crs,
            file_pattern=file_pattern,
            grids={'x': 'x', 'y': 'y'},
            shape=shape,
        )
        self.area = None   # Populated by load_area_local()

    # ------------------------------------------------------------------
    # Area helpers
    # ------------------------------------------------------------------

    def has_area(self) -> bool:
        return self.area is not None

    def load_area_local(self, area_nc_path: str):
        """
        Loads grid-cell area from a local NetCDF file (cell_area.nc).

        Args:
            area_nc_path (str): Full path to the cell_area.nc file.

        Raises:
            FileNotFoundError : If the file does not exist.
            KeyError          : If 'cell_area' variable is missing.
        """
        path = Path(area_nc_path)
        if not path.is_file():
            raise FileNotFoundError(f"Area file not found: {area_nc_path}")

        try:
            ds = xr.open_dataset(path)
            da = ds['cell_area']
            da.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
            da.rio.write_crs(self.crs, inplace=True)

            if self.shape is not None:
                da = clip_data(da, self.shape)

            self.area = da
            print(f"Grid-cell area loaded from: {area_nc_path}")

        except Exception as e:
            print(f"Error loading area file: {e}")
            raise

    def get_area(self) -> xr.DataArray:
        if not self.has_area():
            raise ValueError("Area not loaded. Call load_area_local() first.")
        return self.area

    def get_total_area_km(self, shp: gpd.GeoDataFrame) -> float:
        """
        Total valid ocean area (km2) within a region shape.

        Args:
            shp (GeoDataFrame): Region geometry.

        Returns:
            float: Total area in km2.
        """
        if self.area is None:
            raise ValueError("Grid-cell area not loaded.")

        ds   = self.ds.isel(time=0)
        ds   = clip_data(ds, shp)
        area = clip_data(self.area, shp)

        mask     = xr.where(ds.notnull(), 1, np.nan)
        ice_ext  = mask * area
        ice_ext.name = 'area_km2'
        tot_area = ice_ext.sum(dim=['x', 'y'], skipna=True).values

        return float(tot_area) / 1e6   # m2 -> km2

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------

    def subset_dim(self, dates: list,
                   shp: gpd.GeoDataFrame) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Subsets by time range and optionally clips to a region shape.

        Args:
            dates (list)       : ['YYYY-MM-DD', 'YYYY-MM-DD'] start/end.
            shp (GeoDataFrame) : Region shape for spatial clipping.

        Returns:
            tuple: (sic DataArray, area DataArray) - both clipped if shp given.
        """
        ds = self.ds.sel(time=slice(dates[0], dates[1]))

        if shp is not None and not shp.empty and self.area is not None:
            ds   = clip_data(ds,        shp)
            area = clip_data(self.area, shp)
            return ds, area
        else:
            return ds, self.area

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def format_sic(self, ds: xr.DataArray,
                   threshold: float = 0.15) -> xr.DataArray:
        """
        Binarises SIC data: 1 where conc >= threshold, 0 otherwise, NaN preserved.

        Args:
            ds (xr.DataArray) : Sea ice concentration DataArray.
            threshold (float) : Ice/no-ice threshold (default 0.15).

        Returns:
            xr.DataArray: Binary (0/1/NaN) DataArray.
        """
        return xr.where(ds.isnull(), np.nan,
                        xr.where(ds >= threshold, 1, 0))

    def compute_extent_km(self, ds: xr.DataArray,
                          area: xr.DataArray) -> xr.DataArray:
        """
        Computes sea ice extent (km2) by multiplying binary SIC by cell area
        and summing spatially for each time step.

        Args:
            ds (xr.DataArray)  : Binary SIC DataArray.
            area (xr.DataArray): Grid-cell area DataArray (m2).

        Returns:
            xr.DataArray: Sea ice extent (km2) per time step.

        Raises:
            TypeError: If inputs are not xr.DataArray.
        """
        if isinstance(ds, xr.Dataset):
            dvars = list(ds.data_vars)
            if len(dvars) == 1:
                ds = ds[dvars[0]]
            else:
                raise TypeError(
                    "Input `ds` is a multi-variable Dataset. "
                    "Provide a DataArray or single-variable Dataset."
                )

        ice_cell = ds * area / 1e6   # m2 -> km2

        if not isinstance(ice_cell, xr.DataArray):
            raise TypeError(
                f"Result should be xr.DataArray, got {type(ice_cell)}"
            )

        ice_cell.name = 'seaice_extent'
        ice_ext = ice_cell.sum(dim=['x', 'y'])
        return ice_ext


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def clip_data(ds: xr.DataArray,
              shape: gpd.GeoDataFrame) -> xr.DataArray:
    """
    Clips an xarray DataArray to a GeoDataFrame geometry.
    Automatically reprojects the shape if its CRS differs from the data.

    Args:
        ds (xr.DataArray)   : Spatially-aware DataArray (must have .rio accessor).
        shape (GeoDataFrame): Clip geometry.

    Returns:
        xr.DataArray: Clipped DataArray.
    """
    if not shape.crs.equals(ds.rio.crs):
        print("CRS mismatch - reprojecting shape to match data CRS.")
        shape = shape.to_crs(ds.rio.crs)
    return ds.rio.clip(shape.geometry.apply(mapping), shape.crs)
