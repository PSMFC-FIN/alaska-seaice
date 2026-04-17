"""
Title: Compute total area in square km for Alaska Ecosystem Regions

Description:
    For each defined region, calculates the total grid cell area in square
    kilometers based on the 25 km polar stereographic projection and regional
    shapefiles. Uses a single test SIC file to establish the grid, then clips
    the cell area to each region and sums. Results are printed, plotted, and
    saved as NetCDF.

Regions:
    Alaskan Arctic, Northern Bering Sea, Eastern Bering Sea, Southeastern Bering Sea

Dependencies:
    geopandas, matplotlib, sic (local module)

Parameters:
    CRS          : EPSG:3411 (Polar Stereographic, Hughes 1980 ellipsoid)
    AREA_NC_PATH : NSIDC0771 cell area file
    VAR_NAME     : 'cdr_seaice_conc'

Author: Sunny Bak Hospital
Modified: April 16, 2026
"""

import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt

from sic import SIC25k, clip_data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CRS          = 'epsg:3411'
VAR_NAME     = 'cdr_seaice_conc'
AREA_NC_PATH = 'resources/ref_files/NSIDC0771_CellArea_PS_N25km_v1.0.nc'
TEST_DATA_DIR = 'data/test/'
RESOURCE_DIR  = 'resources/akmarineeco'

REGIONS = {
    # 'AlaskanArctic':        'arctic_sf.shp',
    # 'NorthernBering':       'nbering_sf.shp',
    # 'EasternBering':        'ebering_sf.shp',
    'SoutheasternBering':   'se_bering_sf.shp',
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load SIC and area once — grid is the same for all regions
    sic = SIC25k(
        data_dirs=TEST_DATA_DIR,
        varname=VAR_NAME,
        crs=CRS,
    )
    sic.load_area_local(AREA_NC_PATH)

    for name, shp_file in REGIONS.items():
        print(f"\nRegion: {name}")

        alaska_shp      = gpd.read_file(f'{RESOURCE_DIR}/{shp_file}')
        alaska_shp_proj = alaska_shp.to_crs(CRS)

        # Clip area to region and sum for total area
        area_clipped = clip_data(sic.area, alaska_shp_proj)
        total_area_km2 = float(area_clipped.sum()) / 1e6


        print(f"  Total grid cell area (upper limit) : {total_area_km2:,.2f} km2")
        print(f"  Shapefile area       : {alaska_shp_proj.geometry.area.sum() / 1e6:,.2f} km2")

        # Plot and save
        area_clipped.plot()
        plt.title(f"Cell area – {name}")
        plt.show()

        area_clipped.to_netcdf(f'area_{name}.nc')
        print(f"  Saved: area_{name}.nc")




# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()