import xarray as xr 
import numpy as np


def main():
    area = xr.open_dataset('resources/ref_files/NSIDC0771_CellArea_PS_N25km_v1.0.nc')
    # ds = xr.open_dataset('resources/ref_files/G02202-ancillary-psn25-v06r00.nc')

    # print(ds['cell_area'])  # check units attribute
    # print(ds['cell_area'].values.max())
 

    sic = xr.open_dataset('data/cdr/2025/sic_psn25_20250101_am2_v06r00.nc')
 
    print("area x:", area.x.values[:5], "...")
    print("sic  x:", sic.x.values[:5], "...")
    print("area y:", area.y.values[:5], "...")
    print("sic  y:", sic.y.values[:5], "...")

    print("\nCoords match x:", np.allclose(area.x.values, sic.x.values))
    print("Coords match y:", np.allclose(area.y.values, sic.y.values))
    print("\nShapes — area:", area['cell_area'].shape, "  sic:", sic['cdr_seaice_conc'].shape)

    import geopandas as gpd
    shp  = gpd.read_file('resources/akmarineeco/ebering_sf.shp').to_crs('epsg:3411')
    print("Region area from shapefile (km2):", shp.geometry.area.sum() / 1e6)

if __name__ == "__main__":
    main()
