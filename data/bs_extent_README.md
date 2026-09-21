# Metadata for Daily Sea Ice Extent Baseline Time Series (1985-2015)

## Description
The timeseries of the daily sea ice extent baseline (1985-2015) 
was computed using the NOAA/NSIDC Climate Data Record (CDR) daily sea ice concentration data (SIC)
You can access the SIC data at https://polarwatch.noaa.gov/erddap/griddap/nsidcG02202v4nh1day.html

The remote sensing data were cropped to fit within the boundaries of 
each ecosystem fisheries management region, and the sea ice extent 
was calculated using a SIC threshold of 0.15. The daily sea ice extent was averaged across the referenced 
years (1985-2015) to obtain the baseline daily extent.

For more detailed methods and the Python code used, 
please refer to the methods section and the GitHub repository.

## Data
* month: Month (format: mm)
* day: Day (format: dd)
* seaice_extent_mean: Mean of daily sea ice extent for the month-day across xgrid, ygrid, and date range (unit: square km)
* seaice_extent_std: Standard deviation of daily sea ice extent (unit: square km)
* date: Month-day (format: mm-dd)


## Resource
* link to methods : https://polarwatch.github.io/alaska-seaice/methods.html

## Creators
* NOAA Alaska Fisheries Science Center
* PolarWatch, NOAA CoastWatch 

## Contact
* Elizabeth.Siddon@noaa.gov (elizabeth.siddon@noaa.gov)

## Date Created
* 09/16/2026 updated by Alaska Fisheries Science Center
* 06/30/2024 created by PolarWatch, NOAA CoastWatch

