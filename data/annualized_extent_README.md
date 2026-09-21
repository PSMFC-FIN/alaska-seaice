# Metadata for Annualized Sea Ice Extent Time Series (1985-Current)

## Description

Annual sea ice extent was calculated using NSIDC monthly sea ice concentration (SIC) data at a 25 km resolution with a 0.15 concentration threshold. 
Due to the lag of the most recent data availability, the most current year extent was computed using the near real time data sea ice concentration while the extent of the previous years were computed using the Climate Data Record (CDR).

## Data source
The Climate Data Record monthly SIC data access at https://nsidc.org/data/g02202/versions/6
.html


For more detailed methods and the Python code used, 
please refer to the methods section and the GitHub repository.

## Data Description
region,year,extent
* region: Fisheries ecosystem region ("AlaskanArctic", "EasternBering", "NorthernBering", "SoutheasternBering")
* year: year of the sea ice extent (format: YYYY)
* extent: yearly sea ice extent mean (unit: squre km) 


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

