# Beyond Canopy Cover: Developing & Assessing Satellite-Based Indices of Canopy Quality in Cities

This repository contains code and data used for the study **“Beyond Canopy Cover: Developing & Assessing Satellite-Based Indices of Canopy Quality in Cities.”**

The repository is intended to document key components of the analytical workflow and support reproducibility and adaptation of these methods to other study areas.

# Authors
Robert Moore (a,b), Brice Grunert (a), Kaiguang Zhao (c), and Kevin Mueller (a). 

(a) Cleveland State University, Department of Biological, Geological, and Environmental Sciences, Cleveland, OH, 44115, USA  

(b) Clark University, Department of Geography, Worcester, MA, 01610, USA 

(c) School of Environment and Natural Resources, The Ohio State University, Columbus, OH, 43210, USA 

## Data

Three related datasets are included in this repository.

### 1. Ground-Truthing Analysis Data

The ground-truthing dataset contains aggregated tree survey data for **41 census blocks** where all woody vegetation within each census block boundary was surveyed using field-based methods.

This dataset also includes remotely sensed metrics of canopy quality derived for these census blocks, allowing field-based measurements to be compared with remotely sensed indicators.

### 2. Remote Sensing Analysis Data

The remote sensing dataset contains the larger sample of census blocks used for the remote sensing component of the study.

These census blocks contain the remotely sensed indices of canopy quality used in the analysis.

### 3. Tree Survey Data

The tree survey dataset contains the **individual tree-level observations** collected using field-based survey methods.

Each dataset includes a corresponding **metadata table** describing the variables and columns contained within the dataset.

## Code and Reproducibility

The scripts included in this repository document major components of the analytical workflow and are intended to help researchers reproduce or adapt these methods for other study areas. Although this code primarily utilizes publicly accessible data, our analysis is highly contingent on multi-class land cover data with high spatial resolution.
Specifically, generation of residuals as we have done is not only dependent on canopy cover data but also grass/shrub (or other plant cover) as well as impervious surface need to be accounted for. Methods for the production of the land cover data we used can be found here:

https://www.countyplanning.us/projects/urban-tree-canopy-assessment-update/urban-tree-canopy-assessment-update-land-cover-methodology/. 

The provided code can be used to:

1. **Download NDVI and land surface temperature (LST) imagery** for a user-defined area of interest using `image-export.py`.
2. **Reproduce the data filtering procedures** used to prepare the datasets for analysis. `data-filters.py`
3. **Generate residual-based metrics of LST and NDVI** using the modeling procedures applied in this study `residual-model.py`.
4. **Calculate zonal statistics** for a series of raster images using census blocks or other polygon-based areas of interest to calculate averages and pixel counts which are needed for all subsequent analysis `zonal-stats.py`.
5. **Recreate the statistical modeling procedure** used for census blocks containing both field-based tree survey data and remotely sensed canopy-quality metrics `statistical-modeling.py`

## Scope and Limitations

The code in this repository documents important components of the analytical workflow but does not represent the complete analysis conducted for the study.

Some processing and analysis were performed externally using GIS and statistical software and are therefore not fully represented by the scripts provided here.

The primary purpose of the repository is to provide a record of the computational methods used in the study and to enable researchers to reproduce or adapt these methods for other geographic areas.

## Python Environment

Python package requirements and dependencies are specified in the `pixi.toml` file to enable reproducibility of the computational environment used for the analysis.


Data

This repository contains data obtained from multiple publicly available third-party sources, as well as datasets and derived products generated as part of this research.

Data and derived products created by the authors of this study may be used for research and reproducibility purposes. Users should provide appropriate attribution and cite the associated publication and/or archived repository when using these materials.

Please refer to the dataset metadata and original data providers for additional information regarding individual data sources.

USGS Landsat 7 Level 2, Collection 2, Tier 1
https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2

USGS Landsat 8 Level 2, Collection 2, Tier 1
https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2

Landsat Level 2, Collection 2 Documentation
https://www.usgs.gov/media/files/landsat-8-9-olitirs-collection-2-level-2-data-format-control-book

U.S. TIGER/LINE ShapeFiles - US Decennial Census - American Community Survey
https://data.census.gov/, https://www.census.gov/cgi-bin/geo/shapefiles/index.php

Cuyahoga County GIS - https://gis.cuyahogacounty.us/portal/home/


