# Distribution-based Upscaling on High-Resolution InSAR Permafrost Deformation

Code written and data used for the manuscript of permafrost deformation upscaling, part of Zhijun Liu's PhD project


### Prerequisites

Please check each code file for the exact Python packages required.


## Running the tests

Please prepare the input data and save them in your preferred local directory before running the code. You will need to:

1. Adjust the file paths in the scripts to match your local setup.
2. Substitute the region name in the code where appropriate.

The analysis is organized into three main components, each in a separate file:

- invk_cluster_hist.ipynb  
  This is the main notebook. It:
    • Imports all required packages  
    • Defines most of the shared functions  
    • Provides the core workflow for distribution-based upscaling
    • Provides the core workflow for our heuristic clustering algorithm on histrograms
  Both pearson_corr.py and robustness.py may depend on functions defined in this notebook.

- pearson_corr.py  
  Computes Pearson correlation coefficients between:
    • Annual permafrost deformation (aggregated to 10 km using distance-weighted averaging), and
    • Selected climatic forcings from ERA5-Land and topographic factors from MERIT.

- robustness.py  
  Performs Kolmogorov–Smirnov (K–S) tests to assess the robustness of subset data distribution.

- ERAprocess.sh
  Processes downloaded different ERA-Land data to variables that can be used for correlation analysis. 


# Data Availability
ERA-Land climate forcings: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview

MERIT Hydro topography data: [https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/](https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/)

Sentinel-1 data: [https://scihub.copernicus.eu](https://dataspace.copernicus.eu/data-collections/sentinel-data/sentinel-1)

PALSAR-2 level 1 product: https://search.asf.alaska.edu

10 km Permafrost Deformation: [zenodo] .



## Author

* **Zhijun Liu** - *Initial work* - (https://github.com/anALPAKKA) zhijun.liu@mpimet.mpg

## References
Copernicus Climate Change Service (C3S)(2019): ERA5-Land hourly data from 1950 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.e2161bac
Yamazaki, D., D. Ikeshima, R. Tawatari, T. Yamaguchi, F. O'Loughlin, J. C. Neal, C. C. Sampson, S. Kanae, and P. D. Bates (2017), A high-accuracy map of global terrain elevations, Geophys. Res. Lett., 44, 5844–5853, doi:10.1002/2017GL072874.


## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE.md](LICENSE.md) file for details

## Paper

The accompanying scientific paper is in preparation.



