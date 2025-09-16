#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constants for the LA Megacity Dashboard
"""

import os
from pathlib import Path
import geopandas as gpd

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = os.path.join(ROOT_DIR, 'data')
SHAPEFILE_PATH = os.path.join(DATA_DIR, 'shapefiles')
OCO3_DATA_PATH = os.path.join(DATA_DIR, 'csv')
SITE_DATA_PATH = os.path.join(DATA_DIR, 'hdf_files')

# Print paths for debugging
print(f"ROOT_DIR: {ROOT_DIR}")
print(f"SHAPEFILE_PATH: {SHAPEFILE_PATH}")
print(f"OCO3_DATA_PATH: {OCO3_DATA_PATH}")

# %% Sites and Inlet Heights
sites = ['CIT', 'CNP', 'COM', 'FUL', 'GRA', 'IRV', 'ONT', 'USC', 'SCI', 'VIC', 'LJO','RAN']
site_inlet = ['all', 'all', '45m', 'all', '51m', 'all', '41m', 'all', '27m', '139m', '13m', 'all']
site_dict = {'CIT': 'all', 'CNP': 'all', 'COM': '45m', 'FUL': 'all', 'GRA': '51m',
             'IRV': 'all', 'ONT': '41m', 'USC': 'all', 'SCI': '27m', 'VIC': '139m', 'LJO': '13m'}
gas_dict = {'co2': 'ppm', 'ch4': 'ppb', 'co': 'ppb'}

#sites_co=['CIT', 'CNP', 'COM', 'FUL', 'GRA', 'IRV', 'ONT', 'SCI', 'LJO','RAN']


#site_inlet_co = ['all', 'all', '45m', 'all', '51m', 'all', '41m', '27m', '13m','upwind']

sites_co=['CIT', 'CNP', 'COM', 'FUL', 'GRA', 'IRV', 'ONT', 'SCI', 'LJO','RAN']


site_inlet_co = ['all', 'all', '45m', 'all', '51m', 'all', '41m', '27m', '13m','all']


sites_ch4=['CIT', 'CNP', 'COM', 'FUL', 'GRA', 'IRV', 'ONT', 'USC', 'SCI', 'VIC', 'LJO','RAN']


site_inlet_ch4 = ['all', 'all', '45m', 'all', '51m', 'all', '41m', 'all', '27m', '139m', '13m', 'all']
# %% Data available for years
begin_year = 2015
end_year = 2025
# generate list of years
years = list(range(begin_year, end_year+1, 1))

# %% Types of gases
gas_type = ['co2', 'ch4', 'co']
meas_units = ['ppm', 'ppb', 'ppb']
processing_gas = gas_type[0]
processing_meas_units = meas_units[0]
time_frequency = 'h'

# %% Path for Socab shape file
try:
    geo_df = gpd.read_file(os.path.join(SHAPEFILE_PATH, 'socabbound.shp'))
    geo_df2 = gpd.read_file(os.path.join(SHAPEFILE_PATH, 'paper_towers.shp'))
except Exception as e:
    print(f"Error loading shapefiles: {e}")
    geo_df = None
    geo_df2 = None
# %% All parameters for background calculation are from kris paper
#  Verhulst, K. R., Karion, A., Kim, J., Salameh, P. K., Keeling, R. F., Newman, S., Miller, J., Sloop, C.,
#  Pongetti, T., Rao, P., Wong, C., Hopkins, F. M., Yadav, V., Weiss, R. F., Duren, R. M., and Miller, C. E.:
#  Carbon dioxide and methane measurements from the Los Angeles Megacity Carbon Project – Part 1: calibration,
#  urban enhancements, and uncertainty estimates, Atmos. Chem. Phys., 17, 8313–8341,
#  https://doi.org/10.5194/acp-17-8313-2017, 2017.

# %% Background parameters

# Only retain observations with standard deviation less than below given values
# gas_sd_filter = [0.3, 3, 3]  # ppm CO2, ppb CH4, ppb CO
# gas_sd_filter = {'co2': 0.3, 'ch4': 3, 'co': 3}

# Hour to hour changes (derivative) should be less than the given value for the gas
# gas_stability_filter = {'co2': 0.25, 'ch4': None, 'co': None}  # ppm CO2, ppb CH4, ppb CO

gas_sd_filter = {'CIT': {'co2': 0.3, 'ch4': 3, 'co': 2},
                 'CNP': {'co2': 0.3, 'ch4': 3, 'co': 2},
                 'COM': {'co2': 0.3, 'ch4': 3, 'co': 2},
                 'FUL': {'co2': 0.3, 'ch4': 3, 'co': 2},
                 'GRA': {'co2': 0.3, 'ch4': 3, 'co': 2},
                 'IRV': {'co2': 0.3, 'ch4': 3, 'co': 2},
                 'ONT': {'co2': 0.3, 'ch4': 3, 'co': 2},
                 'USC2': {'co2': 0.3, 'ch4': 3, 'co': 2},
                 'SCI': {'co2': 0.3, 'ch4': 3, 'co': 2},
                 'VIC': {'co2': 0.3, 'ch4': 3, 'co': 2},
                 'LJO': {'co2': 0.3, 'ch4': 3, 'co': 2}}

gas_stability_filter = {'CIT': {'co2': 0.25, 'ch4': None, 'co': None},
                        'CNP': {'co2': 0.50, 'ch4': None, 'co': None},
                        'COM': {'co2': 0.50, 'ch4': None, 'co': None},
                        'FUL': {'co2': 0.50, 'ch4': None, 'co': None},
                        'GRA': {'co2': 0.50, 'ch4': None, 'co': None},
                        'IRV': {'co2': 0.50, 'ch4': None, 'co': None},
                        'ONT': {'co2': 0.50, 'ch4': None, 'co': None},
                        'USC2': {'co2': 0.50, 'ch4': None, 'co': None},
                        'SCI': {'co2': 0.25, 'ch4': None, 'co': None},
                        'VIC': {'co2': 0.50, 'ch4': None, 'co': None},
                        'LJO': {'co2': 0.50, 'ch4': None, 'co': None}}

# Consecutive hour filter only applicable for CO2
gas_num_hours = {'CIT': 4, 'CNP': 4, 'COM': 4, 'FUL': 4, 'GRA': 4,
                    'IRV': 4, 'ONT': 4, 'USC2': 4, 'SCI': 6, 'VIC': 4, 'LJO': 4}
num_hours = 6
# %% CCGCRV Equal Interval Parameters
# Smoothing of background curve requires use of CCGCRV program available from
# https://gml.noaa.gov/aftp/user/thoning/ccgcrv/
# also consult manual
# https://gml.noaa.gov/aftp/user/thoning/ccgcrv/ccgcrv.pdf
short_term = 80  # days
long_term = 667  # days
sample_interval_equal = 0.04166666666  # hourly interval in decimal year
sample_interval_original = 0
polynomial_terms = 3
numer_harmonics = 4
time_zero = 0  # beginning time should be beginning of the data or Jan 1st
gap = 0
debug = False

# %% Plot parameters for CCGCRV and extra information extraction parameters
smooth_figure = True
function_figure = True
print_filtered_results = True

