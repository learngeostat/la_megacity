#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constants and configuration for the LA Megacity Dashboard.
This version is updated for a cloud-native deployment using Google Cloud Storage (GCS).
"""

# ############################################################################
# GCS Configuration
# ############################################################################

# The name of your Google Cloud Storage bucket
BUCKET_NAME = "la-megacity-dashboard-data-1"

# The base URI for the bucket, used to construct all other paths
GCS_BUCKET_URI = f"gs://{BUCKET_NAME}"


# ############################################################################
# GCS Data Paths
# ############################################################################

# Base URIs for the different data folders within the bucket
DATA_PATH = f"{GCS_BUCKET_URI}/data"
SHAPEFILE_PATH = f"{DATA_PATH}/shapefiles"
HDF_FILES_PATH = f"{DATA_PATH}/hdf_files"
CSV_PATH = f"{DATA_PATH}/csv"


# ############################################################################
# Data File URIs
# A centralized dictionary of all data files.
# Use these keys to access the full GCS URI for each file in your application.
# ############################################################################

DATA_FILES = {
    # For surface_observations.py
    'aggregated_data_afternoon': f"{HDF_FILES_PATH}/aggregated_data_afternoon.h5",
    'aggregated_data_allhours.h5': f"{HDF_FILES_PATH}/aggregated_data_allhours.h5",  # Fixed: added missing key

    # For emissions.py (OCO-3)
    'oco3_obs': f"{CSV_PATH}/clipped_oco3_obs.csv",
    
    # For flux_hindcast.py
    'fluxresults1': f"{HDF_FILES_PATH}/fluxresults1.nc",  # Fixed: moved to hdf_files
    'spatial_data': f"{HDF_FILES_PATH}/spatial_data.h5",
}

# A centralized dictionary for all shapefiles. Geopandas reads the .shp file.
SHAPEFILES = {
    'socabbound': f"{SHAPEFILE_PATH}/socabbound.gpkg",
    'paper_towers': f"{SHAPEFILE_PATH}/paper_towers.gpkg",
    'census_tract_clipped': f"{SHAPEFILE_PATH}/census_tract_clipped.gpkg",
    'zip_code_socab': f"{SHAPEFILE_PATH}/zip_code_socab.gpkg",
    'zones_partitoned': f"{SHAPEFILE_PATH}/zones_partitoned.gpkg",
    'census_tracts_emissions_dashboard': f"{SHAPEFILE_PATH}/census_tracts_emissions_dashboard.gpkg"
}


# ############################################################################
# Site and Gas Configuration (Static)
# This section remains unchanged as it contains static configuration data.
# ############################################################################

sites = ['CIT', 'CNP', 'COM', 'FUL', 'GRA', 'IRV', 'ONT', 'USC', 'SCI', 'VIC', 'LJO','RAN']
site_inlet = ['all', 'all', '45m', 'all', '51m', 'all', '41m', 'all', '27m', '139m', '13m', 'all']
site_dict = {'CIT': 'all', 'CNP': 'all', 'COM': '45m', 'FUL': 'all', 'GRA': '51m',
             'IRV': 'all', 'ONT': '41m', 'USC': 'all', 'SCI': '27m', 'VIC': '139m', 'LJO': '13m'}
gas_dict = {'co2': 'ppm', 'ch4': 'ppb', 'co': 'ppb'}
sites_co=['CIT', 'CNP', 'COM', 'FUL', 'GRA', 'IRV', 'ONT', 'SCI', 'LJO','RAN']
site_inlet_co = ['all', 'all', '45m', 'all', '51m', 'all', '41m', '27m', '13m','all']
sites_ch4=['CIT', 'CNP', 'COM', 'FUL', 'GRA', 'IRV', 'ONT', 'USC', 'SCI', 'VIC', 'LJO','RAN']
site_inlet_ch4 = ['all', 'all', '45m', 'all', '51m', 'all', '41m', 'all', '27m', '139m', '13m', 'all']


# ############################################################################
# Time and Processing Parameters (Static)
# ############################################################################

begin_year = 2015
end_year = 2025
years = list(range(begin_year, end_year + 1))
gas_type = ['co2', 'ch4', 'co']
meas_units = ['ppm', 'ppb', 'ppb']
processing_gas = gas_type[0]
processing_meas_units = meas_units[0]
time_frequency = 'h'


# ############################################################################
# Background Calculation Parameters (Static)
# ############################################################################

gas_sd_filter = {
    'CIT': {'co2': 0.3, 'ch4': 3, 'co': 2},
    'CNP': {'co2': 0.3, 'ch4': 3, 'co': 2},
    'COM': {'co2': 0.3, 'ch4': 3, 'co': 2},
    'FUL': {'co2': 0.3, 'ch4': 3, 'co': 2},
    'GRA': {'co2': 0.3, 'ch4': 3, 'co': 2},
    'IRV': {'co2': 0.3, 'ch4': 3, 'co': 2},
    'ONT': {'co2': 0.3, 'ch4': 3, 'co': 2},
    'USC2': {'co2': 0.3, 'ch4': 3, 'co': 2},
    'SCI': {'co2': 0.3, 'ch4': 3, 'co': 2},
    'VIC': {'co2': 0.3, 'ch4': 3, 'co': 2},
    'LJO': {'co2': 0.3, 'ch4': 3, 'co': 2}
}

gas_stability_filter = {
    'CIT': {'co2': 0.25, 'ch4': None, 'co': None},
    'CNP': {'co2': 0.50, 'ch4': None, 'co': None},
    'COM': {'co2': 0.50, 'ch4': None, 'co': None},
    'FUL': {'co2': 0.50, 'ch4': None, 'co': None},
    'GRA': {'co2': 0.50, 'ch4': None, 'co': None},
    'IRV': {'co2': 0.50, 'ch4': None, 'co': None},
    'ONT': {'co2': 0.50, 'ch4': None, 'co': None},
    'USC2': {'co2': 0.50, 'ch4': None, 'co': None},
    'SCI': {'co2': 0.25, 'ch4': None, 'co': None},
    'VIC': {'co2': 0.50, 'ch4': None, 'co': None},
    'LJO': {'co2': 0.50, 'ch4': None, 'co': None}
}

gas_num_hours = {
    'CIT': 4, 'CNP': 4, 'COM': 4, 'FUL': 4, 'GRA': 4,
    'IRV': 4, 'ONT': 4, 'USC2': 4, 'SCI': 6, 'VIC': 4, 'LJO': 4
}


# ############################################################################
# NOTE on Data Loading
#
# The direct loading of shapefiles (e.g., geo_df = gpd.read_file(...)) has
# been REMOVED from this configuration file.
#
# A constants file should only define constants, not perform actions
# like reading data. This data should be loaded in the `init()` function
# of the specific page that requires it (e.g., in `surface_observations.py`).
# ############################################################################


