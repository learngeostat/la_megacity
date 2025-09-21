#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constants and configuration for the LA Megacity Dashboard.
This version is updated for a cloud-native deployment using Google Cloud Storage (GCS).
All shapefiles are stored as GeoPackage (.gpkg) format for better cloud compatibility.
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
    'aggregated_data_allhours': f"{HDF_FILES_PATH}/aggregated_data_allhours.h5",

    # For emissions.py (OCO-3)
    'oco3_obs': f"{CSV_PATH}/clipped_oco3_obs.csv",
    
    # For flux_hindcast.py
    'fluxresults1': f"{HDF_FILES_PATH}/fluxresults1.nc",
    'spatial_data': f"{HDF_FILES_PATH}/spatial_data.h5",
}

# ############################################################################
# Shapefile/GeoPackage Configuration
# 
# All spatial data files are stored as GeoPackage (.gpkg) format for better
# cloud compatibility and single-file storage. GeoPackages are:
# - Self-contained (no need for multiple .shp, .shx, .dbf files)
# - Better cloud storage compatibility
# - More robust for programmatic access
# ############################################################################

SHAPEFILES = {
    'socabbound': f"{SHAPEFILE_PATH}/socabbound.gpkg",
    'paper_towers': f"{SHAPEFILE_PATH}/paper_towers.gpkg", 
    'census_tract_clipped': f"{SHAPEFILE_PATH}/census_tract_clipped.gpkg",
    'zip_code_socab': f"{SHAPEFILE_PATH}/zip_code_socab.gpkg",
    'zones_partitoned': f"{SHAPEFILE_PATH}/zones_partitoned.gpkg",
    'census_tracts_emissions_dashboard': f"{SHAPEFILE_PATH}/census_tracts_emissions_dashboard.gpkg"
}

# Helper function to get just the filename for a shapefile key
def get_shapefile_filename(key):
    """
    Get just the filename (with extension) for a shapefile key.
    
    Args:
        key (str): Key from SHAPEFILES dictionary
        
    Returns:
        str: Filename with .gpkg extension
        
    Example:
        get_shapefile_filename('zip_code_socab') returns 'zip_code_socab.gpkg'
    """
    if key not in SHAPEFILES:
        raise ValueError(f"Unknown shapefile key: {key}")
    return SHAPEFILES[key].split('/')[-1]

# Mapping for spatial aggregation types to their corresponding shapefile keys
SPATIAL_AGG_SHAPEFILES = {
    'zip': 'zip_code_socab',
    'census': 'census_tract_clipped', 
    'custom': 'zones_partitoned'
}

# Feature ID field names for each spatial aggregation type
FEATURE_ID_MAPPING = {
    'zip': 'ZIP_CODE',
    'census': 'TRACTCE',
    'custom': 'Zones'
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
# Cloud Storage Helper Functions
# ############################################################################

def load_geopackage_from_gcs(shapefile_key):
    """
    Helper function to get the full GCS path for a shapefile by key.
    
    Args:
        shapefile_key (str): Key from SHAPEFILES dictionary
        
    Returns:
        str: Full GCS path to the shapefile
        
    Example:
        load_geopackage_from_gcs('zip_code_socab') returns 
        'gs://la-megacity-dashboard-data-1/data/shapefiles/zip_code_socab.gpkg'
    """
    if shapefile_key not in SHAPEFILES:
        raise ValueError(f"Unknown shapefile key: {shapefile_key}. Available keys: {list(SHAPEFILES.keys())}")
    return SHAPEFILES[shapefile_key]


# ############################################################################
# NOTE on Data Loading
#
# The direct loading of spatial files (e.g., gdf = gpd.read_file(...)) has
# been REMOVED from this configuration file.
#
# A constants file should only define constants, not perform actions
# like reading data. This data should be loaded in the `init()` function
# of the specific page that requires it using the helper functions provided
# and appropriate temporary file handling for cloud storage access.
# ############################################################################


