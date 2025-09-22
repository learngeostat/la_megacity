# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 17:29:40 2025

@author: vyadav
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:30:02 2025

@author: vyadav
"""

from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import dash
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
import geopandas as gpd
import xarray as xr
from utils import conc_func as cfunc 
import math
import tempfile
import zipfile
import gcsfs
import io
import h5py


import logging
#logging.getLogger("fiona").disabled = True

logger = logging.getLogger(__name__)

# Hardcoded GCS bucket paths
# Hardcoded GCS bucket paths
GCS_BUCKET = "gs://la-megacity-dashboard-data-1"
GCS_HDF_FOLDER_PATH = f"{GCS_BUCKET}/data/hdf_files/"
# CORRECTED: Point to the hdf_files subfolder for the NetCDF file as well
GCS_NC_FOLDER_PATH = f"{GCS_BUCKET}/data/hdf_files/" 
GCS_SHAPEFILE_FOLDER_PATH = f"{GCS_BUCKET}/data/shapefiles/"



# Global variables
FLUX_DATA = None
UNCERTAINTY_DATA = None
DATES = None
LAT = None
LON = None
LAT_GRID = None
LON_GRID = None
available_dates = []
ZIP_DATA = None
CENSUS_DATA = None
CUSTOM_DATA = None
feature_id_mapping = {
    'zip': 'ZIP_CODE',
    'census': 'TRACTCE',
    'custom': 'Zones'
}

def load_gpkg_from_gcs(gcs_gpkg_path):
    """
    Load GeoPackage from GCS by downloading to temporary local storage first.
    Required because GDAL doesn't work directly with GCS paths.
    """
    
    logger.info(f"Loading GeoPackage from GCS: {gcs_gpkg_path}")
    fs = gcsfs.GCSFileSystem()
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False) as temp_file:
            try:
                # Download the GeoPackage file from GCS
                logger.info(f"Downloading GeoPackage to temporary location: {temp_file.name}")
                fs.get(gcs_gpkg_path, temp_file.name)
                
                # Verify the file was downloaded correctly
                temp_size = os.path.getsize(temp_file.name)
                logger.info(f"Downloaded GeoPackage file size: {temp_size} bytes")
                
                if temp_size == 0:
                    raise ValueError("Downloaded GeoPackage file is empty")
                
                # Load the GeoPackage
                gdf = gpd.read_file(temp_file.name)
                logger.info(f"Successfully loaded GeoPackage with {len(gdf)} features")
                
                return gdf
                
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_file.name)
                    logger.info("Cleaned up temporary GeoPackage file")
                except OSError as e:
                    logger.warning(f"Could not clean up temporary file {temp_file.name}: {e}")
            
    except Exception as e:
        logger.error(f"Failed to load GeoPackage from {gcs_gpkg_path}: {e}")
        raise



def debug_netcdf_file(filename):
    """
    Debug function to inspect NetCDF file on GCS
    """
    import gcsfs
    import tempfile
    import os
    
    fs = gcsfs.GCSFileSystem()
    
    try:
        # Check file existence and properties
        if not fs.exists(filename):
            print(f"❌ File does not exist: {filename}")
            return False
        
        file_info = fs.info(filename)
        print(f"✓ File exists: {filename}")
        print(f"  Size: {file_info.get('size', 'unknown')} bytes")
        print(f"  Type: {file_info.get('type', 'unknown')}")
        print(f"  Last modified: {file_info.get('updated', 'unknown')}")
        
        # Try to read first few bytes to check file signature
        try:
            with fs.open(filename, 'rb') as f:
                first_bytes = f.read(16)
                print(f"  First 16 bytes (hex): {first_bytes.hex()}")
                print(f"  First 16 bytes (ascii): {first_bytes}")
                
                # Check for common file signatures
                if first_bytes.startswith(b'\x89HDF'):
                    print("  ✓ HDF5 signature detected")
                elif first_bytes.startswith(b'CDF'):
                    print("  ✓ NetCDF signature detected")
                else:
                    print("  ⚠️  Unknown file signature")
                    
        except Exception as e:
            print(f"  ❌ Error reading file bytes: {e}")
            
        # Try downloading to temp file
        try:
            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as temp_file:
                fs.get(filename, temp_file.name)
                temp_size = os.path.getsize(temp_file.name)
                print(f"  Downloaded size: {temp_size} bytes")
                
                if temp_size != file_info.get('size', 0):
                    print(f"  ⚠️  Size mismatch during download!")
                
                # Try to read with different tools
                try:
                    import h5py
                    with h5py.File(temp_file.name, 'r') as h5f:
                        print(f"  ✓ Can open as HDF5, keys: {list(h5f.keys())}")
                except Exception as e:
                    print(f"  ❌ Cannot open as HDF5: {e}")
                
                try:
                    import netCDF4
                    with netCDF4.Dataset(temp_file.name, 'r') as nc:
                        print(f"  ✓ Can open with netCDF4, variables: {list(nc.variables.keys())}")
                except Exception as e:
                    print(f"  ❌ Cannot open with netCDF4: {e}")
                
                os.unlink(temp_file.name)
                
        except Exception as e:
            print(f"  ❌ Error downloading file: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error accessing file: {e}")
        return False

def load_spatial_hdf_using_cfunc(gcs_path):
    """
    Load spatial HDF5 data from GCS using the working cfunc approach with temporary files.
    This replicates the exact local functionality.
    """
    import tempfile
    import os
    
    logger.info(f"Loading spatial HDF5 data from {gcs_path} using cfunc")
    fs = gcsfs.GCSFileSystem()
    
    try:
        # Create a temporary file to download the HDF5 data
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            try:
                # Download the file from GCS to temporary local file
                logger.info(f"Downloading HDF5 file to temporary location: {temp_file.name}")
                fs.get(gcs_path, temp_file.name)
                
                # Verify the file was downloaded correctly
                temp_size = os.path.getsize(temp_file.name)
                logger.info(f"Downloaded HDF5 file size: {temp_size} bytes")
                
                if temp_size == 0:
                    raise ValueError("Downloaded HDF5 file is empty")
                
                # Now use the working cfunc function with the local temporary file
                # This is the EXACT same call that works locally
                spatial_data = cfunc.load_dicts_from_hdf(temp_file.name, ['zip', 'census', 'custom'])
                logger.info(f"Successfully loaded spatial data using cfunc with keys: {list(spatial_data.keys())}")
                
                return spatial_data
                
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_file.name)
                    logger.info("Cleaned up temporary HDF5 file")
                except OSError as e:
                    logger.warning(f"Could not clean up temporary file {temp_file.name}: {e}")

    except Exception as e:
        logger.error(f"Failed to load spatial HDF5 data from {gcs_path}: {e}")
        return {}


def check_gcs_files():
    """Debug function to check if required files exist on GCS"""
    fs = gcsfs.GCSFileSystem()
    
    files_to_check = [
        f"{GCS_NC_FOLDER_PATH}fluxresults1.nc",
        f"{GCS_HDF_FOLDER_PATH}spatial_data.h5",
        f"{GCS_SHAPEFILE_FOLDER_PATH}zip_code_socab.shp",
        f"{GCS_SHAPEFILE_FOLDER_PATH}census_tract_clipped.shp", 
        f"{GCS_SHAPEFILE_FOLDER_PATH}zones_partitoned.shp"
    ]
    
    for file_path in files_to_check:
        exists = fs.exists(file_path)
        print(f"File exists: {exists} - {file_path}")
        
        if not exists:
            # List what's actually in the directory
            directory = '/'.join(file_path.split('/')[:-1])
            try:
                files_in_dir = fs.ls(directory)
                print(f"Files in {directory}: {files_in_dir}")
            except Exception as e:
                print(f"Error listing directory {directory}: {e}")

def load_spatial_netcdf_data(gcs_path):
    """
    Load spatial NetCDF data from GCS.
    """
    
    logger.info(f"Loading spatial NetCDF data from {gcs_path}")
    fs = gcsfs.GCSFileSystem()
    
    spatial_data = {}
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as temp_file:
            try:
                # Download the NetCDF file from GCS
                logger.info(f"Downloading NetCDF file to temporary location: {temp_file.name}")
                fs.get(gcs_path, temp_file.name)
                
                downloaded_size = os.path.getsize(temp_file.name)
                logger.info(f"Downloaded NetCDF file size: {downloaded_size} bytes")
                
                # Load the NetCDF dataset
                with xr.open_dataset(temp_file.name) as ds:
                    logger.info(f"NetCDF data variables: {list(ds.data_vars)}")
                    logger.info(f"NetCDF coordinates: {list(ds.coords)}")
                    
                    # Convert each data variable to DataFrame
                    for var_name in ds.data_vars:
                        var = ds[var_name]
                        logger.info(f"Processing {var_name} with shape {var.shape} and dims {var.dims}")
                        
                        if len(var.dims) == 2:
                            # Convert to DataFrame
                            if var.dims[1] == 'time':
                                # Flux data: (regions, time)
                                df = var.to_pandas()
                            elif var.dims[1] == 'coords':
                                # Centroids: (regions, coords)
                                df = var.to_pandas()
                            else:
                                # Fallback: convert with default indexing
                                df = var.to_pandas()
                            
                            spatial_data[var_name] = df
                            logger.info(f"Converted {var_name} to DataFrame with shape {df.shape}")
                            
                            if hasattr(df.columns, 'dtype') and 'datetime' in str(df.columns.dtype):
                                logger.info(f"  Time range: {df.columns[0]} to {df.columns[-1]}")
                            
                            logger.info(f"  Index name: {df.index.name}")
                            logger.info(f"  Sample indices: {df.index[:3].tolist()}")
                        else:
                            logger.warning(f"Skipping {var_name} with unexpected dimensions: {var.dims}")
                
                logger.info(f"Successfully loaded {len(spatial_data)} datasets from NetCDF")
                
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_file.name)
                    logger.info("Cleaned up temporary NetCDF file")
                except OSError:
                    pass

    except Exception as e:
        logger.error(f"Failed to load spatial NetCDF data from {gcs_path}: {e}")
        return {}

    return spatial_data



def manually_reconstruct_dataframe(hdf_file_path, dataset_path, category):
    """
    Manually reconstruct DataFrame from HDF5 components when pd.HDFStore fails.
    """
    try:
        logger.info(f"Manually reconstructing DataFrame for {dataset_path}")
        
        with h5py.File(hdf_file_path, 'r') as f:
            # Remove leading slash for h5py access
            clean_path = dataset_path[1:] if dataset_path.startswith('/') else dataset_path
            
            if clean_path not in f:
                logger.error(f"Dataset {clean_path} not found in HDF5 file")
                return None
                
            dataset_group = f[clean_path]
            
            # Read the DataFrame components
            values = dataset_group['block0_values'][:]
            
            # Read and process index (should be geographic IDs)
            if 'axis0' in dataset_group:
                index_data = dataset_group['axis0'][:]
                if index_data.dtype.kind == 'S':  # String/bytes
                    index_data = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in index_data]
            else:
                index_data = range(values.shape[0])
            
            # Read and process columns (should be time data)
            if 'axis1' in dataset_group:
                column_data = dataset_group['axis1'][:]
                logger.info(f"Raw column data type: {column_data.dtype}")
                logger.info(f"Raw column data sample: {column_data[:5]}")
                
                if column_data.dtype.kind == 'S':  # String/bytes
                    column_data = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in column_data]
                    logger.info(f"Decoded column data sample: {column_data[:5]}")
                
                # Try to convert to datetime - the key fix for the datetime64[ns] issue
                try:
                    # First try: assume these are Unix timestamps in seconds
                    if isinstance(column_data[0], (int, float)) or (isinstance(column_data[0], str) and column_data[0].isdigit()):
                        numeric_data = [float(x) for x in column_data]
                        column_data = pd.to_datetime(numeric_data, unit='s')
                        logger.info(f"Converted numeric timestamps to datetime")
                    else:
                        # Second try: direct datetime conversion
                        column_data = pd.to_datetime(column_data)
                        logger.info(f"Converted string data to datetime")
                    
                    logger.info(f"Final datetime range: {column_data[0]} to {column_data[-1]}")
                    
                except Exception as e:
                    logger.warning(f"Could not convert columns to datetime: {e}")
                    # Keep as original format if conversion fails
                    pass
            else:
                column_data = range(values.shape[1])
            
            # Create the DataFrame
            df = pd.DataFrame(values, index=index_data, columns=column_data)
            
            # Set proper index name based on category
            if category == 'zip':
                df.index.name = 'ZIP_CODE'
            elif category == 'census':
                df.index.name = 'TRACTCE'
            elif category == 'custom':
                df.index.name = 'Zones'
            
            logger.info(f"Reconstructed DataFrame shape: {df.shape}")
            logger.info(f"Index type: {type(df.index)}")
            logger.info(f"Column type: {type(df.columns)}")
            logger.info(f"Sample index values: {df.index[:3].tolist()}")
            logger.info(f"Sample column values: {df.columns[:3].tolist()}")
            
            return df
            
    except Exception as e:
        logger.error(f"Manual reconstruction failed for {dataset_path}: {e}")
        return None



def init():
    """Initialize flux forecast components with NetCDF-based spatial data loading."""
    global FLUX_DATA, UNCERTAINTY_DATA, DATES, LAT, LON, LAT_GRID, LON_GRID, available_dates
    global ZIP_DATA, CENSUS_DATA, CUSTOM_DATA
    
    logger.info("--- Starting flux hindcast data initialization with NetCDF ---")
    check_gcs_files()
    
    try:
        # Initialize GCS filesystem
        fs = gcsfs.GCSFileSystem()
        
        # --- Step 1: Load NetCDF data ---
        filename = f"{GCS_NC_FOLDER_PATH}fluxresults1.nc" 
        logger.info(f"Loading NetCDF data from: {filename}")
        
        if not fs.exists(filename):
            logger.error(f"NetCDF file does not exist on GCS: {filename}")
            return False
        
        data_dict = load_netcdf_data(filename)
        logger.info("Successfully loaded NetCDF data into dictionary.")
        
        # Unpack netCDF data into global variables
        FLUX_DATA = data_dict['flux']
        UNCERTAINTY_DATA = data_dict['uncertainty']
        DATES = data_dict['time']
        LAT = data_dict['latitude']
        LON = data_dict['longitude']
        LAT_GRID = data_dict['lat_grid']
        LON_GRID = data_dict['lon_grid']
        available_dates = [pd.to_datetime(t, unit='s') for t in DATES]
        
        logger.info(f"NetCDF loaded: {len(available_dates)} dates from {available_dates[0]} to {available_dates[-1]}")

        # --- Step 2: Load spatial aggregation data from NetCDF ---
        try:
            spatial_netcdf_filename = f"{GCS_HDF_FOLDER_PATH}spatial_data.nc"
            logger.info(f"Loading spatial NetCDF data from: {spatial_netcdf_filename}")
            
            if not fs.exists(spatial_netcdf_filename):
                logger.warning(f"Spatial NetCDF file does not exist on GCS: {spatial_netcdf_filename}")
                ZIP_DATA, CENSUS_DATA, CUSTOM_DATA = {}, {}, {}
            else:
                # Load spatial data using NetCDF
                spatial_data = load_spatial_netcdf_data(spatial_netcdf_filename)
                logger.info("Successfully loaded spatial NetCDF data.")
                
                # Store the data
                ZIP_DATA = {k: v for k, v in spatial_data.items() if k.startswith('zip_')}
                CENSUS_DATA = {k: v for k, v in spatial_data.items() if k.startswith('census_')}
                CUSTOM_DATA = {k: v for k, v in spatial_data.items() if k.startswith('custom_')}
                
                logger.info(f"Final data keys:")
                logger.info(f"  ZIP: {list(ZIP_DATA.keys())}")
                logger.info(f"  CENSUS: {list(CENSUS_DATA.keys())}")
                logger.info(f"  CUSTOM: {list(CUSTOM_DATA.keys())}")
                
                # Verify data loaded correctly
                for agg_type in ['zip', 'census', 'custom']:
                    est_key = f'{agg_type}_est_flux'
                    unc_key = f'{agg_type}_unc_flux'
                    centroid_key = f'{agg_type}_centroids'
                    
                    if est_key in spatial_data and unc_key in spatial_data:
                        est_df = spatial_data[est_key]
                        unc_df = spatial_data[unc_key]
                        logger.info(f"{agg_type} aggregation:")
                        logger.info(f"  Est flux shape: {est_df.shape}")
                        logger.info(f"  Unc flux shape: {unc_df.shape}")
                        logger.info(f"  Time range: {est_df.columns[0]} to {est_df.columns[-1]}")
                        logger.info(f"  Index name: {est_df.index.name}")
                        logger.info(f"  Sample regions: {est_df.index[:3].tolist()}")
                        
                        if centroid_key in spatial_data:
                            centroid_df = spatial_data[centroid_key]
                            logger.info(f"  Centroids shape: {centroid_df.shape}")
                    else:
                        logger.warning(f"Missing {agg_type} flux data - est_key: {est_key in spatial_data}, unc_key: {unc_key in spatial_data}")
                
                logger.info("Spatial data loading completed successfully.")
            
        except Exception:
            logger.error("Failed to load spatial aggregation data.", exc_info=True)
            ZIP_DATA, CENSUS_DATA, CUSTOM_DATA = {}, {}, {}
        
        logger.info("--- Flux hindcast initialization successful ---")
        return True
        
    except Exception:
        logger.error("--- Critical error in flux hindcast initialization ---", exc_info=True)
        return False
    
def load_spatial_hdf_data_with_time_debug(gcs_path):
    """
    Load spatial HDF5 data with detailed time field debugging.
    """
    
    logger.info(f"Loading spatial HDF5 data from {gcs_path} with time debugging")
    fs = gcsfs.GCSFileSystem()
    
    spatial_data = {}
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            try:
                fs.get(gcs_path, temp_file.name)
                downloaded_size = os.path.getsize(temp_file.name)
                logger.info(f"Downloaded HDF5 file size: {downloaded_size} bytes")
                
                # Open with h5py to examine time fields
                with h5py.File(temp_file.name, 'r') as f:
                    
                    # First, examine one flux dataset in detail for time debugging
                    if 'zip/zip_est_flux' in f:
                        logger.info("\n=== HDF5 TIME FIELD ANALYSIS (zip_est_flux) ===")
                        flux_group = f['zip/zip_est_flux']
                        
                        # Examine axis1 (columns) which should contain time info
                        if 'axis1' in flux_group:
                            axis1_data = flux_group['axis1'][:]
                            logger.info(f"HDF5 axis1 type: {type(axis1_data)}")
                            logger.info(f"HDF5 axis1 dtype: {axis1_data.dtype}")
                            logger.info(f"HDF5 axis1 shape: {axis1_data.shape}")
                            logger.info(f"HDF5 raw axis1 values (first 10): {axis1_data[:10]}")
                            logger.info(f"HDF5 axis1 as strings (first 10): {[str(x) for x in axis1_data[:10]]}")
                            
                            # Try different conversion approaches
                            if axis1_data.dtype.kind == 'S':  # String/bytes
                                decoded_values = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in axis1_data[:10]]
                                logger.info(f"HDF5 decoded string values: {decoded_values}")
                            elif axis1_data.dtype.kind in ['i', 'f']:  # Integer or float
                                logger.info(f"HDF5 numeric values (first 10): {axis1_data[:10]}")
                                # Try converting as timestamps
                                try:
                                    converted_times = pd.to_datetime(axis1_data[:10], unit='s')
                                    logger.info(f"HDF5 converted with unit='s': {converted_times}")
                                except Exception as e:
                                    logger.info(f"Failed unit='s' conversion: {e}")
                                
                                try:
                                    converted_times_ns = pd.to_datetime(axis1_data[:10], unit='ns')
                                    logger.info(f"HDF5 converted with unit='ns': {converted_times_ns}")
                                except Exception as e:
                                    logger.info(f"Failed unit='ns' conversion: {e}")
                    
                    # Try to load datasets using existing pandas method for comparison
                    datasets_to_try = ['zip/zip_est_flux', 'zip/zip_centroids']
                    
                    for dataset_path in datasets_to_try:
                        try:
                            logger.info(f"\nTrying to load {dataset_path} with pandas...")
                            df = pd.read_hdf(temp_file.name, key=dataset_path)
                            key_name = dataset_path.replace('/', '_')
                            spatial_data[key_name] = df
                            logger.info(f"Successfully loaded {key_name} with shape {df.shape}")
                            
                            if 'flux' in dataset_path:
                                logger.info(f"  Column type: {type(df.columns)}")
                                logger.info(f"  Column dtype: {df.columns.dtype}")
                                logger.info(f"  First 5 columns: {df.columns[:5]}")
                                
                        except Exception as e:
                            logger.warning(f"Failed to load {dataset_path} with pandas: {e}")
                
            finally:
                try:
                    os.unlink(temp_file.name)
                    logger.info("Cleaned up temporary HDF5 file")
                except OSError:
                    pass

    except Exception as e:
        logger.error(f"Failed to load spatial HDF5 data from {gcs_path}: {e}")
        return {}

    return spatial_data


def load_spatial_hdf_data(gcs_path):
    """
    Quick fix: Extract data values from HDF5 and use NetCDF time coordinates.
    """
    
    logger.info(f"Loading spatial HDF5 data with NetCDF time coordinates")
    fs = gcsfs.GCSFileSystem()
    
    spatial_data = {}
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            try:
                fs.get(gcs_path, temp_file.name)
                downloaded_size = os.path.getsize(temp_file.name)
                logger.info(f"Downloaded HDF5 file size: {downloaded_size} bytes")
                
                # Get the correct time coordinates from NetCDF globals
                if DATES is not None:
                    correct_time_coords = pd.to_datetime(DATES, unit='s')
                    logger.info(f"Using NetCDF time coordinates: {correct_time_coords[0]} to {correct_time_coords[-1]}")
                else:
                    logger.error("NetCDF DATES not available - cannot proceed")
                    return {}
                
                categories = ['zip', 'census', 'custom']
                
                for category in categories:
                    logger.info(f"Processing category: {category}")
                    
                    # Load centroids normally (they work fine)
                    centroids_path = f'/{category}/{category}_centroids'
                    try:
                        with pd.HDFStore(temp_file.name, mode='r') as store:
                            if centroids_path in store:
                                spatial_data[f'{category}_centroids'] = store[centroids_path]
                                logger.info(f"Loaded {category}_centroids")
                    except Exception as e:
                        logger.error(f"Failed to load centroids for {category}: {e}")
                    
                    # Extract flux data values and reconstruct with correct time
                    for flux_type in ['est_flux', 'unc_flux']:
                        try:
                            flux_path = f'{category}/{category}_{flux_type}'
                            
                            with h5py.File(temp_file.name, 'r') as f:
                                if flux_path in f:
                                    dataset_group = f[flux_path]
                                    values = dataset_group['block0_values'][:]
                                    
                                    # Get geographic indices (axis0)
                                    if 'axis0' in dataset_group:
                                        index_data = dataset_group['axis0'][:]
                                        if index_data.dtype.kind == 'S':
                                            index_data = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in index_data]
                                    else:
                                        index_data = range(values.shape[0])
                                    
                                    logger.info(f"Raw data shape for {category}_{flux_type}: {values.shape}")
                                    logger.info(f"Index data length: {len(index_data)}")
                                    logger.info(f"Time coords length: {len(correct_time_coords)}")
                                    
                                    # Determine correct orientation and transpose if needed
                                    if category == 'zip':
                                        # ZIP: 488x488, should be regions x time
                                        if values.shape[0] == len(correct_time_coords) and values.shape[1] == len(index_data):
                                            # Data is time x regions, need to transpose
                                            values = values.T
                                            logger.info(f"Transposed ZIP data to: {values.shape}")
                                    elif category == 'census':
                                        # Census: shape was (3750, 488), should be regions x time
                                        if values.shape[1] == len(correct_time_coords):
                                            # Data is already regions x time, no transpose needed
                                            logger.info(f"Census data orientation correct: {values.shape}")
                                        else:
                                            values = values.T
                                            logger.info(f"Transposed census data to: {values.shape}")
                                    elif category == 'custom':
                                        # Custom: shape was (5, 488), should be regions x time  
                                        if values.shape[1] == len(correct_time_coords):
                                            # Data is already regions x time, no transpose needed
                                            logger.info(f"Custom data orientation correct: {values.shape}")
                                        else:
                                            values = values.T
                                            logger.info(f"Transposed custom data to: {values.shape}")
                                    
                                    # Create DataFrame with correct time coordinates
                                    if values.shape[1] == len(correct_time_coords):
                                        df = pd.DataFrame(values, index=index_data, columns=correct_time_coords)
                                        
                                        # Set proper index name
                                        if category == 'zip':
                                            df.index.name = 'ZIP_CODE'
                                        elif category == 'census':
                                            df.index.name = 'TRACTCE'
                                        elif category == 'custom':
                                            df.index.name = 'Zones'
                                        
                                        spatial_data[f'{category}_{flux_type}'] = df
                                        logger.info(f"Successfully created {category}_{flux_type} with shape {df.shape}")
                                        logger.info(f"  Time range: {df.columns[0]} to {df.columns[-1]}")
                                        logger.info(f"  Sample regions: {df.index[:3].tolist()}")
                                    else:
                                        logger.error(f"Shape mismatch for {category}_{flux_type}: values={values.shape}, time={len(correct_time_coords)}")
                                        
                        except Exception as e:
                            logger.error(f"Failed to process {category}_{flux_type}: {e}")
                
                logger.info(f"Successfully loaded spatial data with keys: {list(spatial_data.keys())}")
                
            finally:
                try:
                    os.unlink(temp_file.name)
                    logger.info("Cleaned up temporary HDF5 file")
                except OSError:
                    pass

    except Exception as e:
        logger.error(f"Failed to load spatial HDF5 data from {gcs_path}: {e}")
        return {}

    return spatial_data

def load_netcdf_data(filename):
    """
    Load NetCDF data from GCS with robust error handling and file validation.

    Args:
        filename (str): GCS path to the NetCDF file.

    Returns:
        dict: Dictionary containing flux, uncertainty, and metadata.
    """
    import gcsfs
    import tempfile
    import os
    import shutil
    import traceback
    import xarray as xr
    import numpy as np
    
    # Initialize GCS filesystem
    fs = gcsfs.GCSFileSystem()
    
    try:
        logger.info("Starting NetCDF data loading process...")
        
        # Check if file exists first
        if not fs.exists(filename):
            raise FileNotFoundError(f"NetCDF file not found on GCS: {filename}")
        
        # Get file info to verify it's accessible
        file_info = fs.info(filename)
        file_size = file_info.get('size', 0)
        logger.info(f"NetCDF file size: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError("NetCDF file appears to be empty on GCS")
        
        # Create a temporary file with a more explicit approach
        temp_fd, temp_path = tempfile.mkstemp(suffix='.nc', prefix='netcdf_')
        logger.info(f"Created temporary file: {temp_path}")
        
        try:
            # Close the file descriptor since we'll write to it differently
            os.close(temp_fd)
            
            # Download with explicit binary mode and buffering
            logger.info(f"Starting download from GCS to temporary location...")
            
            with fs.open(filename, 'rb') as gcs_file:
                with open(temp_path, 'wb') as local_file:
                    # Copy in chunks to handle large files
                    shutil.copyfileobj(gcs_file, local_file, length=16*1024*1024)  # 16MB chunks
            
            logger.info("Download completed successfully")
            
            # Verify the downloaded file
            if not os.path.exists(temp_path):
                raise ValueError("Temporary file was not created properly")
                
            downloaded_size = os.path.getsize(temp_path)
            logger.info(f"Downloaded file size: {downloaded_size} bytes")
            
            if downloaded_size != file_size:
                raise ValueError(f"File size mismatch: expected {file_size}, got {downloaded_size}")
            
            if downloaded_size == 0:
                raise ValueError("Downloaded NetCDF file is empty")
            
            # Try to open with different engines as fallback
            dataset = None
            engines_to_try = ['h5netcdf', 'netcdf4']
            
            logger.info("Attempting to open NetCDF dataset...")
            
            for engine in engines_to_try:
                try:
                    logger.info(f"Trying engine: {engine}")
                    dataset = xr.open_dataset(temp_path, engine=engine)
                    logger.info(f"Successfully opened NetCDF with engine: {engine}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to open with engine {engine}: {e}")
                    if dataset is not None:
                        dataset.close()
                    dataset = None
                    continue
            
            if dataset is None:
                raise ValueError("Unable to open NetCDF file with any available engine")
            
            # Verify required variables exist
            required_vars = ['flux', 'uncertainty', 'time', 'lat', 'lon', 'lat_grid', 'lon_grid']
            missing_vars = [var for var in required_vars if var not in dataset.variables]
            
            if missing_vars:
                dataset.close()
                raise ValueError(f"NetCDF file missing required variables: {missing_vars}")
            
            logger.info(f"NetCDF contains variables: {list(dataset.variables.keys())}")
            logger.info(f"NetCDF dimensions: {dict(dataset.dims)}")

            # Read variables with error handling
            try:
                logger.info("Reading flux data...")
                flux = dataset['flux'].transpose('latitude', 'longitude', 'time').values.astype(np.float32)
                logger.info(f"Flux data shape: {flux.shape}")
                
                logger.info("Reading uncertainty data...")
                uncertainty = dataset['uncertainty'].transpose('latitude', 'longitude', 'time').values.astype(np.float32)
                logger.info(f"Uncertainty data shape: {uncertainty.shape}")
                
                logger.info("Reading time data...")
                time = dataset['time'].values
                logger.info(f"Time data shape: {time.shape}, dtype: {time.dtype}")
                
                logger.info("Reading coordinate data...")
                lat = dataset['lat'].values.astype(np.float32)
                lon = dataset['lon'].values.astype(np.float32)
                lat_grid = dataset['lat_grid'].values.T.astype(np.float32)
                lon_grid = dataset['lon_grid'].values.T.astype(np.float32)
                
                logger.info(f"Coordinate data shapes - lat: {lat.shape}, lon: {lon.shape}")
                logger.info(f"Grid shapes - lat_grid: {lat_grid.shape}, lon_grid: {lon_grid.shape}")
                
                logger.info("Successfully loaded all NetCDF variables")
                
                # Close the dataset
                dataset.close()
                logger.info("Dataset closed successfully")
                
                result_dict = {
                    'flux': flux,
                    'uncertainty': uncertainty,
                    'time': time,
                    'latitude': lat,
                    'longitude': lon,
                    'lat_grid': lat_grid,
                    'lon_grid': lon_grid,
                }
                
                logger.info("Created result dictionary successfully")
                return result_dict
                
            except Exception as e:
                logger.error(f"Error reading NetCDF variables: {e}")
                logger.error(f"Variable read error traceback: {traceback.format_exc()}")
                dataset.close()
                raise ValueError(f"Error reading NetCDF variables: {e}")
                
        except Exception as e:
            logger.error(f"Error during NetCDF processing: {e}")
            logger.error(f"Processing error traceback: {traceback.format_exc()}")
            raise
            
        finally:
            # Clean up the temporary file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logger.info("Cleaned up temporary NetCDF file")
            except OSError as e:
                logger.warning(f"Could not clean up temporary file {temp_path}: {e}")
                
    except Exception as e:
        logger.error(f"Critical error in load_netcdf_data: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise
    
    
def create_help_modal():
    """Create the help documentation modal for flux hindcast dashboard"""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Emissions Dashboard Help")),
            dbc.ModalBody([
                html.H5("Overview"),
                html.P("This dashboard visualizes greenhouse gas emissions across the Los Angeles region, allowing you to explore both spatial patterns and temporal evolution of fluxes and their associated uncertainties."),
                
                html.H5("Analysis Controls"),
                html.H6("Display Options:"),
                html.Ul([
                    html.Li([
                        html.Strong("Display Type: "),
                        "Toggle between 'Flux' (emission rates) and 'Uncertainty' (associated uncertainties in the estimates)."
                    ]),
                    html.Li([
                        html.Strong("Spatial Aggregation: "),
                        "Choose between native resolution (3km grid), ZIP codes, census tracts, or custom regions for analysis."
                    ]),
                    html.Li([
                        html.Strong("Temporal Aggregation: "),
                        "Select between individual timesteps or monthly averages for temporal analysis."
                    ])
                ]),
                
                html.H6("Visualization Controls:"),
                html.Ul([
                    html.Li([
                        html.Strong("Color Scale: "),
                        "Choose between variable scale (auto-adjusted to data range) or fixed scale (user-defined range)."
                    ]),
                    html.Li([
                        html.Strong("Scale Range: "),
                        "When using fixed scale, set minimum and maximum values for consistent color mapping across different views."
                    ]),
                    html.Li([
                        html.Strong("Animation Controls: "),
                        "Adjust playback speed (50-2000ms) and use play/pause buttons to animate through time periods."
                    ])
                ]),
                
                html.H5("Time Selection"),
                html.Ul([
                    html.Li([
                        html.Strong("Time Slider: "),
                        "Select a specific time period or range for analysis. The slider shows available dates in the dataset."
                    ]),
                    html.Li([
                        html.Strong("Animation: "),
                        "Use play/pause buttons to animate through time periods, with customizable speed settings."
                    ]),
                    html.Li([
                        html.Strong("Aggregation Period: "),
                        "View the currently selected time range and aggregation period below the slider."
                    ])
                ]),
                
                html.H5("Map Visualization"),
                html.Ul([
                    html.Li([
                        html.Strong("Navigation: "),
                        "Pan by dragging, zoom with mouse wheel or touchpad gestures. Double-click to reset view."
                    ]),
                    html.Li([
                        html.Strong("Interaction: "),
                        "Click on regions or grid cells to view their temporal evolution in the time series plot."
                    ]),
                    html.Li([
                        html.Strong("Data Download: "),
                        "Export current view as CSV, full dataset as NetCDF, or spatial aggregation data as ZIP file."
                    ])
                ]),
                
                html.H5("Time Series Plot"),
                html.Ul([
                    html.Li([
                        html.Strong("Domain Average: "),
                        "Shows average temporal patterns across the entire region when no specific area is selected."
                    ]),
                    html.Li([
                        html.Strong("Regional Analysis: "),
                        "Displays temporal patterns for selected regions or grid cells when clicked on the map."
                    ]),
                    html.Li([
                        html.Strong("Data Export: "),
                        "Download the displayed time series data as CSV for further analysis."
                    ])
                ]),
                
                html.H5("Additional Features"),
                html.Ul([
                    html.Li([
                        html.Strong("Panel Layout: "),
                        "Use expansion arrows to focus on either map or time series, and restore button to return to dual-panel view."
                    ]),
                    html.Li([
                        html.Strong("Reset Controls: "),
                        "Use the restart button to reset all controls and panel layout to their default values."
                    ]),
                    html.Li([
                        html.Strong("Collapsible Sections: "),
                        "Click section headers to show/hide different components of the dashboard."
                    ])
                ])
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="hindcast-help-close", className="ms-auto")
            ),
        ],
        id="hindcast-help-modal",
        size="lg",
        is_open=False,
    )

def get_spatial_data(agg_type):
    """
    Get spatial aggregation data for a specific type.
    
    Args:
        agg_type: str, one of 'zip', 'census', or 'custom'
    
    Returns:
        tuple of (estimated_flux, uncertainty_flux) DataFrames or (None, None) if not available
    """
    try:
        data_dict = {
            'zip': ZIP_DATA,
            'census': CENSUS_DATA,
            'custom': CUSTOM_DATA
        }[agg_type]
        
        if not data_dict:
            logger.warning(f"No data available for {agg_type} aggregation")
            return None, None
            
        # Get the keys with the correct prefix
        est_flux_key = f'{agg_type}_est_flux'
        unc_flux_key = f'{agg_type}_unc_flux'
        
        # Get the data from the dictionary
        est_flux = data_dict.get(est_flux_key)
        unc_flux = data_dict.get(unc_flux_key)
        
        if est_flux is None or unc_flux is None:
            logger.warning(f"Missing data for {agg_type} aggregation")
            logger.warning(f"Available keys: {list(data_dict.keys())}")
            logger.warning(f"Looking for: {est_flux_key}, {unc_flux_key}")
            return None, None
            
        logger.info(f"Retrieved {agg_type} data - est_flux: {est_flux.shape}, unc_flux: {unc_flux.shape}")
        return est_flux, unc_flux
    
    except Exception as e:
        logger.error(f"Error getting {agg_type} data: {e}")
        return None, None

def get_current_data_slice(data, time_range, temporal_agg):
    """Helper function to get the current data slice based on time range and aggregation"""
    if isinstance(time_range, list) and len(time_range) == 2:
        start_idx, end_idx = time_range
        if temporal_agg == 'single':
            return data[:, :, start_idx]
        else:
            return np.nanmean(data[:, :, start_idx:end_idx+1], axis=2)
    else:
        return data[:, :, time_range]
    
def get_layout():
    """Return the page layout for flux analysis with collapsible sections"""
    return dbc.Container([
        # Hidden stores
        dcc.Store(id='hindcast-map-state-store'),
        dcc.Store(id='hindcast-animation-status-store'),
        dcc.Store(id='expansion-state', data='none'),  # Added missing Store component
        dcc.Interval(id='hindcast-animation-interval', interval=50, disabled=True),
        
        # Download components
        html.Div([
            dcc.Download(id="hindcast-download-map-current-csv"),
            dcc.Download(id="hindcast-download-full-netcdf"),
            dcc.Download(id="hindcast-download-spatial-agg-zip"),
            dcc.Download(id="hindcast-download-timeseries-current-csv-file"),
        ]),
   
       # Header Section with modern collapsible
        dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            [
                                html.I(className="fas fa-chart-line me-2"),
                                "Emissions (Forecast)",
                                html.I(className="fas fa-chevron-down ms-2", id="header-chevron"),
                            ],
                            color="link",
                            id="header-toggle",
                            className="text-primary fs-4 text-decoration-none p-0 w-100 text-start",
                            style={"box-shadow": "none"}
                        )
                    ], width=10),
                    dbc.Col([
                        html.Div([
                            html.A(
                                dbc.Button([
                                    "Documentation ",
                                    html.I(className="fas fa-file-pdf")
                                ], color="secondary", className="me-2"),
                                href="/assets/flux_hindcast.pdf",
                                target="_blank"  # This opens the PDF in a new browser tab
                            ),
                            dbc.Button([
                                "Help ",
                                html.I(className="fas fa-question-circle")
                            ], color="secondary", className="me-2", id="hindcast-help-button"),
                            dbc.Button([
                                "Reset ",
                                html.I(className="fas fa-redo")
                            ], color="secondary", id="hindcast-restart-button")
                        ], className="d-flex justify-content-end")
                    ], width=2),
                ], className="align-items-center"),
            ]),
            dbc.Collapse(
                dbc.CardBody([
                    html.P([
                        "Analyze greenhouse gas emissions across the Los Angeles region.",
                        "Explore spatial patterns and temporal evolution."
                    ], className="text-muted mb-0")
                ]),
                id="header-collapse",
                is_open=True,
            ),
        ], className="mb-4 shadow-sm"),            
        # Analysis Controls Section
        dbc.Card([
            dbc.CardHeader([
                dbc.Button(
                    [
                        html.I(className="fas fa-sliders-h me-2"),
                        "Analysis Controls",
                        html.I(className="fas fa-chevron-down ms-2", id="controls-chevron"),
                    ],
                    color="link",
                    id="controls-toggle",
                    className="text-primary fs-4 text-decoration-none p-0 w-100 text-start",
                    style={"box-shadow": "none"}
                )
            ]),
            dbc.Collapse(
                dbc.CardBody([
                    # Control panel content (existing controls code)
                    # Analysis Controls Row
                        # Analysis Controls Row
dbc.Row([
    # Display Type Selection
    dbc.Col([
        html.Label("Display Type", className="form-label fw-bold"),
        dcc.Dropdown(
            id='hindcast-display-type',
            options=[
                {'label': 'Flux', 'value': 'flux'},
                {'label': 'Uncertainty', 'value': 'uncertainty'}
            ],
            value='flux',
            clearable=False
        )
    ], md=2),

    # Spatial Aggregation
    dbc.Col([
        html.Label("Spatial Aggregation", className="form-label fw-bold"),
        dcc.Dropdown(
            id='hindcast-spatial-agg',
            options=[
                {'label': 'Native Resolution (3km)', 'value': 'native'},
                {'label': 'ZIP Code', 'value': 'zip'},
                {'label': 'Census Tract', 'value': 'census'},
                {'label': 'Custom Regions', 'value': 'custom'}
            ],
            value='native',
            clearable=False
        )
    ], md=2),
    
    # Temporal Window
    dbc.Col([
        html.Label("Temporal Aggregation", className="form-label fw-bold"),
        dcc.Dropdown(
            id='hindcast-temporal-agg',
            options=[
                {'label': 'None', 'value': 'single'},
                {'label': 'Monthly', 'value': 'custom'}
            ],
            value='single',
            clearable=False
        )
    ], md=2),
    
    # Color Scale Type and Opacity
    dbc.Col([
        html.Label("Color Scale & Opacity", className="form-label fw-bold"),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='hindcast-scale-type',
                    options=[
                        {'label': 'Variable', 'value': 'variable'},
                        {'label': 'Fixed', 'value': 'fixed'}
                    ],
                    value='variable',
                    clearable=False
                )
            ], width=8),
            dbc.Col([
                dbc.Input(
                    id='hindcast-transparency',
                    type='number',
                    min=0,
                    max=1,
                    step=0.1,
                    value=0.7,
                    size='sm',
                    placeholder='Opacity'
                )
            ], width=4)
        ], className="g-1")
    ], md=2),
    
    # Scale Range (initially hidden)
    dbc.Col([
        html.Div([
            html.Label("Scale Range", className="form-label fw-bold"),
            dbc.Row([
                dbc.Col([
                    dbc.Input(
                        id='hindcast-scale-min',
                        type='number',
                        placeholder='Min',
                        step=0.1,
                        size='sm'
                    )
                ], width=6),
                dbc.Col([
                    dbc.Input(
                        id='hindcast-scale-max',
                        type='number', 
                        placeholder='Max',
                        step=0.1,
                        size='sm'
                    )
                ], width=6)
            ], className="g-1")
        ], id='hindcast-scale-range-container', style={'display': 'none'})
    ], md=2),
    
        # Animation Controls
        dbc.Col([
            html.Label("Animation & Speed (ms)", className="form-label fw-bold"),
            dbc.Row([
                # Play/Pause buttons
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-play me-3", 
                              id="hindcast-play-button",
                              style={"cursor": "pointer"}),
                        html.I(className="fas fa-pause", 
                              id="hindcast-pause-button",
                              style={"cursor": "pointer"})
                    ], className="d-flex align-items-center justify-content-center")
                ], width=3),
                # Speed control with label
                dbc.Col([
                    dbc.Input(
                        id='hindcast-animation-speed',
                        type='number',
                        min=50,
                        max=2000,
                        step=50,
                        value=500,
                        size='sm'
                    )
                ], width=7)
            ], className="g-2 align-items-center")
        ], md=2)
    ], className="g-2 mb-3"),
                    
                    ###################################
                    # Time Slider Row (existing code)
                    dbc.Row([
                        dbc.Col([
                            dcc.RangeSlider(
                                id='hindcast-time-slider',
                                min=0,
                                max=len(available_dates) - 1 if len(available_dates) > 0 else 100,
                                step=1,
                                value=[0, len(available_dates) - 1] if len(available_dates) > 0 else [0, 100],
                                marks={
                                    i: {
                                        'label': available_dates[i].strftime('%Y-%m'),
                                        'style': {
                                            'color': '#333',
                                            'font-weight': 'bold',
                                            'padding-top': '10px',
                                            'white-space': 'nowrap',
                                            'font-size': '11px'
                                        }
                                    }
                                    for i in (
                                        range(len(available_dates)) if len(available_dates) <= 8
                                        else [
                                            int(i * (len(available_dates) - 1) / 7)
                                            for i in range(8)
                                        ]
                                    )
                                },
                                className="mt-3"
                            ),
                            # Time Period Display
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Time Period", className="form-label fw-bold mt-3")
                                ], width="auto"),
                                dbc.Col([
                                    html.Div(
                                        id='hindcast-aggregation-period-text', 
                                        className="text-muted mt-3"
                                    )
                                ], width="auto", className="ms-auto")
                            ], className="g-0")
                        ], width=12)
                    ])
                ]),
                id="controls-collapse",
                is_open=True,
            ),
        ], className="mb-4 shadow-sm"),
        
        # Main Content Area with Flex Layout
        html.Div([
            # Left Panel (Map)
            html.Div(
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-map-marked-alt me-2"),
                                        "Spatial Distribution",
                                        html.I(className="fas fa-chevron-down ms-2", id="map-chevron"),
                                    ],
                                    color="link",
                                    id="map-toggle",
                                    className="text-primary fs-4 text-decoration-none p-0 w-100 text-start",
                                    style={"box-shadow": "none"}
                                )
                            ], width=10),
                            dbc.Col([
                                dbc.ButtonGroup([
                                    dbc.DropdownMenu(
                                        [
                                            dbc.DropdownMenuItem("Current View (CSV)", 
                                                id="hindcast-trigger-download-map-current-csv"),
                                            dbc.DropdownMenuItem(divider=True),
                                            dbc.DropdownMenuItem("Full Record (NetCDF)", 
                                                id="hindcast-trigger-download-full-netcdf"),
                                            dbc.DropdownMenuItem(divider=True),
                                            dbc.DropdownMenuItem("Spatial Aggregation Data (ZIP)", 
                                                id="hindcast-trigger-download-spatial-agg-zip"),
                                        ],
                                        label=[html.I(className="fas fa-download me-1"), "Data"],
                                        color="secondary",
                                        size="sm"
                                    )
                                ])
                            ], width=2),
                        ], className="align-items-center"),
                    ]),
                    dbc.Collapse(
                        dbc.CardBody([
                            dcc.Graph(
                                id='hindcast-flux-map',
                                config={
                                    'modeBarButtonsToRemove': ['autoScale2d', 'select2d', 'lasso2d'],
                                    'displaylogo': False,
                                    'scrollZoom': True
                                },
                                style={'height': '65vh'}
                            )
                        ]),
                        id="map-collapse",
                        is_open=True,
                    ),
                ]),
                id="left-panel",
                className="panel-transition flex-grow-1",
                style={'flex': '1', 'minWidth': '0'}
            ),
            
            # Expansion Controls
            html.Div([
                html.Button("→", id="expand-left", className="expand-button"),
                html.Button("←", id="expand-right", className="expand-button")
            ], 
            style={'width': '30px'}, 
            className="d-flex flex-column justify-content-center"
            ),
            
            # Right Panel (Time Series)
            html.Div(
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-clock me-2"),
                                        "Temporal Evolution",
                                        html.I(className="fas fa-chevron-down ms-2", id="timeseries-chevron"),
                                    ],
                                    color="link",
                                    id="timeseries-toggle",
                                    className="text-primary fs-4 text-decoration-none p-0 w-100 text-start",
                                    style={"box-shadow": "none"}
                                )
                            ], width=10),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="fas fa-download me-1"), "Data"],
                                    id="hindcast-trigger-download-timeseries-current-csv",
                                    color="secondary",
                                    size="sm"
                                )
                            ], width=2),
                        ], className="align-items-center"),
                    ]),
                    dbc.Collapse(
                        dbc.CardBody([
                            dcc.Graph(
                                id='hindcast-flux-timeseries',
                                style={'height': '65vh'}
                            )
                        ]),
                        id="timeseries-collapse",
                        is_open=True,
                    ),
                ]),
                id="right-panel",
                className="panel-transition flex-grow-1",
                style={'flex': '1', 'minWidth': '0'}
            )
        ], className="d-flex gap-2", style={'height': '65vh'}),
        
        # Restore button
        html.Button(
            "Restore Panels",
            id="restore-button",
            className="restore-button",
            style={'display': 'none'}
        ),
        create_help_modal(),
        
    ], fluid=True, className="px-4 py-3")

def register_callbacks(app):
    
    
    @app.callback(
    [Output('hindcast-display-type', 'value'),
     Output('hindcast-spatial-agg', 'value'),
     Output('hindcast-temporal-agg', 'value'),
     Output('hindcast-scale-type', 'value'),
     Output('hindcast-scale-min', 'value'),
     Output('hindcast-scale-max', 'value'),
     Output('hindcast-animation-status-store', 'data', allow_duplicate=True),
     Output('hindcast-animation-speed', 'value'),
     Output('hindcast-time-slider', 'value', allow_duplicate=True),
     Output('left-panel', 'style', allow_duplicate=True),
     Output('right-panel', 'style', allow_duplicate=True),
     Output('restore-button', 'style', allow_duplicate=True),
     Output('expansion-state', 'data', allow_duplicate=True),
     Output('hindcast-map-state-store', 'data', allow_duplicate=True)],  # Added this output
    Input('hindcast-restart-button', 'n_clicks'),
    prevent_initial_call=True
)
    def reset_to_initial_state(n_clicks):
        """Reset all controls to their initial values and restore panel layout"""
        
        base_style = {'minWidth': '0'}
        initial_map_state = 'initial'  # Special flag to trigger initial state calculation
        
        return (
            'flux',  # display type
            'native',  # spatial aggregation
            'single',  # temporal aggregation
            'variable',  # scale type
            None,  # scale min
            None,  # scale max
            {'playing': False},  # animation status
            500,  # animation speed
            [0, len(available_dates) - 1] if len(available_dates) > 0 else [0, 100],  # time slider
            {'flex': '1', **base_style},  # left panel style
            {'flex': '1', **base_style},  # right panel style
            {'display': 'none'},  # restore button style
            'none',  # expansion state
            initial_map_state  # map state
        )
    
    
    # Add panel expansion callback
    @app.callback(
        [Output('left-panel', 'style', allow_duplicate=True),
         Output('right-panel', 'style', allow_duplicate=True),
         Output('restore-button', 'style', allow_duplicate=True),
         Output('expansion-state', 'data', allow_duplicate=True)],
        [Input('expand-left', 'n_clicks'),
         Input('expand-right', 'n_clicks'),
         Input('restore-button', 'n_clicks')],
        [State('expansion-state', 'data')],
        prevent_initial_call=True
    )
    def handle_panel_expansion(left_clicks, right_clicks, restore_clicks, current_state):
        ctx = dash.callback_context
        if not ctx.triggered:
            return (
                {'flex': '1', 'minWidth': '0'},
                {'flex': '1', 'minWidth': '0'},
                {'display': 'none'},
                'none'
            )
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        base_style = {'minWidth': '0'}
        
        if button_id == 'restore-button':
            return (
                {'flex': '1', **base_style},
                {'flex': '1', **base_style},
                {'display': 'none'},
                'none'
            )
        elif button_id == 'expand-left':
            if current_state != 'left':
                return (
                    {'flex': '1', **base_style},
                    {'display': 'none'},
                    {'display': 'block', 'position': 'fixed', 'bottom': '20px', 'right': '20px'},
                    'left'
                )
        elif button_id == 'expand-right':
            if current_state != 'right':
                return (
                    {'display': 'none'},
                    {'flex': '1', **base_style},
                    {'display': 'block', 'position': 'fixed', 'bottom': '20px', 'right': '20px'},
                    'right'
                )
        
        # Return current state if no change
        return (
            {'flex': '1', **base_style},
            {'flex': '1', **base_style},
            {'display': 'none'},
            'none'
        )
    
    
    
    # Header collapse callback
    @app.callback(
        [Output("header-collapse", "is_open",allow_duplicate=True),
         Output("header-chevron", "className",allow_duplicate=True)],
        [Input("header-toggle", "n_clicks")],
        [State("header-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_header(n_clicks, is_open):
        if n_clicks:
            return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
        return is_open, "fas fa-chevron-down ms-2"
    
    # Controls collapse callback
    @app.callback(
        [Output("controls-collapse", "is_open",allow_duplicate=True),
         Output("controls-chevron", "className",allow_duplicate=True)],
        [Input("controls-toggle", "n_clicks")],
        [State("controls-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_controls(n_clicks, is_open):
        if n_clicks:
            return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
        return is_open, "fas fa-chevron-down ms-2"
    
    # Map collapse callback
    @app.callback(
        [Output("map-collapse", "is_open",allow_duplicate=True),
         Output("map-chevron", "className",allow_duplicate=True)],
        [Input("map-toggle", "n_clicks")],
        [State("map-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_map(n_clicks, is_open):
        if n_clicks:
            return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
        return is_open, "fas fa-chevron-down ms-2"
    
    # Time series collapse callback
    @app.callback(
        [Output("timeseries-collapse", "is_open",allow_duplicate=True),
         Output("timeseries-chevron", "className",allow_duplicate=True)],
        [Input("timeseries-toggle", "n_clicks")],
        [State("timeseries-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_timeseries(n_clicks, is_open):
        if n_clicks:
            return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
        return is_open, "fas fa-chevron-down ms-2"
    
    
    @app.callback(
    [
        Output('hindcast-flux-map', 'figure'),
        Output('hindcast-flux-timeseries', 'figure'),
        Output('hindcast-map-state-store', 'data', allow_duplicate=True)
    ],
    [
        Input('hindcast-display-type', 'value'),
        Input('hindcast-spatial-agg', 'value'),
        Input('hindcast-temporal-agg', 'value'),
        Input('hindcast-time-slider', 'value'),
        Input('hindcast-flux-map', 'clickData'),
        Input('hindcast-animation-status-store', 'data'),
        Input('hindcast-scale-type', 'value'),
        Input('hindcast-scale-min', 'value'),
        Input('hindcast-scale-max', 'value'),
        Input('hindcast-transparency', 'value')
    ],
    [
        State('hindcast-flux-map', 'relayoutData'),
        State('hindcast-map-state-store', 'data')
    ],
    prevent_initial_call='initial_duplicate'
)
    
    
    def update_dashboard(display_type, spatial_agg, temporal_agg, time_range, 
                    click_data, animation_status, scale_type, scale_min, scale_max,
                    transparency, relayout_data, map_state):
        """Update both map and time series based on user interactions"""
        
        # Initialize figures
        map_fig = go.Figure()
        timeseries_fig = go.Figure()
        
        colorbar_settings = dict(
            orientation='v',
            thickness=20,
            len=0.9,
            y=0.5,
            yanchor='middle',
            x=1.02,
            xanchor='left',
            title=dict(
                text='µmol m² sec⁻²' if display_type == 'flux' else 'Uncertainty',
                side='right'
            ),
            tickfont=dict(size=10)
        )
        
        try:
            # Get data based on display type and spatial aggregation                       
            
            if spatial_agg == 'native':
                # Use raw netCDF data for native resolution
                data = FLUX_DATA if display_type == 'flux' else UNCERTAINTY_DATA
                
                # For map: Always compute mean over slider range
                current_data = data[:, :, time_range[0]:time_range[1]+1].mean(axis=2)
                
                # Create grid cells for native resolution
                features = []
                values = []
                customdata = []  # For additional hover data
                
                for i in range(len(LAT_GRID)-1):
                    for j in range(len(LON_GRID[0])-1):
                        polygon = [[
                            [LON_GRID[i,j], LAT_GRID[i,j]],
                            [LON_GRID[i,j+1], LAT_GRID[i,j+1]],
                            [LON_GRID[i+1,j+1], LAT_GRID[i+1,j+1]],
                            [LON_GRID[i+1,j], LAT_GRID[i+1,j]],
                            [LON_GRID[i,j], LAT_GRID[i,j]]
                        ]]
                        
                        grid_id = f"{i}_{j}"
                        value = current_data[i,j]
                        features.append({
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": polygon
                            },
                            "id": grid_id
                        })
                        values.append(value)
                        customdata.append([LAT_GRID[i,j], LON_GRID[i,j]])  # Store lat/lon for hover
                
                geojson = {
                    "type": "FeatureCollection",
                    "features": features
                }
                
                # Create base choroplethmapbox arguments
                choroplethmapbox_args = {
                    'geojson': geojson,
                    'locations': [f["id"] for f in features],
                    'z': values,
                    'colorscale': 'Turbo',
                    'marker_opacity': transparency if transparency is not None else 0.7,
                    'showscale': True,
                    'colorbar': colorbar_settings,
                    'customdata': customdata,
                    'hovertemplate': (
                        "Grid ID: %{location}<br>" +
                        "Value: %{z:.4f}<br>" +
                        "Lat: %{customdata[0]:.4f}<br>" +
                        "Lon: %{customdata[1]:.4f}" +
                        "<extra></extra>"
                    )
                }
                
                # Add fixed scale if specified
                if scale_type == 'fixed' and scale_min is not None and scale_max is not None:
                    choroplethmapbox_args.update({
                        'zmin': float(scale_min),
                        'zmax': float(scale_max)
                    })
                
                map_fig.add_trace(go.Choroplethmapbox(**choroplethmapbox_args))            
            
            else:
                # Get pre-computed spatial data from NetCDF
                est_flux, unc_flux = get_spatial_data(spatial_agg)
                
                if est_flux is None or unc_flux is None:
                    raise ValueError(f"No data available for {spatial_agg} aggregation")
                
                # Choose appropriate DataFrame based on display type
                df = est_flux if display_type == 'flux' else unc_flux
                
                # Get boundaries for the chosen aggregation from GCS
                GCS_SHAPEFILE_FOLDER_PATH = "gs://la-megacity-dashboard-data-1/data/shapefiles/"
                boundaries_key = {
                    'zip': 'zip_code_socab',
                    'census': 'census_tract_clipped',
                    'custom': 'zones_partitoned'
                }[spatial_agg]
                
                gpkg_names = {
                    'zip_code_socab': 'zip_code_socab.gpkg',
                    'census_tract_clipped': 'census_tract_clipped.gpkg',
                    'zones_partitoned': 'zones_partitoned.gpkg'
                }
                gpkg_path = f"{GCS_SHAPEFILE_FOLDER_PATH}{gpkg_names[boundaries_key]}"
                # Use the new function to load GeoPackage from GCS
                boundaries = load_gpkg_from_gcs(gpkg_path)
                
                # Add the feature ID mapping
                feature_id_mapping = {
                    'zip': 'ZIP_CODE',
                    'census': 'TRACTCE',
                    'custom': 'Zones'
                }
                feature_id_key = f"properties.{feature_id_mapping[spatial_agg]}"
                
                # For map: Always compute mean over slider range
                selected_dates = pd.to_datetime(DATES[time_range[0]:time_range[1]+1], unit='s')
                current_data = df[selected_dates].mean(axis=1)
                
                # Create base choroplethmapbox arguments
                choroplethmapbox_args = {
                    'geojson': boundaries.__geo_interface__,
                    'featureidkey': feature_id_key,
                    'locations': current_data.index,
                    'z': current_data.values,
                    'colorscale': 'Turbo',
                    'marker_opacity': transparency if transparency is not None else 0.7,
                    'showscale': True,
                    'colorbar': colorbar_settings,
                    'hovertemplate': (
                        f"{spatial_agg.title()} ID: %{{location}}<br>" +
                        "Value: %{z:.4f}" +
                        "<extra></extra>"
                    )
                }
                
                # Add fixed scale if specified
                if scale_type == 'fixed' and scale_min is not None and scale_max is not None:
                    choroplethmapbox_args.update({
                        'zmin': float(scale_min),
                        'zmax': float(scale_max)
                    })
                    
                map_fig.add_trace(go.Choroplethmapbox(**choroplethmapbox_args))
    
            # Calculate bounds for optimal view
            lat_bounds = [np.min(LAT_GRID), np.max(LAT_GRID)]
            lon_bounds = [np.min(LON_GRID), np.max(LON_GRID)]
            
            center_lat = (lat_bounds[0] + lat_bounds[1]) / 2
            center_lon = (lon_bounds[0] + lon_bounds[1]) / 2
            
            # Calculate zoom level to fit the data extent
            lon_range = lon_bounds[1] - lon_bounds[0]
            lat_range = lat_bounds[1] - lat_bounds[0]
            zoom = min(
                math.log2(360 / lon_range) - 1,
                math.log2(180 / lat_range) - 1
            )
            zoom = max(min(zoom, 10), 7)  # Constrain zoom between 7 and 10
            
            # Base layout settings for map
            map_layout = dict(
                autosize=True,
                mapbox=dict(
                    style='open-street-map',
                    zoom=zoom,
                    center=dict(lat=center_lat, lon=center_lon),
                ),
                margin=dict(l=0, r=70, t=0, b=0),
                uirevision='constant',
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            # Base layout for time series
            timeseries_layout = dict(
                autosize=True,
                xaxis_title="Date",
                yaxis_title="Emissions (µmol m² sec⁻²)" if display_type == 'flux' else 'Uncertainty (µmol m² sec⁻²)',
                margin=dict(l=50, r=30, t=40, b=40),
                paper_bgcolor='white',
                plot_bgcolor='rgba(240, 242, 245, 0.8)',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    gridwidth=1,
                    showline=True,
                    linecolor='rgba(128, 128, 128, 0.8)',
                    type='date'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    gridwidth=1,
                    showline=True,
                    linecolor='rgba(128, 128, 128, 0.8)'
                ),
                showlegend=True,
                legend=dict(
                    x=0.01,
                    y=0.99,
                    xanchor='left',
                    yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='rgba(128, 128, 128, 0.2)',
                    borderwidth=1
                )
            )
            
            # Apply layouts
            map_fig.update_layout(map_layout)
            timeseries_fig.update_layout(timeseries_layout)
            
            # Handle time series display
            if not click_data:
                if spatial_agg == 'native':
                    # Show domain-wide average for native resolution
                    data = FLUX_DATA if display_type == 'flux' else UNCERTAINTY_DATA
                    domain_means = np.nanmean(np.nanmean(data, axis=0), axis=0)
                    dates = pd.to_datetime(DATES, unit='s')
                    df_ts = pd.DataFrame({'date': dates, 'value': domain_means})
                    
                    # Apply monthly aggregation only if selected
                    if temporal_agg != 'single':
                        df_ts = df_ts.set_index('date').resample('M').mean().reset_index()
                    
                    timeseries_fig.add_trace(go.Scatter(
                        x=df_ts['date'],
                        y=df_ts['value'],
                        mode='lines+markers',
                        name='Domain Average',
                        line=dict(color='rgba(0,0,255,0.7)')
                    ))
                    timeseries_fig.update_layout(title="Domain-wide Average Time Series")
                else:
                    # Show average across all regions for aggregated data
                    est_flux, unc_flux = get_spatial_data(spatial_agg)
                    if est_flux is not None and unc_flux is not None:
                        df = est_flux if display_type == 'flux' else unc_flux
                        dates = df.columns
                        region_mean = df.mean()
                        
                        # Create DataFrame for time series
                        temp_df = pd.DataFrame({'date': dates, 'value': region_mean})
                        
                        # Apply monthly aggregation only if selected
                        if temporal_agg != 'single':
                            temp_df = temp_df.set_index('date').resample('M').mean().reset_index()
                        
                        timeseries_fig.add_trace(go.Scatter(
                            x=temp_df['date'],
                            y=temp_df['value'],
                            mode='lines+markers',
                            name='Regional Average',
                            line=dict(color='rgba(0,0,255,0.7)')
                        ))
                        timeseries_fig.update_layout(title="Regional Average Time Series")
            else:
                # Handle clicked data with proper type checking
                if spatial_agg == 'native':
                    # Handle click for native resolution
                    data = FLUX_DATA if display_type == 'flux' else UNCERTAINTY_DATA
                    location_id = click_data['points'][0]['location']
                    
                    # Ensure location_id is a string and can be split
                    if not isinstance(location_id, str):
                        location_id = str(location_id)
                    
                    # Check if location_id contains underscore for splitting
                    if '_' in location_id:
                        try:
                            i, j = map(int, location_id.split('_'))
                        except (ValueError, IndexError):
                            logger.warning(f"Could not parse location_id: {location_id}")
                            return map_fig, timeseries_fig, map_state
                    else:
                        logger.warning(f"Invalid location_id format for native resolution: {location_id}")
                        return map_fig, timeseries_fig, map_state
                    
                    # Get full time series for clicked location
                    time_series = data[i, j, :]
                    dates = pd.to_datetime(DATES, unit='s')
                    df_ts = pd.DataFrame({'date': dates, 'value': time_series})
                    
                    # Apply monthly aggregation only if selected
                    if temporal_agg != 'single':
                        df_ts = df_ts.set_index('date').resample('M').mean().reset_index()
                    
                    title = f"{display_type.capitalize()} Time Series for Latitude {LAT_GRID[i,j]:.2f}°N, Longitude {LON_GRID[i,j]:.2f}°W"
                else:
                    # Handle click for aggregated data
                    est_flux, unc_flux = get_spatial_data(spatial_agg)
                    if est_flux is not None and unc_flux is not None:
                        df = est_flux if display_type == 'flux' else unc_flux
                        location_id = click_data['points'][0]['location']
                        
                        # Handle different data types for location_id
                        try:
                            # For custom regions, ensure location_id is an integer
                            if spatial_agg == 'custom':
                                if isinstance(location_id, str):
                                    location_id = int(location_id)
                                # location_id should already be int for custom regions
                            else:
                                # For zip and census, ensure location_id is a string
                                if not isinstance(location_id, str):
                                    location_id = str(location_id)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not convert location_id {location_id} for {spatial_agg}: {e}")
                            return map_fig, timeseries_fig, map_state
                        
                        # Check if location_id exists in the DataFrame
                        if location_id not in df.index:
                            logger.warning(f"Location {location_id} not found in {spatial_agg} data")
                            return map_fig, timeseries_fig, map_state
                        
                        # Get full time series for clicked location
                        time_series = df.loc[location_id]
                        dates = df.columns
                        df_temp = pd.DataFrame({'date': dates, 'value': time_series.values})
                        
                        # Apply monthly aggregation only if selected
                        if temporal_agg != 'single':
                            df_temp = df_temp.set_index('date').resample('M').mean().reset_index()
                        
                        df_ts = df_temp  # Reassign for consistent plotting
                        
                        # Create title based on aggregation type and display type
                        if spatial_agg == 'zip':
                            title = f"{display_type.capitalize()} Time Series for ZIP Code {location_id}"
                        elif spatial_agg == 'census':
                            title = f"{display_type.capitalize()} Time Series for Census Tract {location_id}"
                        else:  # custom regions
                            title = f"{display_type.capitalize()} Time Series for Region {location_id}"
                    else:
                        logger.warning(f"No flux data available for {spatial_agg}")
                        return map_fig, timeseries_fig, map_state
                
                timeseries_fig.add_trace(go.Scatter(
                    x=df_ts['date'],
                    y=df_ts['value'],
                    mode='lines+markers',
                    name='Selected Region',
                    line=dict(color='rgba(255,0,0,0.7)')
                ))
                timeseries_fig.update_layout(title=title)
            
            # Update map view state if necessary
            if relayout_data and 'mapbox.center' in relayout_data:
                map_layout['mapbox'].update(
                    center=relayout_data['mapbox.center'],
                    zoom=relayout_data['mapbox.zoom']
                )
            elif map_state:
                map_layout['mapbox'].update(map_state)
            
            # Final layout updates
            map_fig.update_layout(map_layout)
            
            return map_fig, timeseries_fig, map_layout['mapbox']
    
        except Exception as e:
            logger.error(f"Error in dashboard update: {e}")
            return go.Figure(), go.Figure(), dash.no_update
    
                
    # Date range display callback
    @app.callback(
    Output('hindcast-aggregation-period-text', 'children'),
    [Input('hindcast-time-slider', 'value')]
)
    def update_aggregation_period(time_range):
        """Update the displayed date range based on slider selection"""
        if not time_range or len(available_dates) == 0:
            return ""
        
        start_date = available_dates[time_range[0]]
        end_date = available_dates[time_range[1]]
        return f"Spatial Mean Period: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
    
   # Animation control callbacks
    @app.callback(
        Output('hindcast-animation-interval', 'disabled'),
        [Input('hindcast-play-button', 'n_clicks'),
         Input('hindcast-pause-button', 'n_clicks')],
        [State('hindcast-animation-interval', 'disabled')]
    )
    def toggle_animation(play_clicks, pause_clicks, current_state):
        """Toggle animation playback state"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return True  # Initially disabled
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'hindcast-play-button':
            return False  # Enable animation
        elif button_id == 'hindcast-pause-button':
            return True  # Disable animation
        
        return current_state  # Keep current state if neither button clicked
    
    @app.callback(
        Output('hindcast-animation-status-display', 'children'),
        [Input('hindcast-animation-status-store', 'data'),
         Input('hindcast-time-slider', 'value')]
    )
    def update_animation_status(animation_status, time_range):
        """Update the animation status display text"""
        if not time_range:
            return "No time period selected"
        
        # Get the current date from available_dates using time_range
        current_date = available_dates[time_range[0]].strftime('%Y-%m-%d') if available_dates else "No date"
        status = "Playing" if animation_status and animation_status.get('playing', False) else "Paused"
    
        return f"Current Date: {current_date} ({status})"
    
    @app.callback(
        [Output('hindcast-time-slider', 'value'),
         Output('hindcast-animation-status-store', 'data')],
        [Input('hindcast-animation-interval', 'n_intervals')],
        [State('hindcast-time-slider', 'value'),
         State('hindcast-animation-interval', 'disabled')]
    )
    def update_animation_frame(n_intervals, current_range, is_disabled):
        """Update the time slider position during animation"""
        if is_disabled or not current_range:
            return current_range, {'playing': False}
        
        current_idx = current_range[0]
        max_idx = len(available_dates) - 1
        
        # Move to next frame
        next_idx = current_idx + 1 if current_idx < max_idx else 0
        
        # Update the range to show just the current frame
        return [next_idx, next_idx], {'playing': True}
    
    @app.callback(
        Output("hindcast-download-full-netcdf", "data"),
        Input("hindcast-trigger-download-full-netcdf", "n_clicks"),
        State('hindcast-spatial-agg', 'value'),
        prevent_initial_call=True
    )
    def download_full_netcdf(n_clicks, spatial_agg):
        """
        Downloads the full NetCDF file from GCS to a temporary local file,
        then serves it for download.
        """
        if not n_clicks or spatial_agg != 'native':
            # Only allow download for native spatial resolution
            return None
    
        # Initialize the GCS filesystem interface
        fs = gcsfs.GCSFileSystem()
        gcs_path = f"{GCS_NC_FOLDER_PATH}fluxresults1.nc"
    
        try:
            # Create a temporary file on the local disk
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                # Download the file from GCS to the temporary local file
                fs.get(gcs_path, tmp.name)
                
                # Send the temporary local file to the user
                return dcc.send_file(tmp.name, filename="fluxresults1.nc")
    
        except Exception as e:
            print(f"Error in NetCDF download: {e}")
            return None
    
    @app.callback(
        Output("hindcast-download-spatial-agg-zip", "data"),
        Input("hindcast-trigger-download-spatial-agg-zip", "n_clicks"),
        State('hindcast-spatial-agg', 'value'),
        prevent_initial_call=True
    )
    def download_spatial_agg_zip(n_clicks):
        """
        Downloads spatial aggregation data (HDF5 and shapefiles) from GCS,
        zips them, and serves the archive for download.
        """
        if not n_clicks:
            return None
    
        # Initialize the GCS filesystem interface
        fs = gcsfs.GCSFileSystem()
    
        try:
            # Create a temporary local directory to stage files for zipping
            with tempfile.TemporaryDirectory() as temp_dir:
                
                # --- 1. Download HDF5 File ---
                gcs_hdf_path = f"{GCS_HDF_FOLDER_PATH}spatial_data.h5"
                local_hdf_path = os.path.join(temp_dir, "spatial_data.h5")
                if fs.exists(gcs_hdf_path):
                    fs.get(gcs_hdf_path, local_hdf_path)
    
                # --- 2. Download all Shapefile Components ---
                shapefile_bases = ['zip_code_socab', 'census_tract_clipped', 'zones_partitoned']
                shapefile_exts = ['.shp', '.shx', '.dbf', '.prj']
                
                for base_name in shapefile_bases:
                    for ext in shapefile_exts:
                        gcs_shapefile_path = f"{GCS_SHAPEFILE_FOLDER_PATH}{base_name}{ext}"
                        local_shapefile_path = os.path.join(temp_dir, f"{base_name}{ext}")
                        if fs.exists(gcs_shapefile_path):
                            fs.get(gcs_shapefile_path, local_shapefile_path)
    
                # --- 3. Zip the Downloaded Files ---
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # arc_name makes sure the files are in the root of the zip
                            arc_name = os.path.relpath(file_path, temp_dir) 
                            zip_file.write(file_path, arc_name)
    
                zip_buffer.seek(0)
                
                return dcc.send_bytes(zip_buffer.getvalue(), "full_spatial_data.zip")
    
        except Exception as e:
            print(f"Error in spatial aggregation data download: {e}")
            return None
    
    @app.callback(
        Output("hindcast-download-map-current-csv", "data"),
        Input("hindcast-trigger-download-map-current-csv", "n_clicks"),
        [State('hindcast-display-type', 'value'),
         State('hindcast-spatial-agg', 'value'), 
         State('hindcast-time-slider', 'value')],
        prevent_initial_call=True
    )
    def download_current_view_csv(n_clicks, display_type, spatial_agg, time_range):
        """Download time-averaged data as CSV for selected view"""
        if not n_clicks:
            return None
        
        try:
            if spatial_agg == 'native':
                # Handle native resolution data  
                data = FLUX_DATA if display_type == 'flux' else UNCERTAINTY_DATA
                current_data = data[:, :, time_range[0]:time_range[1]+1].mean(axis=2)
                
                # Create DataFrame with grid cell IDs
                rows = []
                for i in range(len(LAT_GRID)):
                    for j in range(len(LON_GRID[0])):
                        rows.append({
                            'grid_id': f"{i}_{j}",
                            f'{display_type}': current_data[i,j], 
                            'latitude': LAT_GRID[i,j],
                            'longitude': LON_GRID[i,j]
                        })
                output_df = pd.DataFrame(rows).set_index('grid_id')
                
                # Generate filename 
                dates = pd.to_datetime(DATES[time_range[0]:time_range[1]+1], unit='s')
                filename = f"native_{display_type}_{dates[0].strftime('%Y%m%d')}_{dates[-1].strftime('%Y%m%d')}.csv"
                
            else:
                # Get aggregated flux data and average over selected time range
                flux_df, _ = get_spatial_data(spatial_agg)
                
                if flux_df is None:
                    raise ValueError(f"No data available for {spatial_agg} aggregation")
                
                flux_df = flux_df if display_type == 'flux' else _
                
                # Average over time range and create output
                time_slice = flux_df.iloc[:, time_range[0]:time_range[1]+1] 
                output_df = time_slice.mean(axis=1).to_frame(f'{display_type}')
                
                # Generate filename with date range
                dates = flux_df.columns[time_range[0]:time_range[1]+1]
                filename = f"{spatial_agg}_{display_type}_{dates[0].strftime('%Y%m%d')}_{dates[-1].strftime('%Y%m%d')}.csv"
            
            return dcc.send_data_frame(output_df.to_csv, filename, index=True)
        
        except Exception as e:
            print(f"Error in CSV download: {e}")
            return None
    
    @app.callback(
        Output("hindcast-download-timeseries-current-csv-file", "data"),
        Input("hindcast-trigger-download-timeseries-current-csv", "n_clicks"),
        [State('hindcast-flux-timeseries', 'figure')],
        prevent_initial_call=True
    )
    def download_timeseries_csv(n_clicks, figure):
        if not n_clicks:
            return None
        
        try:
            # Create an empty DataFrame to store the consolidated data
            consolidated_df = pd.DataFrame()
            
            # Extract data from each trace
            for trace in figure['data']:
                dates = trace['x']
                values = trace['y']
                trace_name = trace['name']
                
                # Create a DataFrame for the current trace
                df = pd.DataFrame({'Date': dates, trace_name: values})
                
                # Merge the current trace DataFrame with the consolidated DataFrame
                if consolidated_df.empty:
                    consolidated_df = df
                else:
                    consolidated_df = pd.merge(consolidated_df, df, on='Date', how='outer')
            
            # Sort the consolidated DataFrame by date
            consolidated_df.sort_values('Date', inplace=True)
            
            # Generate filename based on data
            start_date = consolidated_df['Date'].min().split('T')[0]
            end_date = consolidated_df['Date'].max().split('T')[0]
            filename = f"timeseries_{start_date}_{end_date}.csv"
            
            return dcc.send_data_frame(consolidated_df.to_csv, filename, index=False)
        
        except Exception as e:
            print(f"Error in timeseries CSV download: {e}")
            return None
    # Toggle scale range visibility
    @app.callback(
        Output('hindcast-scale-range-container', 'style'),
        Input('hindcast-scale-type', 'value')
    )
    def toggle_scale_range(scale_type):
        return {'display': 'block'} if scale_type == 'fixed' else {'display': 'none'}
    
    # Update animation speed
    @app.callback(
        Output('hindcast-animation-interval', 'interval'),
        Input('hindcast-animation-speed', 'value')
    )
    def update_animation_speed(speed):
        return speed if speed is not None else 500
    
    @app.callback(
    Output("hindcast-help-modal", "is_open"),
    [
        Input("hindcast-help-button", "n_clicks"),
        Input("hindcast-help-close", "n_clicks")
    ],
    [State("hindcast-help-modal", "is_open")],
)
    def toggle_help_modal(help_clicks, close_clicks, is_open):
        """Toggle help modal visibility"""
        if help_clicks or close_clicks:
            return not is_open
        return is_open
    

    