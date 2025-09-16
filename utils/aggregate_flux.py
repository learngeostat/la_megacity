#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 10:58:10 2025

@author: vyadav
"""

import pandas as pd
import numpy as np
import os
import geopandas as gpd
import xarray as xr
from la_megacity.utils import constants as prm
from shapely.vectorized import contains
from scipy.spatial import cKDTree
from la_megacity.utils import conc_func as cfunc 

filename = os.path.join(prm.SITE_DATA_PATH, 'fluxresults1.nc')
#hdf_filename = os.path.join(prm.SITE_DATA_PATH, 'aggregated_data.h5')
spatial_hdf_filename = os.path.join(prm.SITE_DATA_PATH, 'spatial_data.h5')


def load_netcdf_data(filename):
    """
    Load NetCDF data and ensure it is read in [latitude, longitude, time] order.

    Args:
        filename (str): Path+filename to the NetCDF file.

    Returns:
        dict: Dictionary containing flux, uncertainty, and metadata.
    """
    # Open the NetCDF file
    dataset = xr.open_dataset(filename)

    # Read variables
    # Transpose and convert to float32 where applicable
    flux = dataset['flux'].transpose('latitude', 'longitude', 'time').values.astype(np.float32)
    uncertainty = dataset['uncertainty'].transpose('latitude', 'longitude', 'time').values.astype(np.float32)
    time = dataset['time'].values  # POSIX time remains unchanged
    lat = dataset['lat'].values.astype(np.float32)
    lon = dataset['lon'].values.astype(np.float32)
    lat_grid = dataset['lat_grid'].values.T.astype(np.float32)  # Combined transpose and float32 conversion
    lon_grid = dataset['lon_grid'].values.T.astype(np.float32)  # Combined transpose and float32 conversion

    return {
        'flux': flux,
        'uncertainty': uncertainty,
        'time': time,
        'latitude': lat,
        'longitude': lon,
        'lat_grid': lat_grid,
        'lon_grid': lon_grid,
    }


def meters_to_degrees(meters):
    """Convert meters to approximate degrees at SoCal latitude"""
    return meters / 111111.0  # At ~33.5°N, 1 degree ≈ 111111 meters

def aggregate_to_shapefile_timeseries(
    data_array, lat_grid, lon_grid, time_periods, boundaries_gdf, 
    id_column, method='hybrid', distance_threshold=7000):
    """
    Aggregate 3D data array to shape × time_periods DataFrame.

    Args:
        data_array: 3D array (lat × lon × time)
        lat_grid: 2D array of latitudes
        lon_grid: 2D array of longitudes
        time_periods: array of time values
        boundaries_gdf: GeoDataFrame with shape boundaries
        id_column: column name in boundaries_gdf containing unique identifiers
        method: 'area_weighted', 'nearest_point', or 'hybrid'
        distance_threshold: threshold in meters (default 7000m = 7km)
        
    Returns:
        DataFrame with aggregated data indexed by unique shape IDs.
    """
    # Ensure GeoDataFrame CRS is defined and in WGS84 (EPSG:4326)
    if boundaries_gdf.crs is None:
        print("CRS not defined. Setting CRS to WGS 84 (EPSG:4326).")
        boundaries_gdf.set_crs(epsg=4326, inplace=True)
    elif boundaries_gdf.crs.to_epsg() != 4326:
        raise ValueError("GeoDataFrame CRS is not EPSG:4326. Ensure the CRS is WGS 84 (EPSG:4326).")
    
    # Convert threshold to degrees for internal calculations
    threshold_degrees = meters_to_degrees(distance_threshold)
    
    # Flatten lat/lon grids
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    points = np.column_stack((lon_flat, lat_flat))
    
    # Flatten the data array
    data_flat = data_array.reshape(-1, data_array.shape[2])
    
    # Initialize results dictionary
    results = {}

    # Area-weighted or hybrid method
    if method in ['area_weighted', 'hybrid']:
        for idx, row in boundaries_gdf.iterrows():
            shape_id = row[id_column]
            polygon = row.geometry
            mask = contains(polygon, points[:, 0], points[:, 1])
            
            if mask.any():
                masked_data = data_flat[mask, :]
                results[shape_id] = np.nanmean(masked_data, axis=0)
    
    # Nearest point or hybrid method
    if method in ['nearest_point', 'hybrid']:
        remaining_shapes = set(boundaries_gdf[id_column]) - set(results.keys())
        if remaining_shapes or method == 'nearest_point':
            tree = cKDTree(points)
            for tract_id, tract_group in boundaries_gdf.groupby(id_column):
                if method == 'nearest_point' or tract_id in remaining_shapes:
                    row = tract_group.iloc[0]
                    rep_point = row.geometry.representative_point()
                    distance, nearest_idx = tree.query([rep_point.x, rep_point.y], k=1)
                    
                    if distance <= threshold_degrees:
                        results[tract_id] = data_flat[nearest_idx, :]
                    else:
                        results[tract_id] = np.full(data_array.shape[2], np.nan)
    
    # Create DataFrame
    df = pd.DataFrame(
        data=np.array([results[shape_id] for shape_id in results.keys()]),
        index=list(results.keys()),
        columns=pd.to_datetime(time_periods, unit='s')
    )
    df.index.name = id_column

    return df

def contains(polygon, x, y):
    """Vectorized point-in-polygon test"""
    from shapely.vectorized import contains as shapely_contains
    return shapely_contains(polygon, x, y)

def process_flux_and_uncertainty(data, boundaries_gdf, id_column, prefix=None, method='hybrid', distance_threshold=5000):
    """
    Process both flux and uncertainty data.

    Args:
        data: Dictionary containing flux, uncertainty, and grid data
        boundaries_gdf: GeoDataFrame with shape boundaries
        id_column: column name in boundaries_gdf containing unique identifiers
        prefix: String prefix for the output dictionary keys
        method: Aggregation method ('area_weighted', 'nearest_point', or 'hybrid')
        distance_threshold: Distance threshold in meters
    
    Returns:
        Dictionary containing estimated and uncertainty flux DataFrames
    """
    # Get results for flux
    est_flux_df = aggregate_to_shapefile_timeseries(
        data['flux'], data['lat_grid'], data['lon_grid'], 
        data['time'], boundaries_gdf, id_column,
        method=method, distance_threshold=distance_threshold
    )
    
    # Get results for uncertainty
    unc_flux_df = aggregate_to_shapefile_timeseries(
        data['uncertainty'], data['lat_grid'], data['lon_grid'], 
        data['time'], boundaries_gdf, id_column,
        method=method, distance_threshold=distance_threshold
    )
    
    # Return dictionary with prefixed keys
    prefix = f"{prefix}_" if prefix else ""
    return {
        f'{prefix}est_flux': est_flux_df,
        f'{prefix}unc_flux': unc_flux_df
    }


# Example usage:
"""
import geopandas as gpd
import os
import prm  # your parameters module

# Load boundaries
zip_boundaries = gpd.read_file(os.path.join(prm.SHAPEFILE_PATH, 'zip_code_socab.shp'))
census_boundaries = gpd.read_file(os.path.join(prm.SHAPEFILE_PATH, 'census_tract_clipped.shp'))
custom_boundaries = gpd.read_file(os.path.join(prm.SHAPEFILE_PATH, 'zones_partitoned.shp'))

# Load netCDF data
data = load_netcdf_data(filename)

# Process for each boundary type
spatial_agg_data = process_flux_and_uncertainty(data, zip_boundaries, 'ZIP_CODE')  # prefix='zip' inferred
census_agg_data = process_flux_and_uncertainty(data, census_boundaries, 'TRACTCE')  # prefix='census' inferred
custom_agg_data = process_flux_and_uncertainty(data, custom_boundaries, 'Zones')   # prefix='custom' inferred
"""

filename = os.path.join(prm.SITE_DATA_PATH, 'fluxresults1.nc')

# Load boundaries
zip_boundaries = gpd.read_file(os.path.join(prm.SHAPEFILE_PATH, 'zip_code_socab.shp'))
census_boundaries = gpd.read_file(os.path.join(prm.SHAPEFILE_PATH, 'census_tract_clipped.shp'))
custom_boundaries = gpd.read_file(os.path.join(prm.SHAPEFILE_PATH, 'zones_partitoned.shp'))

# Load netCDF data
data = load_netcdf_data(filename)

# Process for each boundary type
spatial_agg_data = process_flux_and_uncertainty(data, zip_boundaries, 'ZIP_CODE')  # prefix='zip' inferred
census_agg_data = process_flux_and_uncertainty(data, census_boundaries, 'TRACTCE')  # prefix='census' inferred
custom_agg_data = process_flux_and_uncertainty(data, custom_boundaries, 'Zones')   # prefix='custom' inferred

#raw_data_dict, background_data_dict=cfunc.load_two_dicts_from_hdf(hdf_filename)

data_dicts = {
    'zip': spatial_agg_data,
    'census': census_agg_data,
    'custom': custom_agg_data
}
#%%
cfunc.save_dicts_to_hdf(data_dicts, spatial_hdf_filename)
#%%
all_data = cfunc.load_dicts_from_hdf(spatial_hdf_filename)

#%%

def analyze_and_visualize_duplicates(boundaries_gdf, id_column):
    """
    Analyze and visualize duplicate TRACTCE values focusing on spatial characteristics.
    """
    import matplotlib.pyplot as plt
    
    # Find duplicates
    duplicate_tracts = boundaries_gdf[boundaries_gdf[id_column].duplicated(keep=False)].sort_values(id_column)
    
    print(f"\nAnalysis of {id_column} values:")
    print(f"Total records in original data: {len(boundaries_gdf)}")
    print(f"Unique {id_column} values in original: {boundaries_gdf[id_column].nunique()}")
    print(f"Number of duplicated records: {len(duplicate_tracts)}")
    
    if len(duplicate_tracts) > 0:
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: All boundaries with duplicates highlighted
        boundaries_gdf.plot(ax=ax1, color='lightgray', edgecolor='black', alpha=0.5)
        duplicate_tracts.plot(ax=ax1, color='red', edgecolor='black', alpha=0.7)
        
        # Add centroids for duplicates
        for tract_id in duplicate_tracts[id_column].unique():
            tract_parts = duplicate_tracts[duplicate_tracts[id_column] == tract_id]
            print(f"\nTRACTCE: {tract_id}")
            print(f"Number of parts: {len(tract_parts)}")
            
            # Calculate and display centroid for each part
            for idx, part in tract_parts.iterrows():
                centroid = part.geometry.centroid
                print(f"Part centroid coordinates (lon, lat): ({centroid.x:.6f}, {centroid.y:.6f})")
                ax1.plot(centroid.x, centroid.y, 'b*', markersize=10)
                ax1.annotate(f"{tract_id}-{idx}", 
                           (centroid.x, centroid.y),
                           xytext=(5, 5), textcoords='offset points')
                
                # Calculate and display area
                area_sqkm = part.geometry.area * 111.32 * 111.32 * np.cos(np.radians(centroid.y))  # Approximate conversion to km²
                print(f"Part area: {area_sqkm:.2f} km²")
        
        ax1.set_title('Duplicate Census Tracts (red)\nwith Part Centroids (blue stars)')
        
        # Plot 2: Zoom to duplicates
        duplicate_tracts.plot(ax=ax2, color='red', edgecolor='black', alpha=0.7)
        ax2.set_title('Zoomed View of Duplicate Tracts')
        
        plt.tight_layout()
        plt.show()
        
    return duplicate_tracts

census_duplicates = analyze_and_visualize_duplicates(census_boundaries, 'TRACTCE')