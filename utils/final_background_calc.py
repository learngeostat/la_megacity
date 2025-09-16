#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 14:45:33 2024

@author: vyadav
"""

import os

#os.chdir('/Users/vyadav/Previous_computer/PycharmProjects/scientificProject/')
import sys
from la_megacity.utils import background as bg
from la_megacity.utils import conc_func as cfunc 
from la_megacity.utils import constants as prm
from la_megacity.utils import fig_surface_obs as pfigure
#%%
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import copy
import h5py
import numpy as np

#%% Hours at which background and aggregation wuld be performed in UTC
# collect raw data results for all site in a dictionary. 

target_hours = [0, 1, 20, 21, 22, 23] # 
site_smooth = False
write_hdf = True
read_hdf = False
gas='co'
begin_year = prm.begin_year
end_year = prm.end_year
time_frequency=prm.time_frequency

if gas == 'co':
    sites = prm.sites_co
    site_inlet = prm.site_inlet_co
    units = 'ppb'
elif gas == 'co2':
    sites = prm.sites
    site_inlet = prm.site_inlet
    units = 'ppm'
elif gas == 'ch4':
    sites = prm.sites_ch4
    site_inlet = prm.site_inlet_ch4
    units = 'ppb'
else:
    # Handle invalid gas input
    raise ValueError(f"Invalid gas input: {gas}. Valid options are 'co', 'co2', and 'ch4'.") 


raw_dict = {}

# Configuration parameters
target_hours = [0, 1, 20, 21, 22, 23]
site_smooth = False
write_hdf = True
read_hdf = False
begin_year = prm.begin_year
end_year = prm.end_year
time_frequency = prm.time_frequency
stat_type = 'mean'

# Initialize main dictionary for all gases
all_gas_dict = {}

# Process each gas type
for gas in ['co2', 'ch4', 'co']:
    print(f"\nProcessing {gas}...")
    # Set gas-specific parameters
    if gas == 'co':
        sites = prm.sites_co
        site_inlet = prm.site_inlet_co
        units = 'ppb'
    elif gas == 'co2':
        sites = prm.sites
        site_inlet = prm.site_inlet
        units = 'ppm'
    elif gas == 'ch4':
        sites = prm.sites_ch4
        site_inlet = prm.site_inlet_ch4
        units = 'ppb'
    
    # Initialize dictionary for this gas
    all_gas_dict[gas] = {}
    
    # Set up file path and processing parameters
    path_for_files = prm.DATA_DIR + '/' + gas.upper() + '/'
    processing_gas = gas
    processing_meas_units = units
    
    try:
        # Collect raw data for all hours
        all_data = cfunc.collect_allsite_data(
            path_for_files, processing_gas,
            processing_meas_units, begin_year, end_year,
            time_frequency, sites, site_inlet
        )
        
        # Create raw dictionary for different time aggregations
        raw_dict = {}
        
        # Filter for target hours
        filtered_data = all_data[all_data['hour'].isin(target_hours)]
        raw_dict['H'] = filtered_data
        
        # Create aggregations for different time periods
        for time_period in ['D', 'W', 'MS']:
            try:
                all_data_renamed = cfunc.aggregate_allsite_data(
                    filtered_data, 
                    gas,
                    time_period, 
                    stat_type,
                    units=units  # Add units parameter here
                )
                raw_dict[time_period] = all_data_renamed
                print(f"Successfully aggregated {gas} for {time_period}")
            except ValueError as e:
                print(f"Error processing {gas} for {time_period}: {str(e)}")
                continue
        
        # Store in main dictionary
        all_gas_dict[gas] = raw_dict
        print(f"Successfully processed all aggregations for {gas}")
        
    except Exception as e:
        print(f"Error processing {gas}: {str(e)}")
        continue

print("\nProcessing complete!")

# Access data examples:
# CO2 daily data: all_gas_dict['co2']['D']
# CH4 monthly data: all_gas_dict['ch4']['MS']
# CO hourly data: all_gas_dict['co']['H']
    
#%% Collect Data for All Sites. This is Raw Data for All Hours we can do this by gas type

# Time Frequency is Hours
path_for_files=prm.DATA_DIR+'/'+gas.upper()+'/'
processing_gas = gas
processing_meas_units=units


all_data=cfunc.collect_allsite_data(path_for_files, processing_gas,
                         processing_meas_units, begin_year, end_year,
                         time_frequency, sites, site_inlet)

raw_dict = {}

# work with only target hours data

filtered_data = all_data[all_data['hour'].isin(target_hours)]

raw_dict['H']=filtered_data

# Aggregation is only for target hours
stat_type='mean'
all_data_renamed=cfunc.aggregate_allsite_data(filtered_data, processing_gas,
                           'D', stat_type,processing_meas_units)
raw_dict['D']=all_data_renamed

all_data_renamed=cfunc.aggregate_allsite_data(filtered_data, processing_gas,
                           'W', stat_type,processing_meas_units)
raw_dict['W']=all_data_renamed

all_data_renamed=cfunc.aggregate_allsite_data(filtered_data, processing_gas,
                           'MS', stat_type,processing_meas_units)

raw_dict['MS']= all_data_renamed

#%% Get Smoothed Time-Series for VIC and SCI. Apply all Filters

predicted_filtered, original_filtered, predicted_unfiltered, original_unfiltered=bg.return_background_smoothed(gas)


#%% Take Mean of Two Smoothed Time Series of VIC and SCI. This Would be Background

background=bg.combine_backgrounds(predicted_filtered['SCI'],predicted_filtered['VIC'], prm.begin_year,prm.end_year)
background.set_index('datetime_UTC', inplace=True)

#%% Deep copy of background data. This smooth time series would have columns containing background
# subtracted time-series

smooth_time_series = copy.deepcopy(background)

#%% Collect Data for All Sites. This is Raw Data for All Hours

# Time Frequency is Hours
path_for_files=prm.DATA_DIR+'/'+gas.upper()+'/'
processing_gas = gas
processing_meas_units=units

all_data=cfunc.collect_allsite_data(path_for_files, processing_gas,
                         processing_meas_units, begin_year, end_year,
                         time_frequency, sites, site_inlet)


#%% Store all raw data in dictionary at aggregated scales

all_data.set_index('datetime_UTC', inplace=True)

# Merge the selected column into the background DataFrame
all_data = pd.merge(
    all_data,
    background[['background']],  # Only the renamed column
    left_index=True,
    right_index=True,
    how='left'
)

col_name = prm.processing_gas+'_'+prm.processing_meas_units+'_background'
all_data.rename(columns={'background': col_name}, inplace=True)

all_data.reset_index(inplace=True)

#%% Compute Background Subtracted Time-Series
# Subtract background from raw data
smooth_time_series.reset_index(inplace=True)
# smooth_time_series must have column background. all_data contains raw data
non_smoothed=bg.subtract_smooth_background(all_data, smooth_time_series, prm.sites) # raw data - background

# set index to be datetime_UTC for matching background time and observation time

smooth_time_series.set_index('datetime_UTC', inplace=True) # This would have background subtracted time-series

non_smoothed.set_index('datetime_UTC', inplace=True)
#%% Remove sites which have been used as background sites
remove_sites = ['LJO'] # if we want to remove SCI and VIC then remove from here

filtered_sites = [x for x in prm.sites if x not in remove_sites]
#%% Merge background and background subtracted time series

for site in filtered_sites:
    try:
        # Generate the site-specific column name
        site_name = prm.processing_gas + '_' + prm.processing_meas_units + '_' + site

        # Merge the selected column into the background DataFrame
        smooth_time_series = pd.merge(
            smooth_time_series,
            non_smoothed[[site_name]],  # Only the renamed column
            left_index=True,
            right_index=True,
            how='left'
        )
        # Use this line for renaming col at runtime
        #smooth_time_series.rename(columns={site_name: site}, inplace=True)
        
    except Exception as e:
        # Log or print the error and skip to the next site
        print(f"Error processing site {site}: {e}")
        continue

#%% Store all bckground data in dictionary at aggregated scales
# Note we only aggregate for target hours
# Note std dev does not apply for background estimates. 

background_dict = {}

# work with only target hours data

filtered_smoothed_data = smooth_time_series[smooth_time_series['hour'].isin(target_hours)].copy()  # Create explicit copy
filtered_smoothed_data.reset_index(inplace=True)
filtered_smoothed_data.drop(columns=['numeric_time'], inplace=True)

background_dict['H']=filtered_smoothed_data

# Aggregation is only for target hours
stat_type='mean'
all_data_renamed=cfunc.aggregate_background_data(filtered_smoothed_data,'D', stat_type)


background_dict['D']=all_data_renamed

all_data_renamed=cfunc.aggregate_background_data(filtered_smoothed_data,'W', stat_type)
background_dict['W']=all_data_renamed

all_data_renamed=cfunc.aggregate_background_data(filtered_smoothed_data,'MS', stat_type)

background_dict['MS']= all_data_renamed

#%% Write to hdf file
if write_hdf == True:
    hdf_filename = 'aggregated_data.h5'
    
    cfunc.save_two_dicts_to_hdf(raw_dict, background_dict, hdf_filename)
    compressed_size = os.path.getsize(hdf_filename) / (1024 * 1024)  # Convert to MB
    print(f"Compressed HDF5 size: {compressed_size:.2f} MB")

if read_hdf == True:

    loaded_raw, loaded_background = cfunc.load_two_dicts_from_hdf(hdf_filename)

#%% Plotting Code

trace_gas_df = background_dict['W']
trace_gas_df.set_index('datetime_UTC', inplace=True)
                               
pfigure.plot_time_series_for_sites(trace_gas_df,filtered_sites,'concentration')

#%% Smooth Site Time - Series and then compute background
if site_smooth == True:
    # Assuming `prm.sites` is a list of site names
    for site in prm.sites:
        try:
            # Generate the site-specific column name
            site_name = prm.processing_gas + '_' + prm.processing_meas_units + '_' + site
    
            # Select the relevant columns for the site
            selected_columns = ['decimal_time', 'datetime_UTC', 'year','month','day','hour',site_name]
            data_site = all_data[selected_columns]
            data_for_smoothing = copy.deepcopy(data_site)
            data_for_smoothing = data_for_smoothing.dropna()
            
            # Filter rows in data_for_smoothing where 'hour' is in the target hours
            filtered_data = data_for_smoothing[data_for_smoothing['hour'].isin(target_hours)]
    
            # Compute statistics
            original_statistics, predicted_statistics, monthly_means, filtered_results = bg.ccgcrv(
                site, prm.processing_gas, prm.processing_meas_units, filtered_data
            )
    
            # Round datetime_UTC to the nearest hour
            original_statistics['datetime_UTC'] = pd.to_datetime(original_statistics['datetime_UTC']).dt.round('h')
            original_statistics.set_index('datetime_UTC', inplace=True)
    
            # Rename the smooth_value column to include site name
            # site_smoothed_co2 = f"{site}_{prm.processing_gas}"
            original_statistics.rename(columns={'smooth_value': site}, inplace=True)
    
            # Merge the selected column into the background DataFrame
            background = pd.merge(
                background,
                original_statistics[[site]],  # Only the renamed column
                left_index=True,
                right_index=True,
                how='left'
            )
            
            background[site]=background[site]-background['background']
            
            smooth_time_series = pd.merge(
                smooth_time_series,
                original_statistics[[site]],  # Only the renamed column
                left_index=True,
                right_index=True,
                how='left'
            )
        except Exception as e:
            # Log or print the error and skip to the next site
            print(f"Error processing site {site}: {e}")
            continue


#%%



raw_dict = {}

# work with only target hours data

filtered_data = all_data[all_data['hour'].isin(target_hours)]

raw_dict['H']=filtered_data

# Aggregation is only for target hours
stat_type='mean'
all_data_renamed=cfunc.aggregate_allsite_data(filtered_data, prm.processing_gas,
                           'D', stat_type)
raw_dict['D']=all_data_renamed

all_data_renamed=cfunc.aggregate_allsite_data(filtered_data, prm.processing_gas,
                           'W', stat_type)
raw_dict['W']=all_data_renamed

all_data_renamed=cfunc.aggregate_allsite_data(filtered_data, prm.processing_gas,
                           'MS', stat_type)

raw_dict['MS']= all_data_renamed

#%%


hour_type = 'all'

# Configuration parameters
if hour_type == 'afternoon':
    target_hours = [0, 1, 19, 20, 21, 22, 23]
elif hour_type == 'all':
    target_hours = list(np.arange(0, 24))  # Corrected variable name and used target_hours
else:  # Added an else for handling other potential hour_types
    target_hours = [] # Or some other default value if needed
    print(f"Warning: Unknown hour_type: {hour_type}. Using empty target_hours.")


print(target_hours) # Added print statement to check the result.

site_smooth = False
write_hdf = True
read_hdf = True
begin_year = prm.begin_year
end_year = prm.end_year
time_frequency = prm.time_frequency
stat_type = 'mean'

# Initialize dictionaries with gas types as top-level keys
raw_gas_dict = {
    'co2': {},
    'ch4': {},
    'co': {}
}
background_gas_dict = {
    'co2': {},
    'ch4': {},
    'co': {}
}

raw_ratio_gas_dict = {
    'co2_ch4': {},
    'co_co2': {},
    'co_ch4': {}
}

background_ratio_gas_dict = {
    'co2_ch4': {}  # Only CO2:CH4 for background
}

# Process each gas type
for gas in ['co2', 'ch4', 'co']:
    print(f"\nProcessing {gas}...")
    # Set gas-specific parameters
    if gas == 'co':
        sites = prm.sites_co
        site_inlet = prm.site_inlet_co
        units = 'ppb'
    elif gas == 'co2':
        sites = prm.sites
        site_inlet = prm.site_inlet
        units = 'ppm'
    elif gas == 'ch4':
        sites = prm.sites_ch4
        site_inlet = prm.site_inlet_ch4
        units = 'ppb'
    
    try:
        # Set up file path and processing parameters
        path_for_files = prm.DATA_DIR + '/' + gas.upper() + '/'
        processing_gas = gas
        processing_meas_units = units
        
        # Collect raw data for all gases
        all_data = cfunc.collect_allsite_data(
            path_for_files, processing_gas,
            processing_meas_units, begin_year, end_year,
            time_frequency, sites, site_inlet
        )
        
        # Process raw data for all gases
        raw_dict = {}
        filtered_data = all_data[all_data['hour'].isin(target_hours)]
        raw_dict['H'] = filtered_data
        
        # Create aggregations for raw data
        for time_period in ['D', 'W', 'MS']:
            try:
                all_data_renamed = cfunc.aggregate_allsite_data(
                    filtered_data, 
                    gas,
                    time_period, 
                    stat_type,
                    units=units
                )
                raw_dict[time_period] = all_data_renamed
                print(f"Successfully aggregated raw data for {gas} - {time_period}")
            except ValueError as e:
                print(f"Error processing raw data for {gas} - {time_period}: {str(e)}")
                continue
        
        # Store in raw dictionary
        raw_gas_dict[gas] = raw_dict
        
        # Process background data for CO2 and CH4 only
        if gas in ['co2', 'ch4']:
            # Get background data
           
            predicted_filtered, original_filtered, predicted_unfiltered, original_unfiltered = bg.return_background_smoothed(gas)
          
            # Combine backgrounds
            background = bg.combine_backgrounds(predicted_filtered['SCI'], predicted_filtered['VIC'], 
                                             prm.begin_year, prm.end_year)
            background.set_index('datetime_UTC', inplace=True)
            
            # Create deep copy for smooth time series
            smooth_time_series = copy.deepcopy(background)
            
            # Process with background
            all_data.set_index('datetime_UTC', inplace=True)
            all_data = pd.merge(
                all_data,
                background[['background']],
                left_index=True,
                right_index=True,
                how='left'
            )
            
            # Rename background column
            col_name = f"{gas}_{units}_background"
            all_data.rename(columns={'background': col_name}, inplace=True)
            all_data.reset_index(inplace=True)
            
            # Compute background subtracted time series
            smooth_time_series.reset_index(inplace=True)
            non_smoothed=bg.subtract_smooth_background(all_data, smooth_time_series, sites,processing_gas,processing_meas_units) # raw 
            
            # Set indices for matching
            smooth_time_series.set_index('datetime_UTC', inplace=True)
            non_smoothed.set_index('datetime_UTC', inplace=True)
            
            # Remove background sites
            remove_sites = ['LJO']
            filtered_sites = [x for x in sites if x not in remove_sites]
            
            # Merge background and background subtracted time series
            for site in filtered_sites:
                try:
                    site_name = f"{gas}_{units}_{site}"
                    smooth_time_series = pd.merge(
                        smooth_time_series,
                        non_smoothed[[site_name]],
                        left_index=True,
                        right_index=True,
                        how='left'
                    )
                except Exception as e:
                    print(f"Error processing site {site}: {e}")
                    continue
            
            # Store background data
            background_dict = {}
            filtered_smoothed_data = smooth_time_series[smooth_time_series['hour'].isin(target_hours)].copy()
            filtered_smoothed_data.reset_index(inplace=True)
            filtered_smoothed_data.drop(columns=['numeric_time'], inplace=True)
            background_dict['H'] = filtered_smoothed_data
            
            # Aggregate background data
            for time_period in ['D', 'W', 'MS']:
                try:
                    all_data_renamed = cfunc.aggregate_background_data(
                        filtered_smoothed_data,
                        time_period,
                        stat_type
                    )
                    background_dict[time_period] = all_data_renamed
                    print(f"Successfully aggregated background for {gas} - {time_period}")
                except Exception as e:
                    print(f"Error aggregating background for {gas} - {time_period}: {e}")
                    continue
            
            background_gas_dict[gas] = background_dict
        
        print(f"Successfully processed {gas}")
        
    except Exception as e:
        print(f"Error processing {gas}: {str(e)}")
        continue
    
# Process ratios
# Process raw ratios
for ratio_type in ['co2_ch4', 'co_co2', 'co_ch4']:
    print(f"\nProcessing raw {ratio_type} ratio...")
    ratio_dict = {}
    try:
        for time_period in ['H', 'D', 'W', 'MS']:
            # Get numerator and denominator gases
            if ratio_type == 'co2_ch4':
                num_gas = raw_gas_dict['co2'][time_period]
                den_gas = raw_gas_dict['ch4'][time_period]
                conversion = 1000  # Convert CO2 ppm to ppb
                sites = list(set(prm.sites) & set(prm.sites_ch4))
            elif ratio_type == 'co_co2':
                num_gas = raw_gas_dict['co'][time_period]
                den_gas = raw_gas_dict['co2'][time_period]
                conversion = 1000  # Convert CO2 ppm to ppb
                sites = list(set(prm.sites_co) & set(prm.sites))
            else:  # co_ch4
                num_gas = raw_gas_dict['co'][time_period]
                den_gas = raw_gas_dict['ch4'][time_period]
                conversion = 1  # Both already in ppb
                sites = list(set(prm.sites_co) & set(prm.sites_ch4))
            
            # Remove background sites
            sites = [x for x in sites if x not in ['LJO']]
            
            # Initialize ratio dataframe with datetime
            ratio_data = copy.deepcopy(num_gas[['datetime_UTC']])
            
            # Calculate ratio for each site
            for site in sites:
                if ratio_type == 'co2_ch4':
                    num_col = f'co2_ppm_{site}'
                    den_col = f'ch4_ppb_{site}'
                elif ratio_type == 'co_co2':
                    num_col = f'co_ppb_{site}'
                    den_col = f'co2_ppm_{site}'
                else:  # co_ch4
                    num_col = f'co_ppb_{site}'
                    den_col = f'ch4_ppb_{site}'
                
                ratio_col = f'{ratio_type}_ratio_{site}'
                ratio_data[ratio_col] = num_gas[num_col] * conversion / den_gas[den_col]
            
            ratio_dict[time_period] = ratio_data
            print(f"Successfully calculated raw {ratio_type} ratio for {time_period}")
            
        raw_ratio_gas_dict[ratio_type] = ratio_dict
        print(f"Successfully processed raw {ratio_type} ratios")
        
    except Exception as e:
        print(f"Error processing raw {ratio_type} ratios: {str(e)}")
        continue

# Process background ratios (CO2:CH4 only)
print("\nProcessing background CO2:CH4 ratio...")
try:
    ratio_dict = {}
    for time_period in ['H', 'D', 'W', 'MS']:
        # Get background data for both gases
        co2_bg = background_gas_dict['co2'][time_period]
        ch4_bg = background_gas_dict['ch4'][time_period]
        
        # Get common sites between CO2 and CH4 excluding background sites
        sites = list(set(prm.sites) & set(prm.sites_ch4))
        sites = [x for x in sites if x not in ['LJO']]
        
        # Initialize ratio dataframe with datetime
        ratio_data = copy.deepcopy(co2_bg[['datetime_UTC']])
        
        # Calculate background ratio for each site
        for site in sites:
            num_col = f'co2_ppm_{site}'
            den_col = f'ch4_ppb_{site}'
            ratio_col = f'co2_ch4_ratio_{site}'
            
            # Calculate ratio (converting CO2 from ppm to ppb)
            ratio_data[ratio_col] = co2_bg[num_col] * 1000 / ch4_bg[den_col]
        
        ratio_dict[time_period] = ratio_data
        print(f"Successfully calculated background CO2:CH4 ratio for {time_period}")
    
    background_ratio_gas_dict['co2_ch4'] = ratio_dict
    print("Successfully processed background CO2:CH4 ratios")
    
except Exception as e:
    print(f"Error processing background CO2:CH4 ratios: {str(e)}")

print("\nRatio processing complete!")

print("\nProcessing complete!")

# Final structure:
# raw_gas_dict
# ├── 'co2'
# │   ├── 'H': hourly_data
# │   ├── 'D': daily_data
# │   ├── 'W': weekly_data
# │   └── 'MS': monthly_data
# ├── 'ch4' 
# │   └── (same structure)
# └── 'co'
#     └── (same structure)
#
# background_gas_dict
# ├── 'co2'
# │   ├── 'H': hourly_data
# │   ├── 'D': daily_data
# │   ├── 'W': weekly_data
# │   └── 'MS': monthly_data
# ├── 'ch4'
# │   └── (same structure)
# └── 'co'
#     └── (empty dict)

#%% Write to hdf file
if write_hdf == True and hour_type == 'all':  # Corrected comparison and logical AND
    hdf_filename = 'aggregated_data_allhours.h5'
    # cfunc.save_two_dicts_to_hdf(raw_gas_dict, background_gas_dict, hdf_filename)
    cfunc.save_four_dicts_to_hdf(raw_gas_dict, background_gas_dict, raw_ratio_gas_dict, background_ratio_gas_dict, hdf_filename)
    compressed_size = os.path.getsize(hdf_filename) / (1024 * 1024)  # Convert to MB
    print(f"Compressed HDF5 size: {compressed_size:.2f} MB")
elif write_hdf == True and hour_type == 'afternoon': # Corrected comparison and logical AND
    hdf_filename = 'aggregated_data_afternoon.h5'
    cfunc.save_four_dicts_to_hdf(raw_gas_dict, background_gas_dict, raw_ratio_gas_dict, background_ratio_gas_dict, hdf_filename)
    compressed_size = os.path.getsize(hdf_filename) / (1024 * 1024) 
else: # Added an else to handle other cases
    print("No HDF5 file written or loaded based on current conditions.") # Or take other actions
# Loading

    #loaded_raw, loaded_background = cfunc.load_two_dicts_from_hdf(hdf_filename)
    

