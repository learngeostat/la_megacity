#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import geopandas as gpd # Make sure this import is at the top of the file
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import dash
import pandas as pd
import gcsfs
from utils import conc_func as cfunc 
from utils import constants as prm
from utils import fig_surface_obs as pfigure
import plotly.graph_objs as go
import numpy as np
import scipy as sc
from astropy.stats import median_absolute_deviation
import pyperclip
from dash.exceptions import PreventUpdate
import logging
import sys

# Configure logging to be visible in Cloud Run
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

GCS_AFTERNOON_DATA = "gs://la-megacity-dashboard-data-1/data/hdf_files/aggregated_data_afternoon.h5"
GCS_ALL_HOURS_DATA = "gs://la-megacity-dashboard-data-1/data/hdf_files/aggregated_data_allhours.h5"


GEO_DF = None
GEO_DF2 = None


# Global dictionaries for afternoon hours data
raw_data_dict_afternoon = None
background_data_dict_afternoon = None
raw_ratio_dict_afternoon = None
background_ratio_dict_afternoon = None

# Global dictionaries for all hours data
raw_data_dict_all = None
background_data_dict_all = None
raw_ratio_dict_all = None
background_ratio_dict_all = None

# Dictionary to store available sites for each gas type
available_sites = {}

def is_potentially_valid_date(date_str, time_agg):
    """
    Check if input could potentially be a valid date without full validation
    This allows user to continue typing without premature correction
    """
    import re
    
    # Basic patterns based on time aggregation
    if time_agg == 'H':
        # For hourly data, should look like "YYYY-MM-DD :: HH"
        # But we'll accept partial formats during typing
        return bool(re.match(r'^\d{4}(-\d{1,2}(-\d{1,2}( :: \d{1,2})?)?)?$', date_str))
    elif time_agg == 'MS':
        # For monthly data, should look like "YYYY-MM"
        # Accept "YYYY" or "YYYY-M" as partial input
        return bool(re.match(r'^\d{4}(-\d{1,2})?$', date_str))
    else:  # D or W
        # For daily/weekly data, should look like "YYYY-MM-DD"
        # Accept partial formats like "YYYY" or "YYYY-MM"
        return bool(re.match(r'^\d{4}(-\d{1,2}(-\d{1,2})?)?$', date_str))

# 4. Modify the validate_date_input function to be more lenient during user input

def validate_date_input(date_str, time_agg, available_dates):
    """
    Validate and correct date input based on time aggregation
    
    Parameters:
    -----------
    date_str : str
        The date string to validate
    time_agg : str
        Time aggregation ('H', 'D', 'W', or 'MS')
    available_dates : list
        List of available dates to use for bounds checking
    
    Returns:
    --------
    tuple
        (is_valid, corrected_date_str, date_obj)
        - is_valid: boolean indicating if original input was valid
        - corrected_date_str: corrected date string in proper format
        - date_obj: datetime object of corrected date
    """


    
    # Check for empty input
    if not date_str or date_str.strip() == '':
        # Return the first available date as default
        if available_dates is not None and len(available_dates) > 0:
            default_date = available_dates[0]
            format_str = get_format_string(time_agg)
            return False, default_date.strftime(format_str), default_date
        return False, "", None
    
    # Define expected formats based on time aggregation
    format_dict = {
        'H': ['%Y-%m-%d :: %H', '%Y-%m-%d %H', '%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M', '%Y-%m-%d'],
        'D': ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d-%m-%Y', '%Y.%m.%d'],
        'W': ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d-%m-%Y', '%Y.%m.%d'],
        'MS': ['%Y-%m', '%Y/%m', '%m/%Y', '%Y.%m', '%Y-%m-%d']
    }
    
    # Get available formats for this time aggregation
    formats = format_dict.get(time_agg, ['%Y-%m-%d'])
    
    # Try to parse the date with each format
    date_obj = None
    for fmt in formats:
        try:
            date_obj = pd.to_datetime(date_str, format=fmt)
            break
        except:
            continue
    
    # If we couldn't parse with any format, try pandas' flexible parser
    if date_obj is None:
        try:
            date_obj = pd.to_datetime(date_str)
        except:
            # If all parsing fails, return the first available date
            if available_dates is not None and len(available_dates) > 0:
                default_date = available_dates[0]
                format_str = get_format_string(time_agg)
                return False, default_date.strftime(format_str), default_date
            return False, "", None
    
    # Check if date is within available range
    if available_dates is not None and len(available_dates) > 0:
        min_date = pd.to_datetime(available_dates[0])
        max_date = pd.to_datetime(available_dates[-1])
        
        # If date is outside range, clamp to nearest valid date
        if date_obj < min_date:
            date_obj = min_date
        elif date_obj > max_date:
            date_obj = max_date
    
    # Format the corrected date
    format_str = get_format_string(time_agg)
    corrected_str = date_obj.strftime(format_str)
    
    # Check if original input was valid
    is_valid = corrected_str == date_str
    
    return is_valid, corrected_str, date_obj

def get_format_string(time_agg):
    """Return the appropriate date format string based on time aggregation"""
    if time_agg == 'H':
        return '%Y-%m-%d :: %H'
    elif time_agg == 'MS':
        return '%Y-%m'
    else:  # D or W
        return '%Y-%m-%d'
    
def add_date_validation_callbacks(app):
    @app.callback(
        [Output('surface-start-date-input', 'value', allow_duplicate=True),
         Output('surface-end-date-input', 'value', allow_duplicate=True),
         Output('surface-start-date-input', 'style'),
         Output('surface-end-date-input', 'style')],
        [Input('surface-start-date-input', 'value'),
         Input('surface-end-date-input', 'value')],
        [State('surface-gas-selector', 'value'),
         State('surface-time-agg', 'value'),
         State('surface-hour-type', 'value')],
        prevent_initial_call=True
    )
    def validate_date_inputs(start_date_str, end_date_str, selected_gas, time_agg, hour_type):
        """Validate and correct date inputs"""

        
        # Get callback context to determine which input triggered the callback
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Default styles
        valid_style = {'width': '140px'}
        invalid_style = {'width': '140px', 'border': '1px solid red'}
        
        start_style = valid_style
        end_style = valid_style
        
        # Get available dates for reference
        try:
            available_dates = get_available_dates(selected_gas, time_agg, hour_type)
            
            # Handle empty date lists
            if not available_dates or len(available_dates) == 0:
                raise PreventUpdate
        except Exception as e:
            print(f"Error getting available dates: {e}")
            raise PreventUpdate
        
        # Initialize corrected values with current values
        start_corrected = start_date_str
        end_corrected = end_date_str
        
        # Validate the input that triggered the callback
        if trigger_id == 'surface-start-date-input':
            start_valid, start_corrected, start_date_obj = validate_date_input(
                start_date_str, time_agg, available_dates
            )
            
            if not start_valid:
                start_style = invalid_style
            
            # If end date exists, make sure it's not before start date
            if end_date_str and start_date_obj:
                _, _, end_date_obj = validate_date_input(
                    end_date_str, time_agg, available_dates
                )
                
                if end_date_obj and end_date_obj < start_date_obj:
                    end_corrected = start_corrected
                    end_style = invalid_style
        
        elif trigger_id == 'surface-end-date-input':
            end_valid, end_corrected, end_date_obj = validate_date_input(
                end_date_str, time_agg, available_dates
            )
            
            if not end_valid:
                end_style = invalid_style
            
            # If start date exists, make sure it's not after end date
            if start_date_str and end_date_obj:
                _, _, start_date_obj = validate_date_input(
                    start_date_str, time_agg, available_dates
                )
                
                if start_date_obj and start_date_obj > end_date_obj:
                    start_corrected = end_corrected
                    start_style = invalid_style
        
        # Return the corrected values and styles
        return start_corrected, end_corrected, start_style, end_style

def extract_sites_from_columns(columns, gas_type='co2'):
    """
    Extract site names from column names for a specific gas type

    Parameters:
    -----------
    columns : list or Index
        List of column names
    gas_type : str
        Gas type ('co2', 'ch4', or 'co')
    
    Returns:
    --------
    list
        List of unique site names
    """
    sites = []
    # Define the unit based on gas type
    unit = 'ppm' if gas_type == 'co2' else 'ppb'

    # Create the column prefix to search for
    prefix = f"{gas_type}_{unit}_"

    for col in columns:
        if col.startswith(prefix) and not col.endswith('background'):
            # Split the column name and get the last part (site name)
            site = col.replace(prefix, '').split('_')[0]
            # Exclude SD (standard deviation) columns and ensure unique sites
            if not col.startswith(f"{gas_type}_SD") and site not in sites:
                sites.append(site)

    return sorted(sites)  # Sort sites alphabetically


def create_expansion_controls():
    """Create the expansion control column"""
    return html.Div([
        html.Div([
            html.Button("→", id="expand-left", className="expand-button"),
            html.Button("←", id="expand-right", className="expand-button")
        ], className="expand-control")
    ], className="px-0", style={'width': '30px'})

def create_restore_button():
    """Create the restore button"""
    return html.Button(
        "Restore Panels", 
        id="restore-button", 
        className="restore-button",
        style={'display': 'none'}
    )
def create_stats_modal():
    """Create the statistics modal"""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Essential Statistics")),
            dbc.ModalBody(id="stats-modal-body", children=[
                # Table will be populated by callback
                dbc.Table(
                    id="stats-table",
                    bordered=True,
                    hover=True,
                    responsive=True,
                    striped=True,
                )
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="stats-close", className="ms-auto")
            ),
        ],
        id="stats-modal",
        size="lg",
        is_open=False,
    )


# In surface_observations.py

def load_data_from_gcs_hdf(gcs_path):
    """
    Reads a pandas HDFStore file from GCS and reconstructs the nested dictionaries.
    """
    import tempfile
    import os
    
    logging.info(f"Attempting to read HDFStore from {gcs_path}")
    fs = gcsfs.GCSFileSystem()
    
    # Initialize the main dictionaries
    raw_data = {}
    background_data = {}
    raw_ratio_data = {}
    background_ratio_data = {}

    all_dicts = {
        'raw': raw_data,
        'background': background_data,
        'raw_ratio': raw_ratio_data,
        'background_ratio': background_ratio_data
    }

    try:
        # Create a temporary file to download the HDF5 data
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            try:
                # Download the file from GCS to temporary local file
                fs.get(gcs_path, temp_file.name)
                
                # Now use pandas HDFStore with the local temporary file
                with pd.HDFStore(temp_file.name, 'r') as store:
                    # Iterate through all the keys (paths) in the HDF5 file
                    for key in store.keys():
                        # The key looks like '/raw/co2/H'
                        parts = key.strip('/').split('/')
                        if len(parts) != 3:
                            continue # Skip any keys that aren't in the expected format

                        dict_name, gas_name, agg_name = parts

                        # Check if it's one of the top-level dicts we care about
                        if dict_name in all_dicts:
                            # If the gas key doesn't exist yet, create it
                            if gas_name not in all_dicts[dict_name]:
                                all_dicts[dict_name][gas_name] = {}
                            
                            # Read the DataFrame from the store and add it
                            df = store.get(key)
                            all_dicts[dict_name][gas_name][agg_name] = df
                            logging.info(f"Successfully loaded DataFrame for key: {key}")
                            
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass  # File might already be deleted

    except Exception:
        logging.exception(f"Failed to read HDFStore from {gcs_path}")
        # Return empty dicts on failure
        return {}, {}, {}, {}

    return raw_data, background_data, raw_ratio_data, background_ratio_data



# In surface_observations.py

def init():
    """Initialize by loading pre-aggregated data and shapefiles."""
    global raw_data_dict_afternoon, background_data_dict_afternoon
    global raw_data_dict_all, background_data_dict_all
    global available_sites
    global GEO_DF, GEO_DF2  # Add the new globals here

    try:
        logging.info("--- Initializing Surface Observations Page ---")

        # Load shapefiles for the map
        try:
            logging.info("Loading shapefiles from GCS...")
            shapefile_path1 = prm.SHAPEFILES['socabbound']
            shapefile_path2 = prm.SHAPEFILES['paper_towers']
            GEO_DF = gpd.read_file(shapefile_path1)
            GEO_DF2 = gpd.read_file(shapefile_path2)
            logging.info("Successfully loaded shapefiles.")
        except Exception as e:
            logging.error(f"Failed to load shapefiles: {e}")
            # Continue without map data if necessary, or handle as a critical error
            GEO_DF = None
            GEO_DF2 = None

        # Load afternoon hours data using the NEW function
        hdf_filename_afternoon = prm.DATA_FILES['aggregated_data_afternoon']
        raw_data_dict_afternoon, background_data_dict_afternoon, _, _ = \
            load_data_from_gcs_hdf(hdf_filename_afternoon) # <-- USE NEW FUNCTION
        logging.info("Finished loading afternoon hours data.")

        # Load all hours data using the NEW function
        hdf_filename_all = prm.DATA_FILES['aggregated_data_allhours.h5']
        raw_data_dict_all, background_data_dict_all, _, _ = \
            load_data_from_gcs_hdf(hdf_filename_all) # <-- USE NEW FUNCTION
        logging.info("Finished loading all hours data.")
        
        # Populate available sites for each gas type
        available_sites = {}
        for gas in ['co2', 'ch4', 'co']:
            sites_afternoon = set()
            sites_all = set()
            if gas in raw_data_dict_afternoon:
                sites_afternoon = set(extract_sites_from_columns(raw_data_dict_afternoon[gas]['H'].columns, gas))
            if gas in raw_data_dict_all:
                sites_all = set(extract_sites_from_columns(raw_data_dict_all[gas]['H'].columns, gas))
            available_sites[gas] = sorted(sites_afternoon.union(sites_all))
            logging.info(f"Available sites for {gas}: {available_sites[gas]}")

        logging.info("--- Initialization Complete ---")
        return available_sites.get('co2', [])

    except Exception as e:
        logging.exception("A critical error occurred during initialization.")
        # Initialize empty dictionaries in case of error
        raw_data_dict_afternoon, background_data_dict_afternoon = {}, {}
        raw_data_dict_all, background_data_dict_all = {}, {}
        available_sites = {'co2': [], 'ch4': [], 'co': []}
        return []

def essential_stats(data):
    """
    Calculate essential statistics that are reliable even for small datasets.
    Includes basic distributional characteristics and robust measures.
    
    Args:
        data (numpy.ndarray): A one-dimensional array of numerical data.
    
    Returns:
        pandas.DataFrame: DataFrame containing essential statistics.
    
    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input array is empty or contains non-numeric values.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if len(data) == 0 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input array must be non-empty and contain only numeric values.")

    # Calculate quantiles once and reuse
    quantiles = np.percentile(data, [0, 10, 25, 50, 75, 90, 100])
    minimum, first_decile, first_quartile, median, third_quartile, last_decile, maximum = \
        [np.round(x, 3) for x in quantiles]
    
    # Basic statistics
    count = int(len(data))
    mean = np.round(np.nanmean(data), 3)
    
    # Spread measures
    std = np.round(np.std(data), 3)
    iqr = np.round(third_quartile - first_quartile, 3)
    mad = np.round(median_absolute_deviation(data), 3)
    
    # Shape measures
    skewness = np.round(sc.stats.skew(data), 3)
    kurtosis = np.round(sc.stats.kurtosis(data, fisher=False), 3)
    
    # Compile statistics
    stat_names = [
        'count',
        'minimum', 'maximum',
        'first_decile', 'last_decile',
        'first_quartile', 'third_quartile',
        'mean', 'median',
        'std', 'iqr', 'mad',
        'skewness', 'kurtosis'
    ]
    
    stat_values = [
        count,
        minimum, maximum,
        first_decile, last_decile,
        first_quartile, third_quartile,
        mean, median,
        std, iqr, mad,
        skewness, kurtosis
    ]
    
    # Create DataFrame
    statistics = pd.DataFrame({'STAT_NAME': stat_names, 'ESTIMATE': stat_values})
    return statistics


def get_site_options(gas_type='co2', site_dict=None):
    """
    Generate dropdown options for site selection based on gas type.

    Parameters:
    -----------
    gas_type : str
        Gas type ('co2', 'ch4', or 'co')
    site_dict : dict
        Dictionary mapping site codes to names
    
    Returns:
    --------
    list
        List of dictionaries with label and value for dropdown options
    """
    if not isinstance(gas_type, str):
        print(f"Warning: Unexpected gas_type format: {type(gas_type)}. Using 'co2' as default.")
        gas_type = 'co2'
    
    if site_dict is None:
        site_dict = {}

    # Get sites for the specified gas type, defaulting to empty list if not found
    sites = available_sites.get(gas_type, [])

    # Create dropdown options, excluding LJO
    options = []
    for site in sites:
        if site != 'LJO':
            site_name = site_dict.get(site, 'Unknown')
            label = f"{site} ({site_name})" if site_name != 'Unknown' else site
            options.append({'label': label, 'value': site})

    return sorted(options, key=lambda x: x['value'])  # Sort by site code

def create_header_card_surface():
    """Create the header card with text followed by icons"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className="fas fa-industry me-2"),
                            "Surface CO₂ Network",
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
                        # Modified Documentation Button to act as a link
                        html.A(
                            dbc.Button([
                                "Documentation ",
                                html.I(className="fas fa-file-pdf")
                            ], color="secondary", className="me-2"),
                            href="/assets/surface_observations.pdf",
                            target="_blank"  # This opens the PDF in a new browser tab
                        ),
                        
                        # Unchanged Help Button
                        dbc.Button([
                            "Help ",
                            html.I(className="fas fa-question-circle")
                        ], color="secondary", className="me-2", id="surface-help-button"),
                        
                        # Unchanged Reset Button
                        dbc.Button([
                            "Reset ",
                            html.I(className="fas fa-redo")
                        ], color="secondary", id="surface-restart-button")
                
                    ], className="d-flex justify-content-end")
                ], width=2)
            ], className="align-items-center")
        ]),
        dbc.Collapse(
            dbc.CardBody([
                html.P([
                    "Analyze CO₂ concentrations across the Los Angeles region. ",
                    "Explore spatial distribution and temporal patterns."
                ], className="text-muted mb-0")
            ]),
            id="header-collapse",
            is_open=True,
        )
    ], className="mb-4 shadow-sm")

def create_help_modal_surface():
    """Create the help documentation modal"""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Surface CO₂ Network Help")),
            dbc.ModalBody([
                html.H5("Overview"),
                html.P("This dashboard visualizes surface CO₂ measurements across the Los Angeles region, allowing you to explore both spatial distributions and temporal patterns."),
            
                html.H5("Analysis Controls"),
                html.H6("Basic Controls:"),
                html.Ul([
                    html.Li([
                        html.Strong("Select Sites: "),
                        "Choose one or more monitoring sites to analyze."
                    ]),
                    html.Li([
                        html.Strong("Time Series Type: "),
                        "Toggle between 'Raw Data' and 'Background Subtracted' views."
                    ]),
                    html.Li([
                        html.Strong("Time Aggregation: "),
                        "Select how data should be aggregated temporally (hourly to monthly)."
                    ]),
                    html.Li([
                        html.Strong("Analysis Metric: "),
                        "Choose between mean CO₂ values or standard deviation for analysis."
                    ])
                ]),
            
                html.H5("Map Visualization"),
                html.Ul([
                    html.Li([
                        html.Strong("Navigation: "),
                        "Pan by dragging, zoom with mouse wheel or touchpad gestures. Double-click to reset view."
                    ]),
                    html.Li([
                        html.Strong("Site Selection: "),
                        "Click on sites to see their temporal patterns in the time series plot."
                    ])
                ]),
            
                html.H5("Time Series Plot"),
                html.Ul([
                    html.Li([
                        html.Strong("Multi-site Comparison: "),
                        "Compare CO₂ patterns across multiple monitoring sites."
                    ]),
                    html.Li([
                        html.Strong("Interaction: "),
                        "Hover over points for detailed values, zoom by selecting regions, double-click to reset view."
                    ])
                ]),
            
                html.H5("Additional Features"),
                html.Ul([
                    html.Li([
                        html.Strong("Panel Expansion: "),
                        "Use arrows between panels to expand either the map or time series view."
                    ]),
                    html.Li([
                        html.Strong("Data Download: "),
                        "Download data for both spatial and temporal visualizations."
                    ]),
                    html.Li([
                        html.Strong("Reset: "),
                        "Use the reset button to restore all controls to their default values."
                    ])
                ])
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="help-close", className="ms-auto")
            ),
        ],
        id="help-modal_surface",
        size="lg",
        is_open=False,
    )


def get_control_panel(site_options):
    """Return the control panel layout with gas type selection and dynamic controls."""
    return dbc.Card([
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
                # First row - Main controls
                dbc.Row([
                    # Hour Type Selection
                    dbc.Col([
                        html.Label("Hour Type", className="form-label fw-bold"),
                        dcc.Dropdown(
                            id='surface-hour-type',
                            options=[
                                {'label': 'Hours: UTC 11:00-17:00', 'value': 'afternoon'},
                                {'label': 'All Hours', 'value': 'all'}
                            ],
                            value='afternoon',
                            clearable=False,
                            className="mb-2"
                        )
                    ], md=2),
                    
                    # Gas Type Selection
                    dbc.Col([
                        html.Label("Gas", className="form-label fw-bold"),
                        dcc.Dropdown(
                            id='surface-gas-selector',
                            options=[
                                {'label': 'CO₂', 'value': 'co2'},
                                {'label': 'CH₄', 'value': 'ch4'},
                                {'label': 'CO', 'value': 'co'}
                            ],
                            value='co2',
                            clearable=False,
                            className="mb-2"
                        )
                    ], md=1),
                
                    # Site Selection
                    dbc.Col([
                        html.Label("Select Sites", className="form-label fw-bold"),
                        dcc.Dropdown(
                            id='surface-site-selector',
                            options=site_options,
                            value=['GRA'],
                            multi=True,
                            clearable=True,
                            className="mb-2"
                        )
                    ], md=3),
                
                    # Time Series Type
                    dbc.Col([
                        html.Label("Time Series Type", className="form-label fw-bold"),
                        dcc.Dropdown(
                            id='surface-timeseries-type',
                            options=[
                                {'label': 'Raw Data', 'value': 'raw'},
                                {'label': 'Background Subtracted', 'value': 'background'}
                            ],
                            value='raw',
                            clearable=False,
                            className="mb-2"
                        )
                    ], md=2),
                
                    # Time Aggregation
                    dbc.Col([
                        html.Label("Time Aggregation", className="form-label fw-bold"),
                        dcc.Dropdown(
                            id='surface-time-agg',
                            options=[
                                {'label': 'Hourly', 'value': 'H'},
                                {'label': 'Daily', 'value': 'D'},
                                {'label': 'Weekly', 'value': 'W'},
                                {'label': 'Monthly', 'value': 'MS'}
                            ],
                            value='MS',
                            clearable=False,
                            className="mb-2"
                        )
                    ], md=2),
                
                    # Analysis Metric
                    dbc.Col([
                        html.Label("Analysis Metric", className="form-label fw-bold"),
                        dcc.Dropdown(
                            id='surface-analysis-type',
                            options=[
                                {'label': 'Mean', 'value': 'mean'},
                                {'label': 'Standard Deviation', 'value': 'std'}
                            ],
                            value='mean',
                            clearable=False,
                            className="mb-2"
                        )
                    ], md=2)
                ]),          
                # Second row - Time Slider and Y-axis/X-axis controls
                dbc.Row([
                    # Time Slider Column
                    dbc.Col([
                        html.Div([
                                html.Label("X-Axis", className="form-label mb-0 text-center", 
                                style={'fontSize': '0.9rem', 'fontWeight': 'bold', 'width': '100%', 'marginBottom': '5px'}),
                                ]),
                        dcc.RangeSlider(
                            id='surface-time-slider',
                            min=0,
                            max=100,
                            step=1,
                            value=[0, 100],
                            marks={},
                            className="mt-3"
                        ),
                    ], width=8),
                    
                    # Y-axis and X-axis Time Controls Column
                    dbc.Col([
                        # Row 1: Y-Axis Controls (Auto, Min, Max)
                        dbc.Row([
                            dbc.Col([
                                html.Label("Y-Axis", className="form-label fw-bold small"),
                                dbc.Select(
                                    id='y-axis-scale-type',
                                    options=[
                                        {'label': 'Auto', 'value': 'auto'},
                                        {'label': 'Fixed', 'value': 'fixed'}
                                    ],
                                    value='auto', size="sm"
                                )
                            ], md=4),
                            dbc.Col([
                                html.Label("Min", className="form-label fw-bold small"),
                                dbc.Input(
                                    id='y-axis-min', type='number', placeholder='Min',
                                    size="sm", disabled=True, debounce=True
                                )
                            ], md=4),
                            dbc.Col([
                                html.Label("Max", className="form-label fw-bold small"),
                                dbc.Input(
                                    id='y-axis-max', type='number', placeholder='Max',
                                    size="sm", disabled=True, debounce=True
                                )
                            ], md=4)
                        ], align="end", className="mb-2"),

                        # Row 2: X-Axis Time Inputs (Start, End)
                        dbc.Row([
                            dbc.Col([
                                html.Label("X-Axis: Start Time", className="form-label fw-bold small"),
                                dbc.Input(
                                    id='surface-start-date-input', type='text',
                                    placeholder='Start Date', size="sm",
                                    debounce=True, n_submit=0
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("X-Axis: End Time", className="form-label fw-bold small"),
                                dbc.Input(
                                    id='surface-end-date-input', type='text',
                                    placeholder='End Date', size="sm",
                                    debounce=True, n_submit=0
                                )
                            ], md=6)
                        ]),
                        
                        # Hidden div for period text
                        html.Div(id='surface-period-text', className="d-none")

                    ], width=4),
                ], className="mt-3")
            ]),
            id="controls-collapse",
            is_open=True,
        )
    ], className="mb-4 shadow-sm")

def get_main_content():
    """Return the main content layout with collapsible sections and expansion controls."""
    return html.Div([
        # Left Panel (Map)
        html.Div([
            dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                [
                                    html.I(className="fas fa-map-marked-alt me-2"),
                                    "Surface Sites",
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
                                                            id="surface-spatial-download-csv"),
                                        dbc.DropdownMenuItem(divider=True),
                                        dbc.DropdownMenuItem("Copy for AI Analysis",
                                                            id="surface-spatial-copy-ai"),
                                    ],
                                    label=[
                                        html.I(className="fas fa-download me-1"),
                                        "DATA"
                                    ],
                                    color="light",
                                    size="sm"
                                )
                            ])
                        ], width=2)
                    ], className="align-items-center")
                ]),
                dbc.Collapse(
                    dbc.CardBody([
                        dcc.Graph(
                            id='surface-map',
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
                )
            ], className="shadow-sm")
        ], id="left-panel", className="panel-transition flex-grow-1", style={'flex': '1', 'minWidth': '0'}),

        # Expansion Controls
        create_expansion_controls(),

        # Right Panel (Time Series)
        html.Div([
            dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                [
                                    html.I(className="fas fa-clock me-2"),
                                    "Temporal Patterns",
                                    html.I(className="fas fa-chevron-down ms-2", id="timeseries-chevron"),
                                ],
                                color="link",
                                id="timeseries-toggle",
                                className="text-primary fs-4 text-decoration-none p-0 w-100 text-start",
                                style={"box-shadow": "none"}
                            )
                        ], width=10),
                        dbc.Col([
                            html.Div([
                                dbc.Button(
                                    [
                                        html.I(className="fas fa-chart-line"),
                                        html.Span("STATS", className="ms-2")
                                    ],
                                    id="stats-button",
                                    color="light",
                                    size="sm",
                                    className="me-2 d-inline-flex align-items-center"
                                ),
                                dbc.ButtonGroup([
                                    dbc.DropdownMenu(
                                        [
                                            dbc.DropdownMenuItem("Current View (CSV)",
                                                                id="surface-temporal-download-csv"),
                                            dbc.DropdownMenuItem(divider=True),
                                            dbc.DropdownMenuItem("Copy for AI Analysis",
                                                                id="surface-temporal-copy-ai"),
                                        ],
                                        label=[
                                            html.I(className="fas fa-download me-1"),
                                            "DATA"
                                        ],
                                        color="light",
                                        size="sm"
                                    )
                                ])
                            ], className="d-flex justify-content-end align-items-center")
                        ], width=2)
                    ], className="align-items-center")
                ]),
                dbc.Collapse(
                    dbc.CardBody([
                        dcc.Graph(
                            id='surface-timeseries',
                            style={'height': '65vh'}
                        )
                    ]),
                    id="timeseries-collapse",
                    is_open=True,
                )
            ], className="shadow-sm")
        ], id="right-panel", className="panel-transition flex-grow-1", style={'flex': '1', 'minWidth': '0'})
    ], className="d-flex gap-2", style={'height': '65vh'})



def get_layout():
    """Return the complete page layout."""
    global available_sites
    # Initialize site options with default gas type
    site_options = get_site_options(gas_type='co2', site_dict=prm.site_dict)
    return dbc.Container([
        # Stores and download components
        dcc.Store(id='surface-selected-sites-store'),
        dcc.Store(id='surface-map-state-store'),
        dcc.Store(id='expansion-state', data='none'),
        dcc.Download(id="download-dataframe-csv"),  # Added download component
        
        # Layout components
        create_header_card_surface(),
        get_control_panel(site_options),
        get_main_content(),
        create_restore_button(),
        create_help_modal_surface(),
        create_stats_modal()  # Add the stats modal
    ], fluid=True, className="px-4 py-3")


def get_available_dates(selected_gas, time_agg, hour_type):
    """Get list of available dates from the data based on hour type selection"""
    try:
        print(f"Getting available dates for {hour_type} hours")
        
        # Select appropriate dictionary based on hour_type
        if hour_type == 'afternoon':
            data_dict = raw_data_dict_afternoon
        else:  # all hours
            data_dict = raw_data_dict_all
            
        if data_dict is None or not data_dict:
            raise ValueError(f"Data dictionary for {hour_type} hours is not properly initialized")
            
        if selected_gas not in data_dict:
            raise ValueError(f"Selected gas {selected_gas} not found in {hour_type} hours data")
            
        data = data_dict[selected_gas][time_agg]
        dates = pd.to_datetime(data['datetime_UTC']).sort_values().unique()
        
        print(f"Found {len(dates)} available dates")
        return dates
        
    except Exception as e:
        print(f"Error getting available dates for {hour_type} hours: {e}")
        return []

def get_data_for_visualization(selected_gas, selected_sites, time_agg, analysis_type, 
                             timeseries_type, slider_value, hour_type):
    """Get data from the pre-loaded dictionaries based on user selection and gas type"""
    try:
        print(f"\nDetailed data retrieval debug:")
        print(f"Parameters received:")
        print(f"  selected_gas: {selected_gas}")
        print(f"  selected_sites: {selected_sites}")
        print(f"  time_agg: {time_agg}")
        print(f"  analysis_type: {analysis_type}")
        print(f"  timeseries_type: {timeseries_type}")
        print(f"  slider_value: {slider_value}")
        print(f"  hour_type: {hour_type}")
        
        # Select appropriate dictionaries based on hour_type
        if hour_type == 'afternoon':
            raw_dict = raw_data_dict_afternoon
            background_dict = background_data_dict_afternoon
            print("\nUsing afternoon hours data")
        else:  # all hours
            raw_dict = raw_data_dict_all
            background_dict = background_data_dict_all
            print("\nUsing all hours data")

        # Verify dictionaries are properly loaded
        print(f"\nRaw dictionary state:")
        print(f"  Type: {type(raw_dict)}")
        print(f"  Available gases: {list(raw_dict.keys())}")
        print(f"  Available aggregations for {selected_gas}: {list(raw_dict[selected_gas].keys())}")
        
        # Get the data based on time series type
        if timeseries_type == 'raw':
            data = raw_dict[selected_gas][time_agg]
        else:  # background subtracted
            if selected_gas == 'co':
                raise ValueError("Background subtracted data not available for CO")
            data = background_dict[selected_gas][time_agg]
        
        print(f"\nRetrieved data:")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {data.columns.tolist()}")
        
        # CRITICAL FIX: Make a copy and fix datetime_UTC column
        data_copy = data.copy()
        
        # Convert datetime_UTC to proper datetime if it's not already
        if 'datetime_UTC' in data_copy.columns:
            print(f"  datetime_UTC dtype before conversion: {data_copy['datetime_UTC'].dtype}")
            print(f"  Sample datetime_UTC values: {data_copy['datetime_UTC'].head()}")
            print(f"  Type of first datetime value: {type(data_copy['datetime_UTC'].iloc[0])}")
            
            # Handle different possible formats from HDF5 loading
            if data_copy['datetime_UTC'].dtype == 'object':
                # If it's object type, it might be numpy arrays or mixed types
                # Convert each element to datetime
                datetime_series = []
                for i, val in enumerate(data_copy['datetime_UTC']):
                    try:
                        if isinstance(val, np.ndarray):
                            # If it's a numpy array, take the first element
                            val = val[0] if len(val) > 0 else val
                        # Convert to pandas datetime
                        datetime_series.append(pd.to_datetime(val))
                    except Exception as e:
                        print(f"  Error converting datetime at index {i}: {e}")
                        # Use a default date or skip
                        datetime_series.append(pd.NaT)
                
                data_copy['datetime_UTC'] = pd.Series(datetime_series, index=data_copy.index)
            else:
                # If it's already a datetime-like type, just ensure it's proper pandas datetime
                data_copy['datetime_UTC'] = pd.to_datetime(data_copy['datetime_UTC'])
            
            print(f"  datetime_UTC dtype after conversion: {data_copy['datetime_UTC'].dtype}")
            print(f"  Sample converted values: {data_copy['datetime_UTC'].head()}")
        
        # Get available dates for filtering
        available_dates = pd.to_datetime(data_copy['datetime_UTC']).sort_values().unique()
        print(f"\nDate range:")
        print(f"  Start: {available_dates[0]}")
        print(f"  End: {available_dates[-1]}")
        print(f"  Total dates: {len(available_dates)}")
        
        # Ensure slider values are within bounds
        max_index = len(available_dates) - 1
        start_idx = min(max(0, slider_value[0]), max_index)
        end_idx = min(max(0, slider_value[1]), max_index)
        
        # Get start and end dates from slider value
        start_date = available_dates[start_idx]
        end_date = available_dates[end_idx]
        print(f"\nSelected date range:")
        print(f"  Start: {start_date}")
        print(f"  End: {end_date}")
        
        # Base columns we always want
        base_columns = ['datetime_UTC']
        
        # Get columns based on analysis type and gas type
        unit = 'ppm' if selected_gas == 'co2' else 'ppb'
        if analysis_type == 'mean':
            site_columns = [f'{selected_gas}_{unit}_{site}' for site in selected_sites]
        else:  # std
            site_columns = [f'{selected_gas}_SD_minutes_{site}' for site in selected_sites]
            
        print(f"\nRequired columns:")
        print(f"  Base columns: {base_columns}")
        print(f"  Site columns: {site_columns}")
        
        # Check if all required columns exist
        missing_columns = [col for col in site_columns if col not in data_copy.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Combine base columns with site-specific columns
        columns_to_keep = base_columns + site_columns
        
        # Filter data for required columns
        filtered_data = data_copy[columns_to_keep].copy()
        
        # Ensure both datetime columns are timezone-naive for comparison
        start_date = pd.to_datetime(start_date).tz_localize(None) if hasattr(start_date, 'tz') and start_date.tz else start_date
        end_date = pd.to_datetime(end_date).tz_localize(None) if hasattr(end_date, 'tz') and end_date.tz else end_date
        
        # Ensure filtered_data datetime column is also timezone-naive
        if hasattr(filtered_data['datetime_UTC'].iloc[0], 'tz') and filtered_data['datetime_UTC'].iloc[0].tz:
            filtered_data['datetime_UTC'] = filtered_data['datetime_UTC'].dt.tz_localize(None)
        
        print(f"\nApplying date filter:")
        print(f"  Start date type: {type(start_date)}")
        print(f"  End date type: {type(end_date)}")
        print(f"  DataFrame datetime type: {type(filtered_data['datetime_UTC'].iloc[0])}")
        
        # Apply time filter using slider values
        mask = (filtered_data['datetime_UTC'] >= start_date) & \
               (filtered_data['datetime_UTC'] <= end_date)
        filtered_data = filtered_data[mask]
        
        print(f"\nFiltered data:")
        print(f"  Final shape: {filtered_data.shape}")
        print(f"  Date range: {filtered_data['datetime_UTC'].min()} to {filtered_data['datetime_UTC'].max()}")
        
        if filtered_data.empty:
            raise ValueError("No data found for the selected parameters and date range")
            
        return filtered_data
        
    except Exception as e:
        print(f"\nERROR in get_data_for_visualization:")
        print(f"  {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return None


def update_analysis_type_availability(timeseries_type):
    """Disable analysis type selection for background subtracted data

    Args:
        timeseries_type (str): Type of timeseries selected
    
    Returns:
        tuple: (disabled status, value)
    """
    if timeseries_type == 'background':
        return True, 'mean'  # Disable dropdown and force to 'mean'
    return False, dash.no_update  # Enable dropdown and keep current value

def create_error_figure(error_message):
    """Create an error figure to display when data loading fails

    Args:
        error_message (str): Error message to display
    
    Returns:
        go.Figure: Error figure
    """
    return go.Figure(layout=dict(
        title=dict(text='Error Loading Data'),
        annotations=[dict(
            text=str(error_message),
            xref='paper',
            yref='paper',
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color='red')
        )],
        paper_bgcolor='white',
        plot_bgcolor='white'
    ))

def generate_timeseries_title(selected_gas, analysis_type, timeseries_type):
    """Generate the title for the timeseries plot based on gas type"""
    gas_display = {
        'co2': 'CO₂',
        'ch4': 'CH₄',
        'co': 'CO'
    }

    if timeseries_type == 'raw':
        analysis_text = 'Mean' if analysis_type == 'mean' else 'Standard Deviation'
        return f"{analysis_text} {gas_display[selected_gas]} Concentrations"
    return f"Background Subtracted {gas_display[selected_gas]} Concentrations"

def update_surface_figures(selected_gas, selected_sites, analysis_type, time_agg, 
                         timeseries_type, relayoutData, map_state, slider_value, hour_type):
    """Update both map and time series figures based on gas selection"""
    try:
        if not selected_sites:
            selected_sites = ['GRA']
            
        print("\nUpdating surface figures with parameters:")
        print(f"  Gas: {selected_gas}")
        print(f"  Sites: {selected_sites}")
        print(f"  Analysis: {analysis_type}")
        print(f"  Time aggregation: {time_agg}")
        print(f"  Type: {timeseries_type}")
        print(f"  Hour type: {hour_type}")

        # Create map figure with state persistence
        map_fig, updated_map_state = pfigure.site_ioami_map(
            full_geometry=GEO_DF,   # <--- CORRECTED
            half_geometry=GEO_DF2,  # <--- CORRECTED
            selected_sites=selected_sites,
            relayoutData=relayoutData,
            map_state=map_state
        )

        # Get data based on selections
        aggregated_results = get_data_for_visualization(
            selected_gas,
            selected_sites,
            time_agg,
            analysis_type,
            timeseries_type,
            slider_value,
            hour_type
        )

        if aggregated_results is None:
            raise ValueError("Failed to retrieve data for visualization")
        
        if aggregated_results.empty:
            raise ValueError("No data available for the selected parameters")

        # Get appropriate units for the selected gas
        units = 'ppm' if selected_gas == 'co2' else 'ppb'

        # Create time series figure
        title = generate_timeseries_title(
            selected_gas,
            analysis_type, 
            timeseries_type
        )

        time_series_fig = pfigure.site_time_series(
            aggregated_results,
            selected_sites,
            title,
            units,
            analysis_type,
            selected_gas
        )

        return map_fig, time_series_fig, updated_map_state
        
    except Exception as e:
        print(f"\nERROR in update_surface_figures:")
        print(f"  {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

        error_fig = create_error_figure(str(e))
        return error_fig, error_fig, None


def register_callbacks(app):
    
    
    @app.callback(
    [Output('surface-start-date-input', 'value', allow_duplicate=True),
     Output('surface-end-date-input', 'value', allow_duplicate=True),
     Output('surface-start-date-input', 'style'),
     Output('surface-end-date-input', 'style')],
    [Input('surface-start-date-input', 'n_blur'),  # Changed from 'value' to 'n_blur'
     Input('surface-end-date-input', 'n_blur'),    # Changed from 'value' to 'n_blur'
     Input('surface-start-date-input', 'n_submit'),
     Input('surface-end-date-input', 'n_submit')],
    [State('surface-start-date-input', 'value'),   # Added as State
     State('surface-end-date-input', 'value'),     # Added as State
     State('surface-gas-selector', 'value'),
     State('surface-time-agg', 'value'),
     State('surface-hour-type', 'value')],
    prevent_initial_call=True
    )
    
    def validate_date_inputs(start_blur, end_blur, start_submit, end_submit, 
                            start_date_str, end_date_str, selected_gas, time_agg, hour_type):
        """Validate date inputs only on blur or submit"""
        import dash
        from dash.exceptions import PreventUpdate
        
        # Get callback context to determine which input triggered the callback
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Default styles
        valid_style = {'width': '140px'}
        invalid_style = {'width': '140px', 'border': '1px solid red'}
        
        # Initialize with current values and valid styles
        start_corrected = start_date_str
        end_corrected = end_date_str
        start_style = valid_style
        end_style = valid_style
        
        # Only validate if there was a blur or submit event
        if trigger_id not in ['surface-start-date-input', 'surface-end-date-input']:
            raise PreventUpdate
        
        try:
            # Get available dates for reference
            available_dates = get_available_dates(selected_gas, time_agg, hour_type)
            if not available_dates or len(available_dates) == 0:
                raise PreventUpdate
            
            # Only validate the input that triggered the callback (blur or submit)
            if 'start-date' in trigger_id:
                # Skip validation if input is blank or too short (user still typing)
                if not start_date_str or len(start_date_str) < 4:  # At least need year
                    return dash.no_update, dash.no_update, valid_style, dash.no_update
                    
                # Check if input potentially has a valid format before full validation
                if is_potentially_valid_date(start_date_str, time_agg):
                    start_valid, start_corrected, start_date_obj = validate_date_input(
                        start_date_str, time_agg, available_dates
                    )
                    
                    if not start_valid:
                        start_style = invalid_style
                    
                    # Only adjust end date if start date was successfully validated and end date exists
                    if start_date_obj and end_date_str:
                        _, _, end_date_obj = validate_date_input(
                            end_date_str, time_agg, available_dates
                        )
                        
                        if end_date_obj and end_date_obj < start_date_obj:
                            end_corrected = start_corrected
                            end_style = invalid_style
                        else:
                            # Don't update end date if it's not affected
                            end_corrected = dash.no_update
                    else:
                        end_corrected = dash.no_update
                else:
                    # Input not in a potentially valid format, don't correct it
                    start_style = invalid_style
                    return dash.no_update, dash.no_update, start_style, dash.no_update
                    
                return start_corrected, end_corrected, start_style, end_style
                    
            elif 'end-date' in trigger_id:
                # Skip validation if input is blank or too short (user still typing)
                if not end_date_str or len(end_date_str) < 4:  # At least need year
                    return dash.no_update, dash.no_update, dash.no_update, valid_style
                    
                # Check if input potentially has a valid format before full validation
                if is_potentially_valid_date(end_date_str, time_agg):
                    end_valid, end_corrected, end_date_obj = validate_date_input(
                        end_date_str, time_agg, available_dates
                    )
                    
                    if not end_valid:
                        end_style = invalid_style
                    
                    # Only adjust start date if end date was successfully validated and start date exists
                    if end_date_obj and start_date_str:
                        _, _, start_date_obj = validate_date_input(
                            start_date_str, time_agg, available_dates
                        )
                        
                        if start_date_obj and start_date_obj > end_date_obj:
                            start_corrected = end_corrected
                            start_style = invalid_style
                        else:
                            # Don't update start date if it's not affected
                            start_corrected = dash.no_update
                    else:
                        start_corrected = dash.no_update
                else:
                    # Input not in a potentially valid format, don't correct it
                    end_style = invalid_style
                    return dash.no_update, dash.no_update, dash.no_update, end_style
                
                return start_corrected, end_corrected, start_style, end_style
            
            # If we got here, something unexpected happened
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        except Exception as e:
            print(f"Error validating date inputs: {e}")
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        

    """Register callbacks for the surface observations page"""
    
    @app.callback(
            [Output('y-axis-min', 'disabled'),
             Output('y-axis-max', 'disabled')],
            Input('y-axis-scale-type', 'value')
        )
    def toggle_y_axis_inputs(scale_type):
        """Enable/disable y-axis range inputs based on scale type"""
        disabled = scale_type == 'auto'
        return disabled, disabled
    
    @app.callback(
        Output('surface-timeseries', 'figure', allow_duplicate=True),
        [Input('y-axis-scale-type', 'value'),
         Input('y-axis-min', 'value'),
         Input('y-axis-max', 'value')],
        [State('surface-timeseries', 'figure')],
        prevent_initial_call=True
    )
    def update_y_axis_range(scale_type, y_min, y_max, current_fig):
        """Update the y-axis range of the time series plot"""
        if not current_fig:
            return dash.no_update
            
        fig_copy = current_fig.copy()
        
        # Update y-axis settings
        if scale_type == 'fixed' and y_min is not None and y_max is not None:
            try:
                y_min = float(y_min)
                y_max = float(y_max)
                if y_min < y_max:
                    fig_copy['layout']['yaxis'].update({
                        'autorange': False,
                        'range': [y_min, y_max],
                        'fixedrange': True,  # Lock the range
                        'constrain': 'domain'  # Ensure range is strictly maintained
                    })
                    # Update layout to prevent auto-ranging
                    fig_copy['layout'].update({
                        'uirevision': True,  # Maintain UI state
                        'yaxis_autorange': False
                    })
                else:
                    # Reset to auto range if values are invalid
                    fig_copy['layout']['yaxis'].update({
                        'autorange': True,
                        'fixedrange': False
                    })
            except (ValueError, TypeError):
                # Reset to auto range if values are invalid
                fig_copy['layout']['yaxis'].update({
                    'autorange': True,
                    'fixedrange': False
                })
        else:
            # Reset to auto range for 'auto' mode
            fig_copy['layout']['yaxis'].update({
                'autorange': True,
                'fixedrange': False
            })
                
        return fig_copy

# Also need to modify the main figure update callback
    @app.callback(
        [Output('surface-map', 'figure', allow_duplicate=True),
         Output('surface-timeseries', 'figure', allow_duplicate=True),
         Output('surface-map-state-store', 'data', allow_duplicate=True)],
        [Input('surface-gas-selector', 'value'),
         Input('surface-site-selector', 'value'),
         Input('surface-analysis-type', 'value'),
         Input('surface-time-agg', 'value'),
         Input('surface-timeseries-type', 'value'),
         Input('surface-hour-type', 'value'),
         Input('surface-map', 'relayoutData'),
         Input('surface-time-slider', 'value')],
        [State('surface-map-state-store', 'data'),
         State('y-axis-scale-type', 'value'),
         State('y-axis-min', 'value'),
         State('y-axis-max', 'value')],
        prevent_initial_call=True
    )
    def update_figures(selected_gas, selected_sites, analysis_type, time_agg, 
                      timeseries_type, hour_type, relayoutData, slider_value, 
                      map_state, scale_type, y_min, y_max):
        """Update both figures based on user selections"""
        map_fig, time_series_fig, updated_map_state = update_surface_figures(
            selected_gas, selected_sites, analysis_type, time_agg,
            timeseries_type, relayoutData, map_state, slider_value,
            hour_type
        )
        
        # Always respect the fixed y-axis settings when they exist
        if scale_type == 'fixed' and y_min is not None and y_max is not None:
            try:
                y_min = float(y_min)
                y_max = float(y_max)
                if y_min < y_max:
                    time_series_fig['layout']['yaxis'].update({
                        'autorange': False,
                        'range': [y_min, y_max],
                        'fixedrange': True,  # Lock the range
                        'constrain': 'domain'  # Ensure range is strictly maintained
                    })
                    # Also update the layout config to prevent auto-ranging
                    time_series_fig['layout'].update({
                        'uirevision': True,  # Maintain UI state
                        'yaxis_autorange': False
                    })
            except (ValueError, TypeError):
                pass
                
        return map_fig, time_series_fig, updated_map_state

    
    
    @app.callback(
    [Output('y-axis-min', 'value'),
     Output('y-axis-max', 'value')],
    Input('surface-timeseries', 'figure'),
    State('y-axis-scale-type', 'value'),
    prevent_initial_call=True
    )
    def update_y_axis_inputs(figure, scale_type):
        """Update the y-axis input fields with current plot range"""
        if not figure or scale_type == 'auto':
            return None, None
            
        try:
            y_range = figure['layout']['yaxis'].get('range', [])
            if len(y_range) == 2:
                return round(y_range[0], 2), round(y_range[1], 2)
        except (KeyError, IndexError):
            pass
        
        return None, None
    
    
    @app.callback(
    Output("surface-temporal-copy-ai", "n_clicks"),
    Input("surface-temporal-copy-ai", "n_clicks"),
    [State('surface-timeseries', 'figure'),
     State('surface-gas-selector', 'value'),
     State('surface-time-agg', 'value'),
     State('surface-hour-type', 'value')],  # Added hour_type state
    prevent_initial_call=True
)
    def copy_temporal_data_for_ai(n_clicks, figure, gas_type, time_agg, hour_type):
        """Format temporal data for AI analysis and copy to clipboard"""
        if not n_clicks:
            return dash.no_update
            
        try:
            dates = []
            data_by_site = {}
            
            unit = 'ppm' if gas_type == 'co2' else 'ppb'
            for trace in figure['data']:
                site = trace['name']
                data_by_site[site] = trace['y']
                if not dates:
                    # Keep full datetime string for processing
                    dates = trace['x']
    
            text = f"{gas_type.upper()} Monitoring Data\n"
            text += f"Hour Type: {'Afternoon Hours' if hour_type == 'afternoon' else 'All Hours'}\n"
            text += f"Unit: {unit}\n"
            text += f"Sites: {', '.join(data_by_site.keys())}\n"
            
            agg_text = {
                'H': 'Hourly',
                'D': 'Daily',
                'W': 'Weekly',
                'MS': 'Monthly'
            }
            text += f"Time Resolution: {agg_text.get(time_agg, 'Unknown')}\n\n"
            
            # Header
            header = ["Date"] + list(data_by_site.keys())
            text += ",".join(header) + "\n"
            
            # Data rows with appropriate date format
            for i, date_str in enumerate(dates):
                # Format date based on aggregation
                date = pd.to_datetime(date_str)
                if time_agg == 'H':
                    formatted_date = date.strftime('%Y-%m-%d %H:00')  # Include hour for hourly data
                elif time_agg == 'MS':
                    formatted_date = date.strftime('%Y-%m')  # YYYY-MM for monthly
                else:
                    formatted_date = date.strftime('%Y-%m-%d')  # YYYY-MM-DD for weekly/daily
                
                row = [formatted_date]
                for site in data_by_site:
                    val = data_by_site[site][i]
                    row.append(f"{val:.2f}" if val and not np.isnan(val) else "NaN")
                text += ",".join(row) + "\n"
    
            pyperclip.copy(text)
            return 0
            
        except Exception as e:
            print(f"Error copying data: {e}")
            return dash.no_update
        
    @app.callback(
    Output("download-dataframe-csv", "data"),
    Input("surface-temporal-download-csv", "n_clicks"),
    [State('surface-timeseries', 'figure'),
     State('surface-gas-selector', 'value'),
     State('surface-timeseries-type', 'value'),
     State('surface-analysis-type', 'value'),
     State('surface-hour-type', 'value')],  # Added hour_type state
    prevent_initial_call=True
)
    def download_temporal_csv(n_clicks, figure, gas_type, timeseries_type, analysis_type, hour_type):
        """Download the current temporal view as a CSV file"""
        if not n_clicks:
            return None
        
        try:
            # Create an empty DataFrame to store the consolidated data
            consolidated_df = pd.DataFrame()
            
            # Extract data from each trace
            for trace in figure['data']:
                dates = pd.to_datetime(trace['x'])
                values = trace['y']
                site_name = trace['name']
                
                # Create column name based on metadata
                if timeseries_type == 'background':
                    column_name = f"{site_name}_background_subtracted_{gas_type.upper()}"
                else:
                    metric = 'mean' if analysis_type == 'mean' else 'std'
                    column_name = f"{site_name}_{metric}_{gas_type.upper()}"
                
                # Create a DataFrame for the current trace
                df = pd.DataFrame({
                    'datetime_UTC': dates,
                    column_name: values
                })
                
                # Merge with consolidated DataFrame
                if consolidated_df.empty:
                    consolidated_df = df
                else:
                    consolidated_df = pd.merge(
                        consolidated_df, 
                        df, 
                        on='datetime_UTC', 
                        how='outer'
                    )
            
            # Sort by date
            consolidated_df.sort_values('datetime_UTC', inplace=True)
            
            # Generate filename with descriptive components
            start_date = consolidated_df['datetime_UTC'].min().strftime('%Y%m%d')
            end_date = consolidated_df['datetime_UTC'].max().strftime('%Y%m%d')
            analysis_str = 'background' if timeseries_type == 'background' else analysis_type
            filename = f"surface_{gas_type}_{analysis_str}_{hour_type}_{start_date}_{end_date}.csv"
            
            return dict(content=consolidated_df.to_csv(index=False), 
                       filename=filename,
                       type='text/csv')
            
        except Exception as e:
            print(f"Error in temporal CSV download: {e}")
            return None
    


    # Step 3: Modify the existing slider callback to update date inputs

    @app.callback(
        [Output('surface-time-slider', 'min'),
         Output('surface-time-slider', 'max'),
         Output('surface-time-slider', 'value'),
         Output('surface-time-slider', 'marks'),
         Output('surface-period-text', 'children'),
         # Add outputs for date inputs
         Output('surface-start-date-input', 'value'),
         Output('surface-end-date-input', 'value')],
        [Input('surface-gas-selector', 'value'),
         Input('surface-time-agg', 'value'),
         Input('surface-hour-type', 'value'),
         Input('surface-time-slider', 'value')]
    )
    def update_time_slider(selected_gas, time_agg, hour_type, current_slider_value):
        available_dates = get_available_dates(selected_gas, time_agg, hour_type)

        if len(available_dates) == 0:
            return 0, 100, [0, 100], {}, "", "", ""

        max_index = len(available_dates) - 1

        # Ensure slider value is within bounds
        if current_slider_value is None or \
            current_slider_value[0] > max_index or \
            current_slider_value[1] > max_index:
            current_slider_value = [0, max_index]

        # Create marks
        if len(available_dates) <= 8:
            mark_indices = range(len(available_dates))
        else:
            mark_indices = [int(i * max_index / 7) for i in range(8)]

        marks = {
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
            for i in mark_indices
        }

        # Determine date format based on time aggregation
        if time_agg == 'H':
            format_string = '%Y-%m-%d :: %H'
        elif time_agg == 'MS':
            format_string = '%Y-%m'
        else:  # D or W
            format_string = '%Y-%m-%d'
        
        # Format dates for display
        start_date = available_dates[current_slider_value[0]].strftime(format_string)
        end_date = available_dates[current_slider_value[1]].strftime(format_string)
        period_text = f"Duration: {start_date} - {end_date}"

        # Return with additional outputs for date inputs
        return (
            0, 
            max_index, 
            current_slider_value, 
            marks, 
            period_text,
            start_date,  # Value for start date input
            end_date     # Value for end date input
        )


    # Add callback to update site options when gas type changes
    @app.callback(
        Output('surface-site-selector', 'options'),
        Input('surface-gas-selector', 'value')
    )
    def update_site_options(selected_gas):
        """Update site options based on selected gas type"""
        return get_site_options(gas_type=selected_gas, site_dict=prm.site_dict)



    """Register callbacks for the surface observations page"""

    @app.callback(
        [Output("controls-collapse", "is_open"),
          Output("controls-chevron", "className")],
        [Input("controls-toggle", "n_clicks")],
        [State("controls-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_controls_collapse(n_clicks, is_open):
        """Toggle the controls section collapse state"""
        if n_clicks:
            return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
        return is_open, "fas fa-chevron-down ms-2"

    @app.callback(
        Output("help-modal_surface", "is_open", allow_duplicate=True),
        [Input("surface-help-button", "n_clicks"),
          Input("help-close", "n_clicks")],
        [State("help-modal_surface", "is_open")],
        prevent_initial_call=True
    )
    def toggle_help_modal(help_clicks, close_clicks, is_open):
        """Toggle the help modal visibility"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return is_open
    
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == "surface-help-button":
            return not is_open
        elif button_id == "help-close":
            return False
        return is_open

    @app.callback(
        [Output('surface-timeseries-type', 'options'),
          Output('surface-timeseries-type', 'value', allow_duplicate=True),
          Output('surface-analysis-type', 'disabled'),
          Output('surface-analysis-type', 'value', allow_duplicate=True)],
        [Input('surface-gas-selector', 'value'),
          Input('surface-timeseries-type', 'value')],
        prevent_initial_call=True
    )
    def update_control_options(selected_gas, timeseries_type):
        """Update control options based on gas selection"""
        # Default options for time series type
        timeseries_options = [
            {'label': 'Raw Data', 'value': 'raw'},
            {'label': 'Background Subtracted', 'value': 'background'}
        ]
    
        # For CO, only raw data is available
        if selected_gas == 'co':
            timeseries_options = [{'label': 'Raw Data', 'value': 'raw'}]
            new_timeseries_type = 'raw'
            analysis_disabled = False
            analysis_value = dash.no_update
        else:
            # For CO2 and CH4
            new_timeseries_type = timeseries_type
            if timeseries_type == 'background':
                analysis_disabled = True
                analysis_value = 'mean'
            else:
                analysis_disabled = False
                analysis_value = dash.no_update
    
        return timeseries_options, new_timeseries_type, analysis_disabled, analysis_value

    # Step 4: Update the reset callback to include the new components

    @app.callback(
        [Output('surface-site-selector', 'value', allow_duplicate=True),
         Output('surface-gas-selector', 'value', allow_duplicate=True),
         Output('surface-timeseries-type', 'value', allow_duplicate=True),
         Output('surface-time-agg', 'value', allow_duplicate=True),
         Output('surface-analysis-type', 'value', allow_duplicate=True),
         Output('surface-hour-type', 'value', allow_duplicate=True),
         Output('surface-time-slider', 'value', allow_duplicate=True),
         Output('left-panel', 'style', allow_duplicate=True),
         Output('right-panel', 'style', allow_duplicate=True),
         Output('restore-button', 'style', allow_duplicate=True),
         Output('expansion-state', 'data', allow_duplicate=True),
         Output('surface-map', 'figure', allow_duplicate=True),
         Output('surface-timeseries', 'figure', allow_duplicate=True),
         Output('surface-map-state-store', 'data', allow_duplicate=True),
         # Add outputs for date inputs
         Output('surface-start-date-input', 'value', allow_duplicate=True),
         Output('surface-end-date-input', 'value', allow_duplicate=True)],
        [Input('surface-restart-button', 'n_clicks')],
        [State('surface-map-state-store', 'data')],
        prevent_initial_call=True
    )
    def reset_to_initial_state(n_clicks, map_state):
        """Reset all controls and visualizations to their initial state"""
        if not n_clicks:
            return dash.no_update

        # Default values
        selected_gas = 'co2'
        selected_sites = ['GRA']
        analysis_type = 'mean'
        time_agg = 'MS'
        timeseries_type = 'raw'
        hour_type = 'afternoon'
        
        # Get available dates for default slider value
        available_dates = get_available_dates(selected_gas, time_agg, hour_type)
        default_slider_value = [0, len(available_dates) - 1] if len(available_dates) > 0 else [0, 100]
        
        # Get formatted dates for inputs
        start_date = ""
        end_date = ""
        if len(available_dates) > 0:
            format_string = '%Y-%m' if time_agg == 'MS' else '%Y-%m-%d'
            start_date = available_dates[0].strftime(format_string)
            end_date = available_dates[-1].strftime(format_string)
        
        # Default panel styles
        base_style = {'flex': '1', 'minWidth': '0'}
        
        # Update figures with default values
        map_fig, time_series_fig, updated_map_state = update_surface_figures(
            selected_gas,
            selected_sites,
            analysis_type,
            time_agg,
            timeseries_type,
            None,  # relayoutData
            map_state,
            default_slider_value,
            hour_type
        )

        return (
            selected_sites,          # site selector
            selected_gas,           # gas selector
            timeseries_type,        # timeseries type
            time_agg,              # time aggregation
            analysis_type,         # analysis type
            hour_type,            # hour type
            default_slider_value,  # time slider value
            base_style,           # left panel style
            base_style,           # right panel style
            {'display': 'none'},  # restore button style
            'none',              # expansion state
            map_fig,            # map figure
            time_series_fig,    # timeseries figure
            updated_map_state,  # map state
            start_date,        # start date input
            end_date          # end date input
        )

    # Update the main figures callback to match the new function signature
    @app.callback(
    [Output("stats-modal", "is_open"),
     Output("stats-table", "children")],
    [Input("stats-button", "n_clicks"),
     Input("stats-close", "n_clicks")],
    [State("stats-modal", "is_open"),
     State('surface-timeseries', 'figure'),
     State('surface-gas-selector', 'value')],
    prevent_initial_call=True
)
    def toggle_stats_modal(stats_clicks, close_clicks, is_open, figure_data, selected_gas):
        """Toggle the stats modal and compute statistics"""
        import numpy as np
        
        ctx = dash.callback_context
        if not ctx.triggered:
            return is_open, []
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == "stats-close":
            return False, []
        
        if button_id == "stats-button":
            if not figure_data or 'data' not in figure_data:
                return not is_open, [
                    html.Tr([
                        html.Td("No data available")
                    ])
                ]
            
            # Extract data from the figure
            stats_rows = []
            header_row = [html.Th("Statistic")]
            
            # Get units based on gas type
            units = 'ppm' if selected_gas == 'co2' else 'ppb'
            
            try:
                # Process each trace (time series) in the figure
                for trace in figure_data['data']:
                    site_name = trace['name']
                    # Convert y values to numeric numpy array, filtering out any None or non-numeric values
                    y_values = np.array([y for y in trace['y'] if y is not None and not np.isnan(y)], dtype=np.float64)
                    
                    if len(y_values) == 0:
                        continue
                    
                    # Compute statistics for this trace
                    stats = essential_stats(y_values)
                    
                    # Add site to header row
                    header_row.append(html.Th(f"{site_name} ({units})"))
                    
                    # If this is the first trace, create rows for each statistic
                    if len(stats_rows) == 0:
                        for _, row in stats.iterrows():
                            stats_rows.append([
                                html.Td(row['STAT_NAME']),
                                html.Td(f"{row['ESTIMATE']:.2f}")
                            ])
                    else:
                        # Add this site's values to existing rows
                        for i, row in stats.iterrows():
                            stats_rows[i].append(html.Td(f"{row['ESTIMATE']:.2f}"))
                
                # Construct table elements
                table_header = [html.Thead(html.Tr(header_row))]
                table_body = [html.Tbody([html.Tr(row) for row in stats_rows])]
                
                return not is_open, table_header + table_body
                
            except Exception as e:
                print(f"Error computing statistics: {str(e)}")
                return not is_open, [
                    html.Tr([
                        html.Td(f"Error computing statistics: {str(e)}")
                    ])
                ]
        
        return is_open, []
    
    @app.callback(
    Output('surface-time-slider', 'value', allow_duplicate=True),
    [Input('surface-start-date-input', 'n_blur'),  # Changed to use blur events
     Input('surface-end-date-input', 'n_blur'),    # Changed to use blur events
     Input('surface-start-date-input', 'n_submit'),
     Input('surface-end-date-input', 'n_submit')],
    [State('surface-start-date-input', 'value'),
     State('surface-end-date-input', 'value'),
     State('surface-gas-selector', 'value'),
     State('surface-time-agg', 'value'),
     State('surface-hour-type', 'value')],
    prevent_initial_call=True
)
    def update_slider_from_date_inputs(start_blur, end_blur, start_submit, end_submit,
                                     start_date_str, end_date_str, selected_gas, time_agg, hour_type):
        """Update slider position based on validated date inputs"""
        # Skip if either date is not provided
        if not start_date_str or not end_date_str:
            return dash.no_update
        
        # Skip if dates are not in a valid format
        if not is_potentially_valid_date(start_date_str, time_agg) or not is_potentially_valid_date(end_date_str, time_agg):
            return dash.no_update
        
        try:
            # Get available dates
            available_dates = get_available_dates(selected_gas, time_agg, hour_type)
            if len(available_dates) == 0:
                return dash.no_update
            
            # Validate inputs
            start_valid, _, start_date = validate_date_input(start_date_str, time_agg, available_dates)
            end_valid, _, end_date = validate_date_input(end_date_str, time_agg, available_dates)
            
            # Only proceed if both dates are valid
            if start_date is None or end_date is None:
                return dash.no_update
            
            # Ensure end date is not before start date
            if end_date < start_date:
                end_date = start_date
            
            # Find closest indices
            available_dates_pd = pd.to_datetime(available_dates)
            
            # Get index of closest date not earlier than start_date
            start_idx = 0
            for i, date in enumerate(available_dates_pd):
                if date >= start_date:
                    start_idx = i
                    break
            
            # Get index of closest date not later than end_date
            end_idx = len(available_dates_pd) - 1
            for i in range(len(available_dates_pd) - 1, -1, -1):
                if available_dates_pd[i] <= end_date:
                    end_idx = i
                    break
            
            # Return new slider value
            return [start_idx, end_idx]
        
        except Exception as e:
            print(f"Error updating slider from date inputs: {e}")
            return dash.no_update
