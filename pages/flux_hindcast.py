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
from utils import constants as prm
from utils import conc_func as cfunc 
import math
import tempfile
import zipfile
import gcsfs
import io



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


def init():
    """Initialize flux forecast components with detailed logging."""
    global FLUX_DATA, UNCERTAINTY_DATA, DATES, LAT, LON, LAT_GRID, LON_GRID, available_dates
    global ZIP_DATA, CENSUS_DATA, CUSTOM_DATA
    
    logger.info("--- Starting flux hindcast data initialization ---")
    check_gcs_files()
    
    try:
        # Initialize GCS filesystem
        fs = gcsfs.GCSFileSystem()
        
        # --- Step 1: Load NetCDF data ---
        filename = f"{GCS_NC_FOLDER_PATH}fluxresults1.nc" 
        logger.info(f"Attempting to load NetCDF data from: {filename}")
        
        # Check if file exists on GCS
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
        logger.info(f"Unpacked NetCDF data. Found {len(available_dates)} available dates.")

        # --- Step 2: Load spatial aggregation data from HDF5 ---
        try:
            spatial_hdf_filename = f"{GCS_HDF_FOLDER_PATH}spatial_data.h5"
            logger.info(f"Attempting to load HDF5 spatial data from: {spatial_hdf_filename}")
            
            # Check if HDF5 file exists on GCS
            if not fs.exists(spatial_hdf_filename):
                logger.warning(f"HDF5 file does not exist on GCS: {spatial_hdf_filename}")
                ZIP_DATA, CENSUS_DATA, CUSTOM_DATA = {}, {}, {}
            else:
                spatial_data = cfunc.load_dicts_from_hdf(spatial_hdf_filename, ['zip', 'census', 'custom'])
                logger.info("Successfully loaded HDF5 data.")
                
                # Store the data directly as loaded from HDF
                ZIP_DATA = {k: v for k, v in spatial_data.items() if k.startswith('zip_')}
                CENSUS_DATA = {k: v for k, v in spatial_data.items() if k.startswith('census_')}
                CUSTOM_DATA = {k: v for k, v in spatial_data.items() if k.startswith('custom_')}
                logger.info("Successfully parsed and stored spatial aggregation data.")
            
        except Exception:
            # This will log the full error if HDF5 loading fails
            logger.error("Failed to load or process HDF5 spatial aggregation data.", exc_info=True)
            ZIP_DATA, CENSUS_DATA, CUSTOM_DATA = {}, {}, {}
        
        logger.info("--- Flux hindcast data initialization successful. ---")
        return True
        
    except Exception:
        # THIS IS THE MOST IMPORTANT PART: It will log the full traceback for any failure
        logger.error("--- A critical error occurred during flux hindcast data initialization. ---", exc_info=True)
        return False
    
    
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
            raise KeyError(f"No data available for {agg_type} aggregation")
            
        # Get the keys with the correct prefix
        est_flux_key = f'{agg_type}_est_flux'
        unc_flux_key = f'{agg_type}_unc_flux'
        
        # Get the data from the dictionary
        est_flux = data_dict.get(est_flux_key)
        unc_flux = data_dict.get(unc_flux_key)
        
        if est_flux is None or unc_flux is None:
            print(f"Warning: Missing data for {agg_type} aggregation")
            print(f"Available keys: {list(data_dict.keys())}")
            return None, None
            
        return est_flux, unc_flux
    
    except Exception as e:
        print(f"Error getting {agg_type} data: {e}")
        print(f"Data dictionary for {agg_type}: {data_dict if 'data_dict' in locals() else 'Not created'}")
        return None, None

def load_netcdf_data(filename):
    """
    Load NetCDF data from GCS and ensure it is read in [latitude, longitude, time] order.

    Args:
        filename (str): GCS path to the NetCDF file.

    Returns:
        dict: Dictionary containing flux, uncertainty, and metadata.
    """
    import gcsfs
    import tempfile
    import os
    
    # Initialize GCS filesystem
    fs = gcsfs.GCSFileSystem()
    
    # Create a temporary file to download the NetCDF data
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as temp_file:
        try:
            # Download the file from GCS to temporary local file
            fs.get(filename, temp_file.name)
            
            # Open the NetCDF file using the local temporary file
            dataset = xr.open_dataset(temp_file.name, engine="h5netcdf")

            # Read variables
            # Transpose and convert to float32 where applicable
            flux = dataset['flux'].transpose('latitude', 'longitude', 'time').values.astype(np.float32)
            uncertainty = dataset['uncertainty'].transpose('latitude', 'longitude', 'time').values.astype(np.float32)
            time = dataset['time'].values  # POSIX time remains unchanged
            lat = dataset['lat'].values.astype(np.float32)
            lon = dataset['lon'].values.astype(np.float32)
            lat_grid = dataset['lat_grid'].values.T.astype(np.float32)  # Combined transpose and float32 conversion
            lon_grid = dataset['lon_grid'].values.T.astype(np.float32)  # Combined transpose and float32 conversion

            # Close the dataset
            dataset.close()
            
            return {
                'flux': flux,
                'uncertainty': uncertainty,
                'time': time,
                'latitude': lat,
                'longitude': lon,
                'lat_grid': lat_grid,
                'lon_grid': lon_grid,
            }
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass  # File might already be deleted
    
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
                            dbc.Button([
                                "Documentation ",
                                html.I(className="fas fa-file-pdf")
                            ], color="secondary", className="me-2", id="hindcast-doc-button"),
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

# def register_callbacks(app):
#     """Register callbacks for flux hindcast page"""
#     # Add callback registrations here
#     pass

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
                transparency,
                relayout_data, map_state):
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
                # Get pre-computed spatial data from HDF5
                spatial_data = {
                'zip': ZIP_DATA,
                'census': CENSUS_DATA,
                'custom': CUSTOM_DATA
                }.get(spatial_agg)
    
                if not spatial_data:
                    raise ValueError(f"No data available for {spatial_agg} aggregation")
    
                # Get appropriate data based on display type
                key = f"{spatial_agg}_{'est_flux' if display_type == 'flux' else 'unc_flux'}"
                df = spatial_data[key]
    
                # Get boundaries for the chosen aggregation from GCS
                GCS_SHAPEFILE_FOLDER_PATH = "gs://la-megacity-dashboard-data-1/data/shapefiles/"
                boundaries_key = {
                    'zip': 'zip_code_socab',
                    'census': 'census_tract_clipped',
                    'custom': 'zones_partitoned'
                }[spatial_agg]
                
                shapefile_names = {
                    'zip_code_socab': 'zip_code_socab.shp',
                    'census_tract_clipped': 'census_tract_clipped.shp',
                    'zones_partitoned': 'zones_partitoned.shp'
                }
                shapefile_path = f"{GCS_SHAPEFILE_FOLDER_PATH}{shapefile_names[boundaries_key]}"
                boundaries = gpd.read_file(shapefile_path)
    
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
                    df = pd.DataFrame({'date': dates, 'value': domain_means})
    
                    # Apply monthly aggregation only if selected
                    if temporal_agg != 'single':
                        df = df.set_index('date').resample('ME').mean().reset_index()
    
                    timeseries_fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=df['value'],
                        mode='lines+markers',
                        name='Domain Average',
                        line=dict(color='rgba(0,0,255,0.7)')
                    ))
                    timeseries_fig.update_layout(title="Domain-wide Average Time Series")
                else:
                    # Show average across all regions for aggregated data
                    key = f"{spatial_agg}_{'est_flux' if display_type == 'flux' else 'unc_flux'}"
                    df = spatial_data[key]
                    dates = pd.to_datetime(df.columns)
                    region_mean = df.mean()
    
                    # Create DataFrame for time series
                    temp_df = pd.DataFrame({'date': dates, 'value': region_mean})
    
                    # Apply monthly aggregation only if selected
                    if temporal_agg != 'single':
                        temp_df = temp_df.set_index('date').resample('ME').mean().reset_index()
    
                    timeseries_fig.add_trace(go.Scatter(
                        x=temp_df['date'],
                        y=temp_df['value'],
                        mode='lines+markers',
                        name='Regional Average',
                        line=dict(color='rgba(0,0,255,0.7)')
                    ))
                    timeseries_fig.update_layout(title="Regional Average Time Series")
            else:
                if spatial_agg == 'native':
                    # Handle click for native resolution
                    data = FLUX_DATA if display_type == 'flux' else UNCERTAINTY_DATA
                    location_id = click_data['points'][0]['location']
                    i, j = map(int, location_id.split('_'))
    
                    # Get full time series for clicked location
                    time_series = data[i, j, :]
                    dates = pd.to_datetime(DATES, unit='s')
                    df = pd.DataFrame({'date': dates, 'value': time_series})
    
                    # Apply monthly aggregation only if selected
                    if temporal_agg != 'single':
                        df = df.set_index('date').resample('ME').mean().reset_index()
    
                    title = f"{display_type.capitalize()} Time Series for Latitude {LAT_GRID[i,j]:.2f}°N, Longitude {LON_GRID[i,j]:.2f}°W"
                else:
                    # Handle click for aggregated data
                    key = f"{spatial_agg}_{'est_flux' if display_type == 'flux' else 'unc_flux'}"
                    df_agg = spatial_data[key]
                    location_id = click_data['points'][0]['location']
                    if spatial_agg == 'custom':
                        location_id = int(location_id)
    
                    # Get full time series for clicked location
                    time_series = df_agg.loc[location_id]
                    dates = pd.to_datetime(df_agg.columns)
                    df_temp = pd.DataFrame({'date': dates, 'value': time_series.values})
    
                    # Apply monthly aggregation only if selected
                    if temporal_agg != 'single':
                        df_temp = df_temp.set_index('date').resample('ME').mean().reset_index()
    
                    df = df_temp  # Reassign for consistent plotting
                    # Create title based on aggregation type and display type
                    if spatial_agg == 'zip':
                        title = f"{display_type.capitalize()} Time Series for Zipcode {location_id}"
                    elif spatial_agg == 'census':
                        title = f"{display_type.capitalize()} Time Series for Census Tract {location_id}"
                    else:  # custom regions
                        title = f"{display_type.capitalize()} Time Series for Region {location_id}"
    
                timeseries_fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['value'],
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
            print(f"Error in dashboard update: {e}")
            # Return empty figures and no map state update on error
            return go.Figure(), go.Figure(), dash.no_update
    
    
    
    # def update_dashboard(display_type, spatial_agg, temporal_agg, time_range, 
    #                 click_data, animation_status, scale_type, scale_min, scale_max,
    #                 transparency,  # Add this parameter
    #                 relayout_data, map_state):
    #     """Update both map and time series based on user interactions"""
        
    #     # Initialize figures
    #     map_fig = go.Figure()
    #     timeseries_fig = go.Figure()
        
    #     colorbar_settings = dict(
    #         orientation='v',
    #         thickness=20,
    #         len=0.9,
    #         y=0.5,
    #         yanchor='middle',
    #         x=1.02,
    #         xanchor='left',
    #         title=dict(
    #             text='µmol m² sec⁻²' if display_type == 'flux' else 'Uncertainty',
    #             side='right'
    #         ),
    #         tickfont=dict(size=10)
    #     )
        
    #     try:
    #         # Get data based on display type and spatial aggregation                       
            
    #         if spatial_agg == 'native':
    #             # Use raw netCDF data for native resolution
    #             data = FLUX_DATA if display_type == 'flux' else UNCERTAINTY_DATA
                
    #             # For map: Always compute mean over slider range
    #             current_data = data[:, :, time_range[0]:time_range[1]+1].mean(axis=2)
                
    #             # Create grid cells for native resolution
    #             features = []
    #             values = []
    #             customdata = []  # For additional hover data
                
    #             for i in range(len(LAT_GRID)-1):
    #                 for j in range(len(LON_GRID[0])-1):
    #                     polygon = [[
    #                         [LON_GRID[i,j], LAT_GRID[i,j]],
    #                         [LON_GRID[i,j+1], LAT_GRID[i,j+1]],
    #                         [LON_GRID[i+1,j+1], LAT_GRID[i+1,j+1]],
    #                         [LON_GRID[i+1,j], LAT_GRID[i+1,j]],
    #                         [LON_GRID[i,j], LAT_GRID[i,j]]
    #                     ]]
                        
    #                     grid_id = f"{i}_{j}"
    #                     value = current_data[i,j]
    #                     features.append({
    #                         "type": "Feature",
    #                         "geometry": {
    #                             "type": "Polygon",
    #                             "coordinates": polygon
    #                         },
    #                         "id": grid_id
    #                     })
    #                     values.append(value)
    #                     customdata.append([LAT_GRID[i,j], LON_GRID[i,j]])  # Store lat/lon for hover
                
    #             geojson = {
    #                 "type": "FeatureCollection",
    #                 "features": features
    #             }
                
    #             # Create base choroplethmapbox arguments
    #             choroplethmapbox_args = {
    #                 'geojson': geojson,
    #                 'locations': [f["id"] for f in features],
    #                 'z': values,
    #                 'colorscale': 'Turbo',
    #                 'marker_opacity': transparency if transparency is not None else 0.7,
    #                 'showscale': True,
    #                 'colorbar': colorbar_settings,
    #                 'customdata': customdata,
    #                 'hovertemplate': (
    #                     "Grid ID: %{location}<br>" +
    #                     "Value: %{z:.4f}<br>" +
    #                     "Lat: %{customdata[0]:.4f}<br>" +
    #                     "Lon: %{customdata[1]:.4f}" +
    #                     "<extra></extra>"
    #                 )
    #             }
                
    #             # Add fixed scale if specified
    #             if scale_type == 'fixed' and scale_min is not None and scale_max is not None:
    #                 choroplethmapbox_args.update({
    #                     'zmin': float(scale_min),
    #                     'zmax': float(scale_max)
    #                 })
                
    #             map_fig.add_trace(go.Choroplethmapbox(**choroplethmapbox_args))            
            
    #         else:
    #             # Get pre-computed spatial data from HDF5
    #             spatial_data = {
    #             'zip': ZIP_DATA,
    #             'census': CENSUS_DATA,
    #             'custom': CUSTOM_DATA
    #             }.get(spatial_agg)
                
    #             if not spatial_data:
    #                 raise ValueError(f"No data available for {spatial_agg} aggregation")
                
    #             # Get appropriate data based on display type
    #             key = f"{spatial_agg}_{'est_flux' if display_type == 'flux' else 'unc_flux'}"
    #             df = spatial_data[key]
                
    #             # Get boundaries for the chosen aggregation
    #             boundaries_file = {
    #                 'zip': 'zip_code_socab.shp',
    #                 'census': 'census_tract_clipped.shp',
    #                 'custom': 'zones_partitoned.shp'
    #             }[spatial_agg]
                
    #             boundaries = gpd.read_file(os.path.join(prm.SHAPEFILE_PATH, boundaries_file))
                
    #             # Add the feature ID mapping
    #             feature_id_mapping = {
    #                 'zip': 'ZIP_CODE',
    #                 'census': 'TRACTCE',
    #                 'custom': 'Zones'
    #             }
    #             feature_id_key = f"properties.{feature_id_mapping[spatial_agg]}"
                
    #             # Get centroids for hover display
    #             centroids = spatial_data.get(f'{spatial_agg}_centroids', {})
                
    #             # For map: Always compute mean over slider range
    #             selected_dates = pd.to_datetime(DATES[time_range[0]:time_range[1]+1], unit='s')
    #             current_data = df[selected_dates].mean(axis=1)
                
    #             # Create base choroplethmapbox arguments
    #             choroplethmapbox_args = {
    #                 'geojson': boundaries.__geo_interface__,
    #                 'featureidkey': feature_id_key,
    #                 'locations': current_data.index,
    #                 'z': current_data.values,
    #                 'colorscale': 'Turbo',
    #                 'marker_opacity': transparency if transparency is not None else 0.7,
    #                 'showscale': True,
    #                 'colorbar': colorbar_settings,
    #                 'hovertemplate': (
    #                     f"{spatial_agg.title()} ID: %{{location}}<br>" +
    #                     "Value: %{z:.4f}" +
    #                     "<extra></extra>"
    #                 )
    #             }
                
    #             # Add fixed scale if specified
    #             if scale_type == 'fixed' and scale_min is not None and scale_max is not None:
    #                 choroplethmapbox_args.update({
    #                     'zmin': float(scale_min),
    #                     'zmax': float(scale_max)
    #                 })
                    
    #             map_fig.add_trace(go.Choroplethmapbox(**choroplethmapbox_args))
    
    #         # Calculate bounds for optimal view
    #         lat_bounds = [np.min(LAT_GRID), np.max(LAT_GRID)]
    #         lon_bounds = [np.min(LON_GRID), np.max(LON_GRID)]
            
    #         center_lat = (lat_bounds[0] + lat_bounds[1]) / 2
    #         center_lon = (lon_bounds[0] + lon_bounds[1]) / 2
            
    #         # Calculate zoom level to fit the data extent
    #         lon_range = lon_bounds[1] - lon_bounds[0]
    #         lat_range = lat_bounds[1] - lat_bounds[0]
    #         zoom = min(
    #             math.log2(360 / lon_range) - 1,
    #             math.log2(180 / lat_range) - 1
    #         )
    #         zoom = max(min(zoom, 10), 7)  # Constrain zoom between 7 and 10
            
    #         # Base layout settings for map
    #         map_layout = dict(
    #             autosize=True,
    #             mapbox=dict(
    #                 style='open-street-map',
    #                 zoom=zoom,
    #                 center=dict(lat=center_lat, lon=center_lon),
    #             ),
    #             margin=dict(l=0, r=70, t=0, b=0),
    #             uirevision='constant',
    #             paper_bgcolor='white',
    #             plot_bgcolor='white'
    #         )
            
    #         # Base layout for time series
    #         timeseries_layout = dict(
    #             autosize=True,
    #             xaxis_title="Date",
    #             yaxis_title="Emissions (µmol m² sec⁻²)" if display_type == 'flux' else 'Uncertainty (µmol m² sec⁻²)',
    #             margin=dict(l=50, r=30, t=40, b=40),
    #             paper_bgcolor='white',
    #             plot_bgcolor='rgba(240, 242, 245, 0.8)',
    #             xaxis=dict(
    #                 showgrid=True,
    #                 gridcolor='rgba(128, 128, 128, 0.2)',
    #                 gridwidth=1,
    #                 showline=True,
    #                 linecolor='rgba(128, 128, 128, 0.8)',
    #                 type='date'
    #             ),
    #             yaxis=dict(
    #                 showgrid=True,
    #                 gridcolor='rgba(128, 128, 128, 0.2)',
    #                 gridwidth=1,
    #                 showline=True,
    #                 linecolor='rgba(128, 128, 128, 0.8)'
    #             ),
    #             showlegend=True,
    #             legend=dict(
    #                 x=0.01,
    #                 y=0.99,
    #                 xanchor='left',
    #                 yanchor='top',
    #                 bgcolor='rgba(255, 255, 255, 0.8)',
    #                 bordercolor='rgba(128, 128, 128, 0.2)',
    #                 borderwidth=1
    #             )
    #         )
            
    #         # Apply layouts
    #         map_fig.update_layout(map_layout)
    #         timeseries_fig.update_layout(timeseries_layout)
            
    #         # Handle time series display
    #         if not click_data:
    #             if spatial_agg == 'native':
    #                 # Show domain-wide average for native resolution
    #                 domain_means = np.nanmean(np.nanmean(data, axis=0), axis=0)
    #                 dates = pd.to_datetime(DATES, unit='s')
    #                 df = pd.DataFrame({'date': dates, 'value': domain_means})
                    
    #                 # Apply monthly aggregation only if selected
    #                 if temporal_agg != 'single':
    #                     df = df.set_index('date').resample('ME').mean().reset_index()
                    
    #                 timeseries_fig.add_trace(go.Scatter(
    #                     x=df['date'],
    #                     y=df['value'],
    #                     mode='lines+markers',
    #                     name='Domain Average',
    #                     line=dict(color='rgba(0,0,255,0.7)')
    #                 ))
    #                 timeseries_fig.update_layout(title="Domain-wide Average Time Series")
    #             else:
    #                 # Show average across all regions for aggregated data
    #                 dates = pd.to_datetime(df.columns)
    #                 region_mean = df.mean()
                    
    #                 # Create DataFrame for time series
    #                 temp_df = pd.DataFrame({'date': dates, 'value': region_mean})
                    
    #                 # Apply monthly aggregation only if selected
    #                 if temporal_agg != 'single':
    #                     temp_df = temp_df.set_index('date').resample('ME').mean().reset_index()
                    
    #                 timeseries_fig.add_trace(go.Scatter(
    #                     x=temp_df['date'],
    #                     y=temp_df['value'],
    #                     mode='lines+markers',
    #                     name='Regional Average',
    #                     line=dict(color='rgba(0,0,255,0.7)')
    #                 ))
    #                 timeseries_fig.update_layout(title="Regional Average Time Series")
    #         else:
    #             if spatial_agg == 'native':
    #                 # Handle click for native resolution
    #                 location_id = click_data['points'][0]['location']
    #                 i, j = map(int, location_id.split('_'))
                    
    #                 # Get full time series for clicked location
    #                 time_series = data[i, j, :]
    #                 dates = pd.to_datetime(DATES, unit='s')
    #                 df = pd.DataFrame({'date': dates, 'value': time_series})
                    
    #                 # Apply monthly aggregation only if selected
    #                 if temporal_agg != 'single':
    #                     df = df.set_index('date').resample('ME').mean().reset_index()
                    
    #                 title = f"{display_type.capitalize()} Time Series for Latitude {LAT_GRID[i,j]:.2f}°N, Longitude {LON_GRID[i,j]:.2f}°W"
    #             else:
    #                 # Handle click for aggregated data
    #                 location_id = click_data['points'][0]['location']
    #                 if spatial_agg == 'custom':
    #                     location_id = int(location_id)
                    
    #                 # Get full time series for clicked location
    #                 time_series = df.loc[location_id]
    #                 dates = pd.to_datetime(df.columns)
    #                 df_temp = pd.DataFrame({'date': dates, 'value': time_series.values})
                    
    #                 # Apply monthly aggregation only if selected
    #                 if temporal_agg != 'single':
    #                     df_temp = df_temp.set_index('date').resample('ME').mean().reset_index()
                    
    #                 df = df_temp  # Reassign for consistent plotting
    #                 # Create title based on aggregation type and display type
    #                 if spatial_agg == 'zip':
    #                     title = f"{display_type.capitalize()} Time Series for Zipcode {location_id}"
    #                 elif spatial_agg == 'census':
    #                     title = f"{display_type.capitalize()} Time Series for Census Tract {location_id}"
    #                 else:  # custom regions
    #                     title = f"{display_type.capitalize()} Time Series for Region {location_id}"
                
    #             timeseries_fig.add_trace(go.Scatter(
    #                 x=df['date'],
    #                 y=df['value'],
    #                 mode='lines+markers',
    #                 name='Selected Region',
    #                 line=dict(color='rgba(255,0,0,0.7)')
    #             ))
    #             timeseries_fig.update_layout(title=title)
            
    #         # Update map view state if necessary
    #         if relayout_data and 'mapbox.center' in relayout_data:
    #             map_layout['mapbox'].update(
    #                 center=relayout_data['mapbox.center'],
    #                 zoom=relayout_data['mapbox.zoom']
    #             )
    #         elif map_state:
    #             map_layout['mapbox'].update(map_state)
            
    #         # Final layout updates
    #         map_fig.update_layout(map_layout)
            
    #         return map_fig, timeseries_fig, map_layout['mapbox']
    
    #     except Exception as e:
    #         print(f"Error in dashboard update: {e}")
    #         return map_fig, timeseries_fig, None
                
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
    