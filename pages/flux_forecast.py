# -*- coding: utf-8 -*-
"""
Modified Emissions Dashboard for ABM-based emissions data
Displays census tract level emissions with temporal analysis
"""

from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import dash
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
import geopandas as gpd
import math
import tempfile
import zipfile
import gcsfs
import io
from utils import constants as prm

import logging
logging.getLogger("fiona").disabled = True

# Global variables
EMISSIONS_GDF = None
available_years = list(range(2025, 2055))

def init():
    """Initialize emissions dashboard components"""
    global EMISSIONS_GDF
    
    try:
        # Load the merged shapefile with emissions data
        shapefile = 'census_tracts_emissions_dashboard.shp'
        EMISSIONS_GDF = gpd.read_file(prm.SHAPEFILES['census_tracts_emissions_dashboard'])
        
        print("Successfully loaded emissions shapefile")
        print(f"Features loaded: {len(EMISSIONS_GDF)}")
        print(f"Available columns: {list(EMISSIONS_GDF.columns)}")
        
        return True
        
    except Exception as e:
        print(f"Error loading emissions data: {e}")
        return False

def get_column_name(display_variable, selected_year=None):
    """Map display variable to shapefile column name"""
    
    if display_variable == 'absolute':
        return f'abs_{selected_year}'
    elif display_variable == 'per_capita':
        return f'pc_{selected_year}'
    elif display_variable == 'median_income':
        return 'med_income'
    elif display_variable == 'households':
        return 'households'
    elif display_variable == 'percent_reduction':
        return 'pct_reduct'  # Updated to match actual shapefile column
    elif display_variable == 'pc_percent_reduction':
        return 'pc_pct_red'

def get_time_series_data(tract_data, display_variable):
    """Get time series for selected tract and variable"""
    
    years = available_years
    
    if display_variable == 'absolute':
        return [tract_data[f'abs_{year}'].iloc[0] for year in years]
    elif display_variable == 'per_capita':
        return [tract_data[f'pc_{year}'].iloc[0] for year in years]
    else:
        # Static variables - return constant value across years
        value = tract_data[get_column_name(display_variable)].iloc[0]
        return [value] * len(years)

def get_variable_info(display_variable):
    """Get display info for different variables"""
    
    variable_info = {
        'absolute': {
            'title': 'Absolute CO₂ Emissions',
            'unit': 't CO₂e/yr',
            'colorbar_title': 'Emissions<br>(t CO₂e/yr)',
            'time_varying': True
        },
        'per_capita': {
            'title': 'Per Capita CO₂ Emissions', 
            'unit': 't CO₂e/household/yr',
            'colorbar_title': 'Per Capita<br>(t CO₂e/hh/yr)',
            'time_varying': True
        },
        'median_income': {
            'title': 'Median Household Income',
            'unit': 'USD',
            'colorbar_title': 'Income<br>(USD)',
            'time_varying': False
        },
        'households': {
            'title': 'Number of Households',
            'unit': 'households',
            'colorbar_title': 'Households<br>(count)',
            'time_varying': False
        },
        'percent_reduction': {
            'title': 'Absolute Emissions Reduction (2025-2054)',
            'unit': '%',
            'colorbar_title': 'Reduction<br>(%)',
            'time_varying': False
        },
        'pc_percent_reduction': {
            'title': 'Per Capita Emissions Reduction (2025-2054)',
            'unit': '%',
            'colorbar_title': 'PC Reduction<br>(%)',
            'time_varying': False
        }
    }
    
    return variable_info.get(display_variable, variable_info['absolute'])

def create_header_card_emissions():
    """Create the header card with text followed by icons"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className="fas fa-chart-area me-2"),
                            "Agent Based Model (ABM) Emissions Analysis",
                            html.I(className="fas fa-chevron-down ms-2", id="emissions-header-chevron"),
                        ],
                        color="link",
                        id="emissions-header-toggle",
                        className="text-primary fs-4 text-decoration-none p-0 w-100 text-start",
                        style={"box-shadow": "none"}
                    )
                ], width=10),
                dbc.Col([
                    html.Div([
                        # Modified Documentation Button to link to a local PDF
                        html.A(
                            dbc.Button([
                                "Documentation ",
                                html.I(className="fas fa-file-pdf")
                            ], color="secondary", className="me-2"),
                            href="/assets/agent_based_model.pdf",  # <-- CHANGE THIS FILENAME
                            target="_blank"  # Opens the PDF in a new tab
                        ),
                
                        # Unchanged Help Button
                        dbc.Button([
                            "Help ",
                            html.I(className="fas fa-question-circle")
                        ], color="secondary", className="me-2", id="emissions-help-button"),
                
                        # Unchanged Reset Button
                        dbc.Button([
                            "Reset ",
                            html.I(className="fas fa-redo")
                        ], color="secondary", id="emissions-restart-button")
                
                    ], className="d-flex justify-content-end")
                ], width=2)
            ], className="align-items-center")
        ]),
        dbc.Collapse(
            dbc.CardBody([
                html.P([
                    "Explore simulated household CO₂ emissions across Los Angeles County census tracts (2025-2054).",
                    " Analyze spatial patterns, temporal trends, and socioeconomic relationships."
                ], className="text-muted mb-0")
            ]),
            id="emissions-header-collapse",
            is_open=True,
        )
    ], className="mb-4 shadow-sm")

def create_help_modal_emissions():
    """Create the help documentation modal"""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("ABM Emissions Analysis Help")),
            dbc.ModalBody([
                html.H5("Overview"),
                html.P("This dashboard visualizes simulated household CO₂ emissions across Los Angeles County census tracts from 2025-2054, allowing you to explore both spatial distributions and temporal patterns."),
            
                html.H5("Analysis Controls"),
                html.H6("Basic Controls:"),
                html.Ul([
                    html.Li([
                        html.Strong("Display Variable: "),
                        "Choose between absolute emissions, per capita emissions, or socioeconomic variables."
                    ]),
                    html.Li([
                        html.Strong("Year: "),
                        "Select specific year for spatial analysis (disabled for static variables)."
                    ]),
                    html.Li([
                        html.Strong("Color Scale: "),
                        "Toggle between linear and logarithmic color scales for emissions data."
                    ])
                ]),
            
                html.H5("Map Visualization"),
                html.Ul([
                    html.Li([
                        html.Strong("Navigation: "),
                        "Pan by dragging, zoom with mouse wheel. Double-click to reset view."
                    ]),
                    html.Li([
                        html.Strong("Tract Selection: "),
                        "Click on census tracts to see their temporal patterns in the time series plot."
                    ]),
                    html.Li([
                        html.Strong("Color Coding: "),
                        "Darker colors indicate higher emissions or values for the selected variable."
                    ])
                ]),
            
                html.H5("Time Series Plot"),
                html.Ul([
                    html.Li([
                        html.Strong("County-wide View: "),
                        "When no tract is selected, shows county-wide statistics with percentile bands."
                    ]),
                    html.Li([
                        html.Strong("Individual Tract: "),
                        "Click a tract on the map to see its trajectory compared to county distribution."
                    ]),
                    html.Li([
                        html.Strong("Static Variables: "),
                        "For non-time-varying data, displays distribution histograms instead."
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
                        "Download current view as CSV or complete shapefile as ZIP."
                    ]),
                    html.Li([
                        html.Strong("Reset: "),
                        "Use the reset button to restore all controls to their default values."
                    ])
                ])
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="emissions-help-close", className="ms-auto")
            ),
        ],
        id="emissions-help-modal",
        size="lg",
        is_open=False,
    )

def create_expansion_controls():
    """Create the expansion control column"""
    return html.Div([
        html.Button("→", id="emissions-expand-left", className="expand-button"),
        html.Button("←", id="emissions-expand-right", className="expand-button")
    ], 
    style={'width': '30px'}, 
    className="d-flex flex-column justify-content-center"
    )

def create_restore_button():
    """Create the restore button"""
    return html.Button(
        "Restore Panels", 
        id="emissions-restore-button", 
        className="restore-button",
        style={'display': 'none'}
    )

def get_control_panel():
    """Return the control panel layout with collapsible controls."""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Button(
                [
                    html.I(className="fas fa-sliders-h me-2"),
                    "Analysis Controls",
                    html.I(className="fas fa-chevron-down ms-2", id="emissions-controls-chevron"),
                ],
                color="link",
                id="emissions-controls-toggle",
                className="text-primary fs-4 text-decoration-none p-0 w-100 text-start",
                style={"box-shadow": "none"}
            )
        ]),
        dbc.Collapse(
            dbc.CardBody([
                # First row with existing dropdowns
                dbc.Row([
                    # Variable Selection
                    dbc.Col([
                        html.Label("Display Variable", className="form-label fw-bold"),
                        dcc.Dropdown(
                            id='emissions-display-variable',
                            options=[
                                {'label': 'Absolute CO₂ Emissions', 'value': 'absolute'},
                                {'label': 'Per Capita CO₂ Emissions', 'value': 'per_capita'},
                                {'label': 'Median Household Income', 'value': 'median_income'},
                                {'label': 'Number of Households', 'value': 'households'},
                                {'label': 'Absolute % Reduction (2025-2054)', 'value': 'percent_reduction'},
                                {'label': 'Per Capita % Reduction (2025-2054)', 'value': 'pc_percent_reduction'}
                            ],
                            value='absolute',
                            clearable=False,
                            className="mb-2"
                        )
                    ], md=3),
                    
                    # Year Selection
                    dbc.Col([
                        html.Label("Year", className="form-label fw-bold"),
                        dcc.Dropdown(
                            id='emissions-year-select',
                            options=[
                                {'label': str(year), 'value': year}
                                for year in available_years
                            ],
                            value=2025,
                            clearable=False,
                            className="mb-2"
                        )
                    ], md=2),
                    
                    # Time Series Focus Selection
                    dbc.Col([
                        html.Label("Time Series Focus", className="form-label fw-bold"),
                        dcc.Dropdown(
                            id='emissions-timeseries-focus',
                            options=[
                                {'label': 'County-wide View', 'value': 'county'},
                                {'label': 'Census Tract View', 'value': 'tract'}
                            ],
                            value='county',
                            clearable=False,
                            className="mb-2"
                        )
                    ], md=2),
                    
                    # Original Color Scale Selection (keeping for backward compatibility)
                    dbc.Col([
                        html.Label("Color Scale", className="form-label fw-bold"),
                        dcc.Dropdown(
                            id='emissions-color-scale',
                            options=[
                                {'label': 'Linear Scale', 'value': 'linear'},
                                {'label': 'Log Scale', 'value': 'log'}
                            ],
                            value='linear',
                            clearable=False,
                            className="mb-2"
                        )
                    ], md=5),
                ]),
                
                # Second row with new map visualization controls
                dbc.Row([
                    # Color Scale Type
                    dbc.Col([
                        html.Label("Color Scale Type", className="form-label fw-bold"),
                        dcc.Dropdown(
                            id='emissions-color-scale-type',
                            options=[
                                {'label': 'Auto', 'value': 'auto'},
                                {'label': 'Fixed', 'value': 'fixed'}
                            ],
                            value='auto',
                            clearable=False,
                            className="mb-2"
                        )
                    ], md=3),
                    
                    # Min Scale
                    dbc.Col([
                        html.Label("Min Scale", className="form-label fw-bold"),
                        dbc.Input(
                            id='emissions-scale-min',
                            type='number',
                            placeholder='Min Value',  # UPDATED PLACEHOLDER
                            step=0.1,
                            disabled=True,  # Initially disabled since default is 'auto'
                            className="mb-2",
                            style={'height': '38px'}  # Match dropdown height
                        )
                    ], md=3),
                    
                    # Max Scale
                    dbc.Col([
                        html.Label("Max Scale", className="form-label fw-bold"),
                        dbc.Input(
                            id='emissions-scale-max',
                            type='number',
                            placeholder='Max Value',  # UPDATED PLACEHOLDER
                            step=0.1,
                            disabled=True,  # Initially disabled since default is 'auto'
                            className="mb-2",
                            style={'height': '38px'}  # Match dropdown height
                        )
                    ], md=3),
                    
                    # Transparency
                    dbc.Col([
                        html.Label("Transparency", className="form-label fw-bold"),
                        dbc.Input(
                            id='emissions-transparency',
                            type='number',
                            min=0,
                            max=1,
                            step=0.1,
                            value=0.7,
                            placeholder='0.0 - 1.0',
                            className="mb-2",
                            style={'height': '38px'}  # Match dropdown height
                        )
                    ], md=3)
                ], className="mt-3")  # Add top margin to separate from first row
            ]),
            id="emissions-controls-collapse",
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
                                    "Spatial Distribution",
                                    html.I(className="fas fa-chevron-down ms-2", id="emissions-map-chevron"),
                                ],
                                color="link",
                                id="emissions-map-toggle",
                                className="text-primary fs-4 text-decoration-none p-0 w-100 text-start",
                                style={"box-shadow": "none"}
                            )
                        ], width=10),
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.DropdownMenu(
                                    [
                                        dbc.DropdownMenuItem("Current View (CSV)", 
                                            id="emissions-trigger-download-current-csv"),
                                        dbc.DropdownMenuItem(divider=True),
                                        dbc.DropdownMenuItem("Full Shapefile (ZIP)", 
                                            id="emissions-trigger-download-shapefile-zip"),
                                    ],
                                    label=[html.I(className="fas fa-download me-1"), "Data"],
                                    color="secondary",
                                    size="sm"
                                )
                            ])
                        ], width=2)
                    ], className="align-items-center")
                ]),
                dbc.Collapse(
                    dbc.CardBody([
                        dcc.Graph(
                            id='emissions-map',
                            config={
                                'modeBarButtonsToRemove': ['autoScale2d', 'select2d', 'lasso2d'],
                                'displaylogo': False,
                                'scrollZoom': True
                            },
                            style={'height': '65vh'}
                        )
                    ]),
                    id="emissions-map-collapse",
                    is_open=True,
                )
            ], className="shadow-sm")
        ], id="emissions-left-panel", className="panel-transition flex-grow-1", style={'flex': '1', 'minWidth': '0'}),

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
                                    "Temporal Evolution",
                                    html.I(className="fas fa-chevron-down ms-2", id="emissions-timeseries-chevron"),
                                ],
                                color="link",
                                id="emissions-timeseries-toggle",
                                className="text-primary fs-4 text-decoration-none p-0 w-100 text-start",
                                style={"box-shadow": "none"}
                            )
                        ], width=10),
                        dbc.Col([
                            html.Div([
                                dbc.ButtonGroup([
                                    dbc.Button(
                                        [html.I(className="fas fa-download me-1"), "DATA"],
                                        id="emissions-trigger-download-timeseries-csv",
                                        color="secondary",
                                        size="sm"
                                    )
                                ])
                            ], className="d-flex justify-content-end")
                        ], width=2)
                    ], className="align-items-center")
                ]),
                dbc.Collapse(
                    dbc.CardBody([
                        dcc.Graph(
                            id='emissions-timeseries',
                            style={'height': '65vh'}
                        )
                    ]),
                    id="emissions-timeseries-collapse",
                    is_open=True,
                )
            ], className="shadow-sm")
        ], id="emissions-right-panel", className="panel-transition flex-grow-1", style={'flex': '1', 'minWidth': '0'})
    ], className="d-flex gap-2", style={'height': '65vh'})

def get_layout():
    """Return the complete page layout."""
    return dbc.Container([
        # Hidden stores

        dcc.Store(id='emissions-map-state-store'),
        dcc.Store(id='emissions-expansion-state', data='none'),
        dcc.Store(id='emissions-selected-tract', data=None),  # NEW: Store for selected tract
        
        # Download components
        html.Div([
            dcc.Download(id="emissions-download-current-csv"),
            dcc.Download(id="emissions-download-shapefile-zip"),
            dcc.Download(id="emissions-download-timeseries-csv"),
        ]),
   
        # Layout components
        create_header_card_emissions(),
        get_control_panel(),
        get_main_content(),
        create_restore_button(),
        create_help_modal_emissions()
    ], fluid=True, className="px-4 py-3")

def register_callbacks(app):
    """Register callbacks for the emissions dashboard"""    
    
    # Header collapse callback
    @app.callback(
        [Output("emissions-header-collapse", "is_open"),
         Output("emissions-header-chevron", "className")],
        [Input("emissions-header-toggle", "n_clicks")],
        [State("emissions-header-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_header_collapse(n_clicks, is_open):
        """Toggle the header section collapse state"""
        if n_clicks:
            return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
        return is_open, "fas fa-chevron-down ms-2"

    # Controls collapse callback
    @app.callback(
        [Output("emissions-controls-collapse", "is_open"),
         Output("emissions-controls-chevron", "className")],
        [Input("emissions-controls-toggle", "n_clicks")],
        [State("emissions-controls-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_controls_collapse(n_clicks, is_open):
        """Toggle the controls section collapse state"""
        if n_clicks:
            return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
        return is_open, "fas fa-chevron-down ms-2"

    # Map collapse callback
    @app.callback(
        [Output("emissions-map-collapse", "is_open"),
         Output("emissions-map-chevron", "className")],
        [Input("emissions-map-toggle", "n_clicks")],
        [State("emissions-map-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_map_collapse(n_clicks, is_open):
        """Toggle the map section collapse state"""
        if n_clicks:
            return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
        return is_open, "fas fa-chevron-down ms-2"
    # 4a. Scale Range Toggle Callback
    @app.callback(
    [Output('emissions-scale-min', 'disabled'),
     Output('emissions-scale-max', 'disabled'),
     Output('emissions-scale-min', 'value', allow_duplicate=True),      # NEW OUTPUT
     Output('emissions-scale-max', 'value', allow_duplicate=True)],     # NEW OUTPUT
    Input('emissions-color-scale-type', 'value'),
    prevent_initial_call=True
    )
    def toggle_scale_inputs(color_scale_type):
        """Enable/disable Min/Max inputs and clear values when Auto is selected"""
        if color_scale_type == 'auto':
            # Disable inputs and clear values when Auto is selected
            return True, True, None, None
        else:
            # Enable inputs when Fixed is selected, keep existing values
            return False, False, dash.no_update, dash.no_update
    
    @app.callback(
    [Output('emissions-color-scale-type', 'value', allow_duplicate=True),
     Output('emissions-scale-min', 'value', allow_duplicate=True),
     Output('emissions-scale-max', 'value', allow_duplicate=True)],
    Input('emissions-color-scale', 'value'),
    State('emissions-color-scale-type', 'value'),
    prevent_initial_call=True
)
    def reset_scale_on_linear_log_change(color_scale, current_scale_type):
        """Reset color scale type to Auto and clear values when switching between linear/log"""
        # Always reset to Auto when linear/log scale changes
        # This prevents confusion with inappropriate fixed ranges
        return 'auto', None, None
    
    
    # Timeseries collapse callback
    @app.callback(
        [Output("emissions-timeseries-collapse", "is_open"),
         Output("emissions-timeseries-chevron", "className")],
        [Input("emissions-timeseries-toggle", "n_clicks")],
        [State("emissions-timeseries-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_timeseries_collapse(n_clicks, is_open):
        """Toggle the timeseries section collapse state"""
        if n_clicks:
            return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
        return is_open, "fas fa-chevron-down ms-2"

    # Help modal callback
    @app.callback(
        Output("emissions-help-modal", "is_open", allow_duplicate=True),
        [Input("emissions-help-button", "n_clicks"),
         Input("emissions-help-close", "n_clicks")],
        [State("emissions-help-modal", "is_open")],
        prevent_initial_call=True
    )
    def toggle_help_modal(help_clicks, close_clicks, is_open):
        """Toggle the help modal visibility"""
        ctx = dash.callback_context
        if not ctx.triggered:
            return is_open
    
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == "emissions-help-button":
            return not is_open
        elif button_id == "emissions-help-close":
            return False
        return is_open


    # 4b. Transparency Validation Callback
    @app.callback(
        Output('emissions-transparency', 'value', allow_duplicate=True),
        Input('emissions-transparency', 'value'),
        prevent_initial_call=True
    )
    def validate_transparency(transparency_value):
        """Validate transparency input and reset to 0.7 if out of range"""
        if transparency_value is None:
            return 0.7
        
        try:
            # Convert to float and validate range
            transparency_float = float(transparency_value)
            
            # Check if value is within valid range (0.0 to 1.0)
            if transparency_float < 0.0 or transparency_float > 1.0:
                return 0.7  # Reset to default if out of range
            
            return transparency_float  # Return valid value
            
        except (ValueError, TypeError):
            # If conversion fails, return default
            return 0.7


    # Panel expansion callbacks
    @app.callback(
    [Output('emissions-left-panel', 'style'),
     Output('emissions-right-panel', 'style'),
     Output('emissions-restore-button', 'style'),
     Output('emissions-expansion-state', 'data')],
    [Input('emissions-expand-left', 'n_clicks'),
     Input('emissions-expand-right', 'n_clicks'),
     Input('emissions-restore-button', 'n_clicks')],
    [State('emissions-expansion-state', 'data')],
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
        
        if button_id == 'emissions-restore-button':
            return (
                {'flex': '1', **base_style},
                {'flex': '1', **base_style},
                {'display': 'none'},
                'none'
            )
        elif button_id == 'emissions-expand-left':
            if current_state != 'left':
                return (
                    {'flex': '1', **base_style},
                    {'display': 'none'},  # Hide right panel completely
                    {'display': 'block', 'position': 'fixed', 'bottom': '20px', 'right': '20px'},
                    'left'
                )
        elif button_id == 'emissions-expand-right':
            if current_state != 'right':
                return (
                    {'display': 'none'},  # Hide left panel completely
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

    @app.callback(
        Output('emissions-year-select', 'disabled'),
        Input('emissions-display-variable', 'value')
    )
    def toggle_year_selector(display_variable):
        """Disable year selector for static variables"""
        var_info = get_variable_info(display_variable)
        return not var_info['time_varying']
    
    @app.callback(
    [Output('emissions-map', 'figure'),
     Output('emissions-timeseries', 'figure'),
     Output('emissions-map-state-store', 'data', allow_duplicate=True),
     Output('emissions-selected-tract', 'data', allow_duplicate=True)],
    [Input('emissions-display-variable', 'value'),
     Input('emissions-year-select', 'value'),
     Input('emissions-color-scale', 'value'),
     Input('emissions-timeseries-focus', 'value'),
     Input('emissions-color-scale-type', 'value'),  # NEW INPUT
     Input('emissions-scale-min', 'value'),         # NEW INPUT
     Input('emissions-scale-max', 'value'),         # NEW INPUT
     Input('emissions-transparency', 'value'),      # NEW INPUT
     Input('emissions-map', 'clickData'),
     Input('emissions-restart-button', 'n_clicks')],
    [State('emissions-map', 'relayoutData'),
     State('emissions-map-state-store', 'data'),
     State('emissions-selected-tract', 'data')],
    prevent_initial_call='initial_duplicate'
)
    def update_dashboard(display_variable, selected_year, color_scale, timeseries_focus, 
                        color_scale_type, scale_min, scale_max, transparency,  # NEW PARAMETERS
                        click_data, restart_clicks, relayout_data, map_state, selected_tract):
        
        ctx = dash.callback_context
        if ctx.triggered:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger_id == 'emissions-restart-button':
                # Reset everything on restart
                click_data = None
                selected_tract = None
            elif trigger_id == 'emissions-timeseries-focus':
                # Reset selected tract when focus changes
                selected_tract = None
            elif trigger_id == 'emissions-map' and timeseries_focus == 'tract' and click_data:
                # Only update selected tract if in tract view mode
                selected_tract = click_data['points'][0]['location']
       
        # Initialize figures
        map_fig = go.Figure()
        timeseries_fig = go.Figure()
        
        # Validate and set default values for new parameters
        if transparency is None or transparency < 0 or transparency > 1:
            transparency = 0.7
        
        try:
            # Get variable information
            var_info = get_variable_info(display_variable)
            
            # Get column name for current selection
            column_name = get_column_name(display_variable, selected_year)
            
            # Check if column exists
            if column_name not in EMISSIONS_GDF.columns:
                print(f"Warning: Column {column_name} not found")
                return map_fig, timeseries_fig, None, None
            
            # Get data for mapping
            current_data = EMISSIONS_GDF[column_name]
            
            # Set up color scale - only apply log to emissions variables
            if (color_scale == 'log' and 
                display_variable in ['absolute', 'per_capita'] and 
                (current_data > 0).all()):
                # Use log scale only for emissions data with all positive values
                color_params = dict(
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title=f"Log10({var_info['colorbar_title']})",
                        orientation='v',
                        thickness=20,
                        len=0.9,
                        y=0.5,
                        yanchor='middle',
                        x=1.02,
                        xanchor='left',
                        tickfont=dict(size=10)
                    )
                )
                # Add log transformation
                map_data = np.log10(current_data.replace(0, np.nan))
            else:
                # Use linear scale for all other cases
                color_params = dict(
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title=var_info['colorbar_title'],
                        orientation='v',
                        thickness=20,
                        len=0.9,
                        y=0.5,
                        yanchor='middle',
                        x=1.02,
                        xanchor='left',
                        tickfont=dict(size=10)
                    )
                )
                map_data = current_data
            
            # NEW: Add fixed scale parameters if color_scale_type is 'fixed'
            if color_scale_type == 'fixed' and scale_min is not None and scale_max is not None:
                if color_scale == 'log' and display_variable in ['absolute', 'per_capita']:
                    # For log scale, apply log to the min/max values
                    color_params['zmin'] = np.log10(max(scale_min, 0.001))  # Avoid log(0)
                    color_params['zmax'] = np.log10(max(scale_max, 0.001))
                else:
                    # For linear scale, use values directly
                    color_params['zmin'] = float(scale_min)
                    color_params['zmax'] = float(scale_max)
            
            # Create choropleth map with transparency
            map_fig.add_trace(go.Choroplethmapbox(
                geojson=EMISSIONS_GDF.__geo_interface__,
                featureidkey="properties.GEOID",
                locations=EMISSIONS_GDF['GEOID'],
                z=map_data,
                marker_opacity=transparency,  # NEW: Use dynamic transparency instead of hardcoded 0.7
                hovertemplate=(
                    f"<b>Census Tract: %{{location}}</b><br>"
                    f"{var_info['title']}: %{{z:.3f}} {var_info['unit']}<br>"
                    "<extra></extra>"
                ),
                **color_params
            ))
            
            # Calculate map bounds
            bounds = EMISSIONS_GDF.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            
            # Calculate zoom level
            lon_range = bounds[2] - bounds[0]
            lat_range = bounds[3] - bounds[1] 
            zoom = min(
                math.log2(360 / lon_range) - 1,
                math.log2(180 / lat_range) - 1
            )
            zoom = max(min(zoom, 10), 7)
            
            # Map layout
            map_layout = dict(
                autosize=True,
                mapbox=dict(
                    style='open-street-map',
                    zoom=zoom,
                    center=dict(lat=center_lat, lon=center_lon),
                ),
                margin=dict(l=0, r=70, t=40, b=0),
                title=f"{var_info['title']}" + (f" ({selected_year})" if var_info['time_varying'] else ""),
                uirevision='constant',
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
            
            # Time series layout (unchanged)
            timeseries_layout = dict(
                autosize=True,
                xaxis_title="Year",
                yaxis_title=f"{var_info['title']} ({var_info['unit']})",
                margin=dict(l=50, r=30, t=40, b=40),
                paper_bgcolor='white',
                plot_bgcolor='rgba(240, 242, 245, 0.8)',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    gridwidth=1,
                    showline=True,
                    linecolor='rgba(128, 128, 128, 0.8)'
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
                    x=0.98,
                    y=0.98,
                    xanchor='right',
                    yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='rgba(128, 128, 128, 0.3)',
                    borderwidth=1
                )
            )
            
            # Time series logic (unchanged - keeping existing implementation)
            if timeseries_focus == 'county':
                # County-wide view logic...
                if var_info['time_varying']:
                    # Calculate statistics across all tracts for each year
                    median_data = []
                    q25_data = []
                    q75_data = []
                    q10_data = []
                    q90_data = []
                    min_data = []
                    max_data = []
                    
                    for year in available_years:
                        col = get_column_name(display_variable, year)
                        if col in EMISSIONS_GDF.columns:
                            year_data = EMISSIONS_GDF[col]
                            median_data.append(year_data.median())
                            q25_data.append(year_data.quantile(0.25))
                            q75_data.append(year_data.quantile(0.75))
                            q10_data.append(year_data.quantile(0.10))
                            q90_data.append(year_data.quantile(0.90))
                            min_data.append(year_data.min())
                            max_data.append(year_data.max())
                        else:
                            median_data.append(np.nan)
                            q25_data.append(np.nan)
                            q75_data.append(np.nan)
                            q10_data.append(np.nan)
                            q90_data.append(np.nan)
                            min_data.append(np.nan)
                            max_data.append(np.nan)
                    
                    # Add percentile ranges and lines
                    timeseries_fig.add_trace(go.Scatter(
                        x=available_years + available_years[::-1],
                        y=q90_data + q10_data[::-1],
                        fill='toself',
                        fillcolor='rgba(100, 150, 200, 0.3)',
                        line=dict(color='rgba(100, 150, 200, 0.5)', width=0.5),
                        hoverinfo="skip",
                        showlegend=True,
                        name='10th-90th Percentile Range'
                    ))
                    
                    timeseries_fig.add_trace(go.Scatter(
                        x=available_years + available_years[::-1],
                        y=q75_data + q25_data[::-1],
                        fill='toself',
                        fillcolor='rgba(100, 150, 200, 0.6)',
                        line=dict(color='rgba(100, 150, 200, 0.8)', width=0.5),
                        hoverinfo="skip",
                        showlegend=True,
                        name='Interquartile Range (25-75%)'
                    ))
                    
                    timeseries_fig.add_trace(go.Scatter(
                        x=available_years,
                        y=median_data,
                        mode='lines',
                        name='Median Emissions',
                        line=dict(color='#b30000', width=3)
                    ))
                    
                    timeseries_fig.add_trace(go.Scatter(
                        x=available_years,
                        y=max_data,
                        mode='lines',
                        name='Maximum',
                        line=dict(color='#d62728', width=1.5, dash='longdash'),
                        showlegend=True
                    ))
                    
                    timeseries_fig.add_trace(go.Scatter(
                        x=available_years,
                        y=min_data,
                        mode='lines',
                        name='Minimum',
                        line=dict(color='#1f77b4', width=1.5, dash='longdash'),
                        showlegend=True
                    ))
                    
                    timeseries_fig.update_layout(title="County-wide Emission Trajectory & Variability")
                else:
                    # For static variables, show histogram
                    timeseries_fig.add_trace(go.Histogram(
                        x=current_data,
                        nbinsx=30,
                        name='Distribution',
                        marker=dict(color='rgba(0,100,200,0.7)')
                    ))
                    timeseries_fig.update_layout(
                        title=f"County-wide Distribution of {var_info['title']}",
                        xaxis_title=f"{var_info['title']} ({var_info['unit']})",
                        yaxis_title="Number of Census Tracts"
                    )
            
            else:  # timeseries_focus == 'tract'
                # Tract view logic...
                if not selected_tract:
                    # Show instruction with background
                    if var_info['time_varying']:
                        # Calculate county background statistics
                        median_data = []
                        q25_data = []
                        q75_data = []
                        q10_data = []
                        q90_data = []
                        
                        for year in available_years:
                            col = get_column_name(display_variable, year)
                            if col in EMISSIONS_GDF.columns:
                                year_data = EMISSIONS_GDF[col]
                                median_data.append(year_data.median())
                                q25_data.append(year_data.quantile(0.25))
                                q75_data.append(year_data.quantile(0.75))
                                q10_data.append(year_data.quantile(0.10))
                                q90_data.append(year_data.quantile(0.90))
                            else:
                                median_data.append(np.nan)
                                q25_data.append(np.nan)
                                q75_data.append(np.nan)
                                q10_data.append(np.nan)
                                q90_data.append(np.nan)
                        
                        # Add faded background
                        timeseries_fig.add_trace(go.Scatter(
                            x=available_years + available_years[::-1],
                            y=q90_data + q10_data[::-1],
                            fill='toself',
                            fillcolor='rgba(200, 200, 200, 0.2)',
                            line=dict(color='rgba(200, 200, 200, 0.3)', width=0.5),
                            hoverinfo="skip",
                            showlegend=False,
                            name='County Range'
                        ))
                        
                        timeseries_fig.add_trace(go.Scatter(
                            x=available_years,
                            y=median_data,
                            mode='lines',
                            name='County Median',
                            line=dict(color='rgba(150,150,150,0.5)', width=2, dash='dash')
                        ))
                        
                        timeseries_fig.update_layout(title="Click a census tract on the map to view its time series")
                    else:
                        # Show histogram with instruction
                        timeseries_fig.add_trace(go.Histogram(
                            x=current_data,
                            nbinsx=30,
                            name='All Tracts',
                            marker=dict(color='rgba(150,150,150,0.5)')
                        ))
                        timeseries_fig.update_layout(
                            title="Click a census tract on the map to see its value",
                            xaxis_title=f"{var_info['title']} ({var_info['unit']})",
                            yaxis_title="Number of Census Tracts"
                        )
                else:
                    # Show selected tract vs county
                    tract_data = EMISSIONS_GDF[EMISSIONS_GDF['GEOID'] == selected_tract]
                    
                    if len(tract_data) > 0:
                        if var_info['time_varying']:
                            time_series_values = get_time_series_data(tract_data, display_variable)
                            
                            # Add county background bands
                            median_data = []
                            q25_data = []
                            q75_data = []
                            q10_data = []
                            q90_data = []
                            min_data = []
                            max_data = []
                            
                            for year in available_years:
                                col = get_column_name(display_variable, year)
                                if col in EMISSIONS_GDF.columns:
                                    year_data = EMISSIONS_GDF[col]
                                    median_data.append(year_data.median())
                                    q25_data.append(year_data.quantile(0.25))
                                    q75_data.append(year_data.quantile(0.75))
                                    q10_data.append(year_data.quantile(0.10))
                                    q90_data.append(year_data.quantile(0.90))
                                    min_data.append(year_data.min())
                                    max_data.append(year_data.max())
                                else:
                                    median_data.append(np.nan)
                                    q25_data.append(np.nan)
                                    q75_data.append(np.nan)
                                    q10_data.append(np.nan)
                                    q90_data.append(np.nan)
                                    min_data.append(np.nan)
                                    max_data.append(np.nan)
                            
                            # Add county background
                            timeseries_fig.add_trace(go.Scatter(
                                x=available_years + available_years[::-1],
                                y=q90_data + q10_data[::-1],
                                fill='toself',
                                fillcolor='rgba(150, 150, 150, 0.3)',
                                line=dict(color='rgba(150, 150, 150, 0.5)', width=0.5),
                                hoverinfo="skip",
                                showlegend=True,
                                name='County 10th-90th Percentile'
                            ))
                            
                            timeseries_fig.add_trace(go.Scatter(
                                x=available_years + available_years[::-1],
                                y=q75_data + q25_data[::-1],
                                fill='toself',
                                fillcolor='rgba(150, 150, 150, 0.6)',
                                line=dict(color='rgba(150, 150, 150, 0.8)', width=0.5),
                                hoverinfo="skip",
                                showlegend=True,
                                name='County Interquartile Range'
                            ))
                            
                            timeseries_fig.add_trace(go.Scatter(
                                x=available_years,
                                y=median_data,
                                mode='lines',
                                name='County Median',
                                line=dict(color='rgba(100,100,100,0.7)', width=2, dash='dash')
                            ))
                            
                            timeseries_fig.add_trace(go.Scatter(
                                x=available_years,
                                y=max_data,
                                mode='lines',
                                name='County Maximum',
                                line=dict(color='#d62728', width=1.5, dash='longdash'),
                                showlegend=True
                            ))
                            
                            timeseries_fig.add_trace(go.Scatter(
                                x=available_years,
                                y=min_data,
                                mode='lines',
                                name='County Minimum',
                                line=dict(color='#1f77b4', width=1.5, dash='longdash'),
                                showlegend=True
                            ))
                            
                            # Add selected tract line
                            timeseries_fig.add_trace(go.Scatter(
                                x=available_years,
                                y=time_series_values,
                                mode='lines+markers',
                                name=f'Tract {selected_tract}',
                                line=dict(color='rgba(200,50,50,0.9)', width=4),
                                marker=dict(size=8, color='rgba(200,50,50,0.9)')
                            ))
                            
                            timeseries_fig.update_layout(title=f"Tract {selected_tract} vs County Distribution")
                        else:
                            # For static variables, show tract value vs county distribution
                            tract_value = tract_data[column_name].iloc[0]
                            
                            timeseries_fig.add_trace(go.Histogram(
                                x=current_data,
                                nbinsx=30,
                                name='All Tracts',
                                marker=dict(color='rgba(100,100,100,0.5)')
                            ))
                            
                            timeseries_fig.add_vline(
                                x=tract_value,
                                line_dash="dash",
                                line_color="red",
                                line_width=3,
                                annotation_text=f"Tract {selected_tract}: {tract_value:.2f}"
                            )
                            
                            timeseries_fig.update_layout(
                                title=f"Tract {selected_tract} vs County Distribution",
                                xaxis_title=f"{var_info['title']} ({var_info['unit']})",
                                yaxis_title="Number of Census Tracts"
                            )
            
            # Apply layouts
            map_fig.update_layout(map_layout)
            timeseries_fig.update_layout(timeseries_layout)
            
            return map_fig, timeseries_fig, map_layout['mapbox'], selected_tract
            
        except Exception as e:
            print(f"Error in dashboard update: {e}")
            return map_fig, timeseries_fig, None, None
    
    # Reset callback
    # Reset callback (also needs to be updated)
    # 3. Updated Reset Callback
    @app.callback(
        [Output('emissions-display-variable', 'value', allow_duplicate=True),
         Output('emissions-year-select', 'value', allow_duplicate=True),
         Output('emissions-color-scale', 'value', allow_duplicate=True),
         Output('emissions-timeseries-focus', 'value', allow_duplicate=True),
         Output('emissions-color-scale-type', 'value', allow_duplicate=True),  # NEW OUTPUT
         Output('emissions-scale-min', 'value', allow_duplicate=True),         # NEW OUTPUT
         Output('emissions-scale-max', 'value', allow_duplicate=True),         # NEW OUTPUT
         Output('emissions-transparency', 'value', allow_duplicate=True),      # NEW OUTPUT
         Output('emissions-left-panel', 'style', allow_duplicate=True),
         Output('emissions-right-panel', 'style', allow_duplicate=True),
         Output('emissions-restore-button', 'style', allow_duplicate=True),
         Output('emissions-expansion-state', 'data', allow_duplicate=True),
         Output('emissions-header-collapse', 'is_open', allow_duplicate=True),
         Output('emissions-controls-collapse', 'is_open', allow_duplicate=True),
         Output('emissions-map-collapse', 'is_open', allow_duplicate=True),
         Output('emissions-timeseries-collapse', 'is_open', allow_duplicate=True),
         Output('emissions-map-state-store', 'data', allow_duplicate=True),
         Output('emissions-selected-tract', 'data', allow_duplicate=True)],
        [Input('emissions-restart-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def reset_to_initial_state(n_clicks):
        """Reset all controls to their initial state"""
        if not n_clicks:
            return dash.no_update
    
        # Default values
        display_variable = 'absolute'
        selected_year = 2025
        color_scale = 'linear'
        timeseries_focus = 'county'
        color_scale_type = 'auto'     # NEW DEFAULT
        scale_min = None              # NEW DEFAULT
        scale_max = None              # NEW DEFAULT
        transparency = 0.7            # NEW DEFAULT
    
        # Default panel styles
        base_style = {'flex': '1', 'minWidth': '0'}
    
        return (
            display_variable,       # display variable
            selected_year,         # year select
            color_scale,          # color scale
            timeseries_focus,     # timeseries focus
            color_scale_type,     # NEW: color scale type
            scale_min,            # NEW: scale min
            scale_max,            # NEW: scale max
            transparency,         # NEW: transparency
            base_style,           # left panel style
            base_style,           # right panel style
            {'display': 'none'},  # restore button style
            'none',              # expansion state
            True,               # header collapse
            True,               # controls collapse
            True,               # map collapse
            True,               # timeseries collapse
            None,               # Reset map state store
            None                # Reset selected tract
        )

        
    # Download callbacks
    @app.callback(
        Output("emissions-download-current-csv", "data"),
        Input("emissions-trigger-download-current-csv", "n_clicks"),
        [State('emissions-display-variable', 'value'),
         State('emissions-year-select', 'value')],
        prevent_initial_call=True
    )
    def download_current_csv(n_clicks, display_variable, selected_year):
        """Download current view as CSV"""
        if not n_clicks:
            return None
        
        try:
            var_info = get_variable_info(display_variable)
            column_name = get_column_name(display_variable, selected_year)
            
            # Create output dataframe
            output_df = EMISSIONS_GDF[['GEOID', column_name]].copy()
            output_df = output_df.rename(columns={column_name: f"{display_variable}_{selected_year if var_info['time_varying'] else 'value'}"})
            
            filename = f"emissions_{display_variable}_{selected_year if var_info['time_varying'] else 'static'}.csv"
            
            return dcc.send_data_frame(output_df.to_csv, filename, index=False)
        
        except Exception as e:
            print(f"Error in CSV download: {e}")
            return None
    
    @app.callback(
        Output("emissions-download-shapefile-zip", "data"),
        Input("emissions-trigger-download-shapefile-zip", "n_clicks"),
        prevent_initial_call=True
    )
    def download_shapefile_zip(n_clicks):
        """Download complete shapefile as ZIP from GCS."""
        if not n_clicks:
            return None
        
        try:
            fs = gcsfs.GCSFileSystem()
    
            with tempfile.TemporaryDirectory() as temp_dir:
                # 1. Define shapefile components and get the base GCS path from constants
                shapefile_exts = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
                shapefile_key = 'census_tracts_emissions_dashboard'
                # Reconstruct the base GCS path from the full .shp path in constants
                base_gcs_path = prm.SHAPEFILES[shapefile_key].replace('.shp', '')
    
                # 2. Download all shapefile components from GCS to the temporary directory
                for ext in shapefile_exts:
                    gcs_path = base_gcs_path + ext
                    local_path = os.path.join(temp_dir, f"{shapefile_key}{ext}")
                    if fs.exists(gcs_path):
                        fs.get(gcs_path, local_path)
                
                # 3. Zip the downloaded files
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Use a simpler name in the zip archive
                            arc_name = file.replace(shapefile_key, 'emissions_data')
                            zip_file.write(file_path, arc_name)
    
                zip_buffer.seek(0)
                
                return dcc.send_bytes(zip_buffer.getvalue(), 'emissions_shapefile.zip')
        
        except Exception as e:
            print(f"Error in shapefile download: {e}")
            return None
        
        
        
    
    @app.callback(
        Output("emissions-download-timeseries-csv", "data"),
        Input("emissions-trigger-download-timeseries-csv", "n_clicks"),
        [State('emissions-timeseries', 'figure')],
        prevent_initial_call=True
    )
    def download_timeseries_csv(n_clicks, figure):
        """Download time series data as CSV"""
        if not n_clicks or not figure:
            return None
        
        try:
            # Extract data from time series figure
            consolidated_df = pd.DataFrame()
            
            for trace in figure['data']:
                if trace['type'] == 'scatter':  # Only process line/scatter plots
                    x_data = trace['x']
                    y_data = trace['y']
                    trace_name = trace['name']
                    
                    df = pd.DataFrame({'Year': x_data, trace_name: y_data})
                    
                    if consolidated_df.empty:
                        consolidated_df = df
                    else:
                        consolidated_df = pd.merge(consolidated_df, df, on='Year', how='outer')
            
            if not consolidated_df.empty:
                consolidated_df.sort_values('Year', inplace=True)
                filename = "emissions_timeseries.csv"
                return dcc.send_data_frame(consolidated_df.to_csv, filename, index=False)
            else:
                return None
        
        except Exception as e:
            print(f"Error in timeseries CSV download: {e}")
            return None
