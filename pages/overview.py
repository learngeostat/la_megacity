#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Overview Page for LA Megacity Information System
Provides comprehensive description of system capabilities and features
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go

def init():
    """Initialize any required resources for the overview page"""
    pass

def create_header_card():
    """Create the main header card"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className="fas fa-info-circle me-2"),
                            "Los Angeles Megacity Greenhouse Gas Information System",
                            html.I(className="fas fa-chevron-down ms-2", id="overview-header-chevron"),
                        ],
                        color="link",
                        id="overview-header-toggle",
                        className="text-primary fs-4 text-decoration-none p-0 w-100 text-start",
                        style={"box-shadow": "none"}
                    )
                ], width=12)
            ], className="align-items-center")
        ]),
        dbc.Collapse(
            dbc.CardBody([
                html.P(
                    "A comprehensive platform for monitoring, analyzing, and visualizing greenhouse gas emissions "
                    "across the Los Angeles metropolitan area using ground-based measurements, satellite observations, "
                    "and emission modeling systems. Near-real-time status monitor for measurements and fluxes is under development.",
                    className="text-muted mb-3 fs-5"
                ),
                # System Overview Statistics
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-broadcast-tower fa-2x text-primary mb-2"),
                                    html.H4("12+", className="mb-1"),
                                    html.P("Monitoring Stations", className="text-muted mb-0")
                                ], className="text-center")
                            ])
                        ], className="h-100 border-0 bg-light")
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-satellite fa-2x text-success mb-2"),
                                    html.H4("1", className="mb-1"),
                                    html.P("Satellite Data Source", className="text-muted mb-0")
                                ], className="text-center")
                            ])
                        ], className="h-100 border-0 bg-light")
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-map-marked-alt fa-2x text-info mb-2"),
                                    html.H4("3,500+", className="mb-1"),
                                    html.P("Census Tracts Covered", className="text-muted mb-0")
                                ], className="text-center")
                            ])
                        ], className="h-100 border-0 bg-light")
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="fas fa-clock fa-2x text-warning mb-2"),
                                    html.H4("Development", className="mb-1"),
                                    html.P("Near-Real-Time Monitor", className="text-muted mb-0")
                                ], className="text-center")
                            ])
                        ], className="h-100 border-0 bg-light")
                    ], md=3)
                ], className="g-3")
            ]),
            id="overview-header-collapse",
            is_open=True,
        )
    ], className="mb-4 shadow-sm")

def create_oco3_card():
    """Create OCO-3 SAM Mode card"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Button(
                [
                    html.I(className="fas fa-satellite me-2"),
                    "OCO-3 Snapshot Area Mode (SAM)",
                    html.I(className="fas fa-chevron-down ms-2", id="oco3-chevron"),
                ],
                color="link",
                id="oco3-toggle",
                className="text-primary fs-5 text-decoration-none p-0 w-100 text-start",
                style={"box-shadow": "none"}
            )
        ]),
        dbc.Collapse(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Overview", className="text-primary"),
                        html.P(
                            "Analyze column-averaged CO₂ concentrations from the Orbiting Carbon Observatory-3 satellite "
                            "using high-resolution Snapshot Area Mode observations over the Los Angeles metropolitan area."
                        ),
                        
                        html.H6("Key Capabilities", className="text-primary mt-3"),
                        html.Ul([
                            html.Li("High-resolution satellite XCO₂ measurements with spatial detail"),
                            html.Li("Animation controls for temporal pattern visualization"),
                            html.Li("ZIP code aggregation for neighborhood-level analysis"),
                            html.Li("Variable and fixed color scaling for comparative studies"),
                            html.Li("Statistical analysis including mean and standard deviation"),
                            html.Li("Interactive spatial and temporal data exploration")
                        ], className="mb-3"),
                        
                        html.H6("Data Characteristics", className="text-primary"),
                        html.P("NASA's OCO-3 satellite provides detailed spatial coverage of atmospheric CO₂ concentrations with enhanced resolution over urban areas through the SAM observation mode.")
                    ], md=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Usage Guide", className="text-success mb-3"),
                                html.Div([
                                    html.Strong("Time Animation: "), 
                                    html.Span("Use play/pause controls to visualize temporal changes"), html.Br(), html.Br(),
                                    html.Strong("View Modes: "), 
                                    html.Span("Switch between raw observations and ZIP code aggregation"), html.Br(), html.Br(),
                                    html.Strong("Color Scaling: "), 
                                    html.Span("Adjust ranges for enhanced pattern detection"), html.Br(), html.Br(),
                                    html.Strong("Interaction: "), 
                                    html.Span("Click ZIP codes to generate time series plots")
                                ])
                            ])
                        ], className="bg-light border-0")
                    ], md=4)
                ])
            ]),
            id="oco3-collapse",
            is_open=False,
        )
    ], className="mb-4 shadow-sm")

def create_surface_card():
    """Create Surface Observations card"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Button(
                [
                    html.I(className="fas fa-industry me-2"),
                    "Surface Observations Network",
                    html.I(className="fas fa-chevron-down ms-2", id="surface-chevron"),
                ],
                color="link",
                id="surface-toggle",
                className="text-primary fs-5 text-decoration-none p-0 w-100 text-start",
                style={"box-shadow": "none"}
            )
        ]),
        dbc.Collapse(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Overview", className="text-primary"),
                        html.P(
                            "Monitor atmospheric CO₂, CH₄, and CO concentrations across the Los Angeles basin "
                            "using a network of ground-based high-precision sensors providing continuous measurements."
                        ),
                        
                        html.H6("Key Capabilities", className="text-primary mt-3"),
                        html.Ul([
                            html.Li("Real-time and historical concentration data from multiple stations"),
                            html.Li("Multi-gas analysis capabilities (CO₂, CH₄, CO)"),
                            html.Li("Background subtraction for enhanced signal detection"),
                            html.Li("Flexible temporal aggregation from hourly to monthly scales"),
                            html.Li("Advanced statistical analysis and quality control tools"),
                            html.Li("Interactive site selection and cross-station comparison")
                        ], className="mb-3"),
                        
                        html.H6("Data Characteristics", className="text-primary"),
                        html.P("Ground-based monitoring stations equipped with high-precision gas analyzers providing continuous, calibrated measurements with comprehensive quality assurance protocols.")
                    ], md=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Usage Guide", className="text-success mb-3"),
                                html.Div([
                                    html.Strong("Site Selection: "), 
                                    html.Span("Choose from available monitoring locations"), html.Br(), html.Br(),
                                    html.Strong("Time Controls: "), 
                                    html.Span("Use slider or date inputs for period selection"), html.Br(), html.Br(),
                                    html.Strong("Analysis Mode: "), 
                                    html.Span("Switch between raw and background-subtracted data"), html.Br(), html.Br(),
                                    html.Strong("Data Export: "), 
                                    html.Span("Download data or copy for external analysis")
                                ])
                            ])
                        ], className="bg-light border-0")
                    ], md=4)
                ])
            ]),
            id="surface-collapse",
            is_open=False,
        )
    ], className="mb-4 shadow-sm")

def create_flux_hindcast_card():
    """Create Flux Hindcast card"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Button(
                [
                    html.I(className="fas fa-history me-2"),
                    "Flux Hindcast Analysis",
                    html.I(className="fas fa-chevron-down ms-2", id="hindcast-chevron"),
                ],
                color="link",
                id="hindcast-toggle",
                className="text-primary fs-5 text-decoration-none p-0 w-100 text-start",
                style={"box-shadow": "none"}
            )
        ]),
        dbc.Collapse(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Overview", className="text-primary"),
                        html.P(
                            "Analyze historical CH₄ gas emissions patterns using inverse modeling techniques that "
                            "combine atmospheric measurements with transport models to estimate past emission sources and magnitudes."
                        ),
                        
                        html.H6("Key Capabilities", className="text-primary mt-3"),
                        html.Ul([
                            html.Li("Flux magnitude and uncertainty visualization with confidence intervals"),
                            html.Li("Multiple spatial aggregation levels (native grid, ZIP codes, census tracts, custom regions)"),
                            html.Li("Temporal aggregation controls from daily to seasonal scales"),
                            html.Li("Animation capabilities for time series pattern identification"),
                            html.Li("Advanced color scaling with logarithmic and linear options"),
                            html.Li("Comprehensive data export in multiple formats")
                        ], className="mb-3"),
                        
                        html.H6("Data Characteristics", className="text-primary"),
                        html.P("Inverse modeling results combining atmospheric observations with meteorological transport models to provide spatially and temporally resolved emission estimates with quantified uncertainties.")
                    ], md=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Usage Guide", className="text-success mb-3"),
                                html.Div([
                                    html.Strong("Display Options: "), 
                                    html.Span("Toggle between flux and uncertainty visualizations"), html.Br(), html.Br(),
                                    html.Strong("Spatial Scale: "), 
                                    html.Span("Choose appropriate aggregation resolution"), html.Br(), html.Br(),
                                    html.Strong("Time Animation: "), 
                                    html.Span("Control temporal playback for pattern analysis"), html.Br(), html.Br(),
                                    html.Strong("Region Selection: "), 
                                    html.Span("Click areas to generate detailed time series")
                                ])
                            ])
                        ], className="bg-light border-0")
                    ], md=4)
                ])
            ]),
            id="hindcast-collapse",
            is_open=False,
        )
    ], className="mb-4 shadow-sm")

def create_flux_forecast_card():
    """Create Flux Forecast card"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Button(
                [
                    html.I(className="fas fa-chart-line me-2"),
                    "Flux Forecast System",
                    html.I(className="fas fa-chevron-down ms-2", id="forecast-chevron"),
                ],
                color="link",
                id="forecast-toggle",
                className="text-primary fs-5 text-decoration-none p-0 w-100 text-start",
                style={"box-shadow": "none"}
            )
        ]),
        dbc.Collapse(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Overview", className="text-primary"),
                        html.P(
                            "Explore projected greenhouse gas emissions across Los Angeles County using advanced "
                            "modeling approaches that integrate socioeconomic factors, policy scenarios, and behavioral patterns "
                            "to forecast future emission trajectories."
                        ),
                        
                        html.H6("Key Capabilities", className="text-primary mt-3"),
                        html.Ul([
                            html.Li("Multi-scenario emission projections with policy impact assessment"),
                            html.Li("Socioeconomic variable integration and demographic trend analysis"),
                            html.Li("Emission reduction pathway evaluation and optimization"),
                            html.Li("Interactive temporal focus controls for detailed period analysis"),
                            html.Li("Flexible scaling options (linear and logarithmic) for data visualization"),
                            html.Li("High-resolution spatial analysis at census tract level")
                        ], className="mb-3"),
                        
                        html.H6("Data Characteristics", className="text-primary"),
                        html.P("Advanced modeling simulations incorporating household behavior patterns, policy implementation scenarios, demographic changes, and economic factors to provide comprehensive future emission projections.")
                    ], md=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Usage Guide", className="text-success mb-3"),
                                html.Div([
                                    html.Strong("Scenario Selection: "), 
                                    html.Span("Choose from available policy and behavioral scenarios"), html.Br(), html.Br(),
                                    html.Strong("Variable Analysis: "), 
                                    html.Span("Switch between emissions and socioeconomic indicators"), html.Br(), html.Br(),
                                    html.Strong("Temporal Focus: "), 
                                    html.Span("Adjust analysis period for detailed examination"), html.Br(), html.Br(),
                                    html.Strong("Tract Selection: "), 
                                    html.Span("Click census tracts for detailed trend analysis")
                                ])
                            ])
                        ], className="bg-light border-0")
                    ], md=4)
                ])
            ]),
            id="forecast-collapse",
            is_open=False,
        )
    ], className="mb-4 shadow-sm")

def create_technical_info_card():
    """Create technical information and getting started card"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Button(
                [
                    html.I(className="fas fa-info me-2"),
                    "System Information & Getting Started",
                    html.I(className="fas fa-chevron-down ms-2", id="info-chevron"),
                ],
                color="link",
                id="info-toggle",
                className="text-primary fs-5 text-decoration-none p-0 w-100 text-start",
                style={"box-shadow": "none"}
            )
        ]),
        dbc.Collapse(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Getting Started", className="text-primary"),
                        html.Div([
                            html.Strong("1. Module Selection: "), 
                            html.Span("Choose the appropriate analysis module based on your research objectives."), html.Br(),
                            html.Strong("2. Parameter Configuration: "), 
                            html.Span("Use control panels to set time periods, spatial regions, and analysis options."), html.Br(),
                            html.Strong("3. Interactive Exploration: "), 
                            html.Span("Click on maps and use animation controls to explore patterns."), html.Br(),
                            html.Strong("4. Data Export: "), 
                            html.Span("Download results in various formats for further analysis.")
                        ], className="mb-4"),
                        
                        html.H6("Technical Specifications", className="text-primary"),
                        html.Ul([
                            html.Li("Real-time quality control and validation algorithms"),
                            html.Li("Multi-scale spatial and temporal aggregation capabilities"),
                            html.Li("Interactive visualization with zoom, pan, and selection tools"),
                            html.Li("Statistical analysis including uncertainty quantification"),
                            html.Li("Multiple data export formats (CSV, NetCDF, ZIP archives)")
                        ], className="mb-3")
                    ], md=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Contact & Support", className="text-primary mb-3"),
                                html.P("For technical support, data access requests, or collaboration inquiries:", className="text-muted mb-2"),
                                html.P("Vineet Yadav", className="mb-1"),
                                html.P("yadavvineet@gmail.com", className="text-muted mb-3"),
                                
                                html.H6("Version Information", className="text-primary"),
                                html.P("Beta System Version: 2.0", className="text-muted mb-1"),
                                html.P("Last Updated: 2025", className="text-muted mb-0")
                            ])
                        ], className="bg-light border-0")
                    ], md=4)
                ])
            ]),
            id="info-collapse",
            is_open=True,
        )
    ], className="mb-4 shadow-sm")

def get_layout():
    """Return the complete overview page layout"""
    return dbc.Container([
        # Main Overview Card (includes system stats)
        create_header_card(),
        
        # OCO-3 SAM Mode Card
        create_oco3_card(),
        
        # Surface Observations Card
        create_surface_card(),
        
        # Flux Hindcast Card
        create_flux_hindcast_card(),
        
        # Flux Forecast Card
        create_flux_forecast_card(),
        
        # Technical Information & Getting Started
        create_technical_info_card()
        
    ], fluid=True, className="px-4 py-3")

def register_callbacks(app):
    """Register callbacks for the overview page"""
    
    # Define all the collapse components and their corresponding elements
    collapse_components = [
        ("overview-header", "overview-header-collapse", "overview-header-chevron"),
        ("oco3", "oco3-collapse", "oco3-chevron"),
        ("surface", "surface-collapse", "surface-chevron"),
        ("hindcast", "hindcast-collapse", "hindcast-chevron"),
        ("forecast", "forecast-collapse", "forecast-chevron"),
        ("info", "info-collapse", "info-chevron")
    ]
    
    # Create callbacks for each collapse component
    for toggle_id, collapse_id, chevron_id in collapse_components:
        @app.callback(
            [Output(collapse_id, "is_open"),
             Output(chevron_id, "className")],
            [Input(f"{toggle_id}-toggle", "n_clicks")],
            [State(collapse_id, "is_open")],
            prevent_initial_call=True
        )
        def make_toggle_callback(collapse_id=collapse_id, chevron_id=chevron_id):
            def toggle_collapse(n_clicks, is_open):
                if n_clicks:
                    new_state = not is_open
                    chevron_class = "fas fa-chevron-up ms-2" if new_state else "fas fa-chevron-down ms-2"
                    return new_state, chevron_class
                return is_open, "fas fa-chevron-down ms-2"
            return toggle_collapse
        
        # Create the actual callback with unique function
        app.callback(
            [Output(collapse_id, "is_open"),
             Output(chevron_id, "className")],
            [Input(f"{toggle_id}-toggle", "n_clicks")],
            [State(collapse_id, "is_open")],
            prevent_initial_call=True
        )(make_toggle_callback())
