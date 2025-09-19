#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Overview Page for LA Megacity Dashboard
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
                html.P([
                    "A comprehensive platform for monitoring, analyzing, and visualizing greenhouse gas emissions ",
                    "across the Los Angeles metropolitan area using ground-based measurements, satellite observations, ",
                    "and emission modeling systems."
                ], className="text-muted mb-0 fs-5")
            ]),
            id="overview-header-collapse",
            is_open=True,
        )
    ], className="mb-4 shadow-sm")

def create_system_overview_card():
    """Create system overview statistics card"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Button(
                [
                    html.I(className="fas fa-chart-pie me-2"),
                    "System Overview",
                    html.I(className="fas fa-chevron-down ms-2", id="system-overview-chevron"),
                ],
                color="link",
                id="system-overview-toggle",
                className="text-primary fs-5 text-decoration-none p-0 w-100 text-start",
                style={"box-shadow": "none"}
            )
        ]),
        dbc.Collapse(
            dbc.CardBody([
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
                                    html.H4("Annual", className="mb-1"),
                                    html.P("Data Updates", className="text-muted mb-0")
                                ], className="text-center")
                            ])
                        ], className="h-100 border-0 bg-light")
                    ], md=3)
                ], className="g-3")
            ]),
            id="system-overview-collapse",
            is_open=True,
        )
    ], className="mb-4 shadow-sm")

def create_dashboard_descriptions():
    """Create cards describing each dashboard module"""
    
    # Surface Observations Card
    surface_card = dbc.Card([
        dbc.CardHeader([
            dbc.Button(
                [
                    html.I(className="fas fa-industry me-2"),
                    "Surface CO₂ Network",
                    html.I(className="fas fa-chevron-down ms-2", id="surface-desc-chevron"),
                ],
                color="link",
                id="surface-desc-toggle",
                className="text-primary fs-5 text-decoration-none p-0 w-100 text-start",
                style={"box-shadow": "none"}
            )
        ]),
        dbc.Collapse(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Purpose", className="text-primary"),
                        html.P("Monitor atmospheric CO₂, CH₄, and CO concentrations across the Los Angeles basin using a network of ground-based sensors."),
                        
                        html.H6("Key Features", className="text-primary mt-3"),
                        html.Ul([
                            html.Li("Real-time and historical concentration data"),
                            html.Li("Multi-gas analysis (CO₂, CH₄, CO)"),
                            html.Li("Background subtraction capabilities"),
                            html.Li("Temporal aggregation (hourly to monthly)"),
                            html.Li("Statistical analysis tools"),
                            html.Li("Interactive site selection and comparison")
                        ], className="mb-3"),
                        
                        html.H6("Data Sources", className="text-primary"),
                        html.P("Ground-based monitoring stations with high-precision gas analyzers providing continuous measurements.")
                    ], md=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Navigation Tips", className="text-success"),
                                html.Small([
                                    html.Strong("• Select Sites: "), "Choose monitoring locations", html.Br(),
                                    html.Strong("• Time Range: "), "Use slider or date inputs", html.Br(),
                                    html.Strong("• Analysis: "), "Switch between raw/background data", html.Br(),
                                    html.Strong("• Export: "), "Download data or copy for AI analysis"
                                ])
                            ])
                        ], className="bg-light border-0")
                    ], md=4)
                ])
            ]),
            id="surface-desc-collapse",
            is_open=False,
        )
    ], className="mb-3")
    
    # OCO-3 Observations Card  
    oco3_card = dbc.Card([
        dbc.CardHeader([
            dbc.Button(
                [
                    html.I(className="fas fa-satellite me-2"),
                    "OCO-3 XCO₂ Snapshot Area Mode",
                    html.I(className="fas fa-chevron-down ms-2", id="oco3-desc-chevron"),
                ],
                color="link",
                id="oco3-desc-toggle",
                className="text-primary fs-5 text-decoration-none p-0 w-100 text-start",
                style={"box-shadow": "none"}
            )
        ]),
        dbc.Collapse(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Purpose", className="text-primary"),
                        html.P("Analyze column-averaged CO₂ concentrations from the Orbiting Carbon Observatory-3 satellite using Snapshot Area Mode (SAM) observations."),
                        
                        html.H6("Key Features", className="text-primary mt-3"),
                        html.Ul([
                            html.Li("High-resolution satellite XCO₂ measurements"),
                            html.Li("Animation controls for temporal visualization"),
                            html.Li("ZIP code aggregation capabilities"),
                            html.Li("Variable and fixed color scaling"),
                            html.Li("Statistical analysis (mean/standard deviation)"),
                            html.Li("Interactive spatial and temporal analysis")
                        ], className="mb-3"),
                        
                        html.H6("Data Sources", className="text-primary"),
                        html.P("NASA's OCO-3 satellite measurements providing detailed spatial coverage of atmospheric CO₂ concentrations.")
                    ], md=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Navigation Tips", className="text-success"),
                                html.Small([
                                    html.Strong("• Time Control: "), "Use play/pause for animation", html.Br(),
                                    html.Strong("• View Mode: "), "Switch between observations/ZIP aggregation", html.Br(),
                                    html.Strong("• Scale: "), "Adjust color ranges for comparison", html.Br(),
                                    html.Strong("• Interaction: "), "Click ZIP codes for time series"
                                ])
                            ])
                        ], className="bg-light border-0")
                    ], md=4)
                ])
            ]),
            id="oco3-desc-collapse",
            is_open=False,
        )
    ], className="mb-3")
    
    # Emissions (Hindcast) Card
    hindcast_card = dbc.Card([
        dbc.CardHeader([
            dbc.Button(
                [
                    html.I(className="fas fa-chart-line me-2"),
                    "Emissions (Hindcast Analysis)",
                    html.I(className="fas fa-chevron-down ms-2", id="hindcast-desc-chevron"),
                ],
                color="link",
                id="hindcast-desc-toggle",
                className="text-primary fs-5 text-decoration-none p-0 w-100 text-start",
                style={"box-shadow": "none"}
            )
        ]),
        dbc.Collapse(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Purpose", className="text-primary"),
                        html.P("Analyze historical CH₄ gas emissions patterns using inverse modeling and measurement-based approaches."),
                        
                        html.H6("Key Features", className="text-primary mt-3"),
                        html.Ul([
                            html.Li("Flux and uncertainty visualization"),
                            html.Li("Multiple spatial aggregations (native, ZIP, census, custom)"),
                            html.Li("Temporal aggregation controls"),
                            html.Li("Animation capabilities for time series"),
                            html.Li("Advanced color scaling options"),
                            html.Li("Comprehensive data export options")
                        ], className="mb-3"),
                        
                        html.H6("Data Sources", className="text-primary"),
                        html.P("Inverse modeling results combining atmospheric measurements with transport models to estimate emissions.")
                    ], md=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Navigation Tips", className="text-success"),
                                html.Small([
                                    html.Strong("• Display: "), "Toggle flux/uncertainty views", html.Br(),
                                    html.Strong("• Aggregation: "), "Choose spatial resolution", html.Br(),
                                    html.Strong("• Animation: "), "Control temporal playback", html.Br(),
                                    html.Strong("• Selection: "), "Click regions for time series"
                                ])
                            ])
                        ], className="bg-light border-0")
                    ], md=4)
                ])
            ]),
            id="hindcast-desc-collapse",
            is_open=False,
        )
    ], className="mb-3")
    
    # ABM Emissions Card
    abm_card = dbc.Card([
        dbc.CardHeader([
            dbc.Button(
                [
                    html.I(className="fas fa-chart-area me-2"),
                    "Agent Based Model (ABM) Emissions Analysis",
                    html.I(className="fas fa-chevron-down ms-2", id="abm-desc-chevron"),
                ],
                color="link",
                id="abm-desc-toggle",
                className="text-primary fs-5 text-decoration-none p-0 w-100 text-start",
                style={"box-shadow": "none"}
            )
        ]),
        dbc.Collapse(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Purpose", className="text-primary"),
                        html.P("Explore simulated household CO₂ emissions across Los Angeles County census tracts using agent-based modeling for future scenarios (2025-2054)."),
                        
                        html.H6("Key Features", className="text-primary mt-3"),
                        html.Ul([
                            html.Li("Absolute and per capita emissions analysis"),
                            html.Li("Socioeconomic variable integration"),
                            html.Li("Emission reduction scenario analysis"),
                            html.Li("Interactive temporal focus controls"),
                            html.Li("Linear and logarithmic scaling options"),
                            html.Li("Census tract-level spatial resolution")
                        ], className="mb-3"),
                        
                        html.H6("Data Sources", className="text-primary"),
                        html.P("Agent-based model simulations incorporating household behavior, policy scenarios, and socioeconomic factors.")
                    ], md=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Navigation Tips", className="text-success"),
                                html.Small([
                                    html.Strong("• Variables: "), "Switch between emissions/demographics", html.Br(),
                                    html.Strong("• Time Focus: "), "County vs. tract-level analysis", html.Br(),
                                    html.Strong("• Scaling: "), "Adjust color maps for visualization", html.Br(),
                                    html.Strong("• Selection: "), "Click tracts for detailed trends"
                                ])
                            ])
                        ], className="bg-light border-0")
                    ], md=4)
                ])
            ]),
            id="abm-desc-collapse",
            is_open=False,
        )
    ], className="mb-3")
    
    return [surface_card, oco3_card, hindcast_card, abm_card]

def create_technical_specifications():
    """Create technical specifications section"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Button(
                [
                    html.I(className="fas fa-cogs me-2"),
                    "Technical Specifications",
                    html.I(className="fas fa-chevron-down ms-2", id="tech-specs-chevron"),
                ],
                color="link",
                id="tech-specs-toggle",
                className="text-primary fs-5 text-decoration-none p-0 w-100 text-start",
                style={"box-shadow": "none"}
            )
        ]),
        dbc.Collapse(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Data Processing", className="text-primary"),
                        html.Ul([
                            html.Li("Real-time quality control and validation"),
                            html.Li("Background subtraction algorithms"),
                            html.Li("Temporal aggregation (1-hour to monthly)"),
                            html.Li("Spatial aggregation at multiple scales"),
                            html.Li("Statistical outlier detection")
                        ])
                    ], md=6),
                    dbc.Col([
                        html.H6("Visualization Features", className="text-primary"),
                        html.Ul([
                            html.Li("Interactive mapping with zoom/pan"),
                            html.Li("Time series animation controls"),
                            html.Li("Multi-panel layouts with expansion"),
                            html.Li("Customizable color scales"),
                            html.Li("Data export in multiple formats")
                        ])
                    ], md=6)
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.H6("Supported Data Formats", className="text-primary"),
                        dbc.Row([
                            dbc.Col([
                                html.Strong("Input: "),
                                html.Span("NetCDF, HDF5, CSV, Shapefiles")
                            ], md=6),
                            dbc.Col([
                                html.Strong("Export: "),
                                html.Span("CSV, NetCDF, ZIP archives")
                            ], md=6)
                        ])
                    ], md=12)
                ])
            ]),
            id="tech-specs-collapse",
            is_open=False,
        )
    ], className="mb-4 shadow-sm")

def create_getting_started():
    """Create getting started guide"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Button(
                [
                    html.I(className="fas fa-rocket me-2"),
                    "Getting Started",
                    html.I(className="fas fa-chevron-down ms-2", id="getting-started-chevron"),
                ],
                color="link",
                id="getting-started-toggle",
                className="text-primary fs-5 text-decoration-none p-0 w-100 text-start",
                style={"box-shadow": "none"}
            )
        ]),
        dbc.Collapse(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("1. Choose Your Analysis", className="text-primary"),
                            html.P("Select the appropriate dashboard based on your research needs:"),
                            html.Ul([
                                html.Li(html.Strong("Surface Network: "), "For detailed atmospheric concentration analysis"),
                                html.Li(html.Strong("OCO-3: "), "For satellite-based regional CO₂ patterns"),
                                html.Li(html.Strong("Hindcast: "), "For historical emission estimates"),
                                html.Li(html.Strong("ABM: "), "For future scenario modeling")
                            ], className="mb-4")
                        ])
                    ], md=6),
                    dbc.Col([
                        html.Div([
                            html.H6("2. Configure Analysis Parameters", className="text-primary"),
                            html.P("Use the control panels to customize your analysis:"),
                            html.Ul([
                                html.Li("Select time periods and spatial regions"),
                                html.Li("Choose appropriate aggregation levels"),
                                html.Li("Configure visualization settings"),
                                html.Li("Set analysis metrics and variables")
                            ], className="mb-4")
                        ])
                    ], md=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("3. Explore Interactively", className="text-primary"),
                            html.P("Use interactive features to dive deeper:"),
                            html.Ul([
                                html.Li("Click on maps to see detailed time series"),
                                html.Li("Use animation controls for temporal patterns"),
                                html.Li("Expand panels for focused analysis"),
                                html.Li("Generate statistics for quantitative insights")
                            ])
                        ])
                    ], md=6),
                    dbc.Col([
                        html.Div([
                            html.H6("4. Export and Share", className="text-primary"),
                            html.P("Save your analysis results:"),
                            html.Ul([
                                html.Li("Download data in various formats"),
                                html.Li("Copy formatted data for AI analysis"),
                                html.Li("Export visualizations"),
                                html.Li("Access comprehensive datasets")
                            ])
                        ])
                    ], md=6)
                ])
            ]),
            id="getting-started-collapse",
            is_open=True,
        )
    ], className="mb-4 shadow-sm")

def get_layout():
    """Return the complete overview page layout"""
    return dbc.Container([
        # Header
        create_header_card(),
        
        # System Overview Statistics
        create_system_overview_card(),
        
        # Getting Started Guide
        create_getting_started(),
        
        # Dashboard Descriptions
        html.H4("Dashboard Modules", className="mb-3 text-primary"),
        html.Div(create_dashboard_descriptions()),
        
        # Technical Specifications
        create_technical_specifications(),
        
        # Footer with additional information
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Contact & Support", className="text-primary"),
                        html.P("For technical support, data access requests, or collaboration inquiries, please refer to the documentation or contact Vineet Yadav at yadavvineet@gmail.com.", className="text-muted")
                    ], md=8),
                    dbc.Col([
                        html.H6("Version Information", className="text-primary"),
                        html.P("System Version: 2.0", className="text-muted mb-1"),
                        html.P("Last Updated: 2025", className="text-muted mb-0")
                    ], md=4)
                ])
            ])
        ], className="mt-4 shadow-sm border-0 bg-light")
        
    ], fluid=True, className="px-4 py-3")

def register_callbacks(app):
    """Register callbacks for the overview page"""
    
    # Header collapse callback
    @app.callback(
        [Output("overview-header-collapse", "is_open"),
         Output("overview-header-chevron", "className")],
        [Input("overview-header-toggle", "n_clicks")],
        [State("overview-header-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_header_collapse(n_clicks, is_open):
        if n_clicks:
            return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
        return is_open, "fas fa-chevron-down ms-2"
    
    # System overview collapse callback
    @app.callback(
        [Output("system-overview-collapse", "is_open"),
         Output("system-overview-chevron", "className")],
        [Input("system-overview-toggle", "n_clicks")],
        [State("system-overview-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_system_overview_collapse(n_clicks, is_open):
        if n_clicks:
            return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
        return is_open, "fas fa-chevron-down ms-2"
    
    # Getting started collapse callback
    @app.callback(
        [Output("getting-started-collapse", "is_open"),
         Output("getting-started-chevron", "className")],
        [Input("getting-started-toggle", "n_clicks")],
        [State("getting-started-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_getting_started_collapse(n_clicks, is_open):
        if n_clicks:
            return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
        return is_open, "fas fa-chevron-down ms-2"
    
    # Dashboard description collapse callbacks
    for module in ["surface", "oco3", "hindcast", "abm"]:
        @app.callback(
            [Output(f"{module}-desc-collapse", "is_open"),
             Output(f"{module}-desc-chevron", "className")],
            [Input(f"{module}-desc-toggle", "n_clicks")],
            [State(f"{module}-desc-collapse", "is_open")],
            prevent_initial_call=True
        )
        def toggle_desc_collapse(n_clicks, is_open):
            if n_clicks:
                return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
            return is_open, "fas fa-chevron-down ms-2"
    
    # Technical specifications collapse callback
    @app.callback(
        [Output("tech-specs-collapse", "is_open"),
         Output("tech-specs-chevron", "className")],
        [Input("tech-specs-toggle", "n_clicks")],
        [State("tech-specs-collapse", "is_open")],
        prevent_initial_call=True
    )
    def toggle_tech_specs_collapse(n_clicks, is_open):
        if n_clicks:
            return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
        return is_open, "fas fa-chevron-down ms-2"
