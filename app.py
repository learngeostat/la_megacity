from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import os
import logging
import sys

# All working dependencies from Phase 3
import geopandas as gpd
import xarray as xr
import scipy
import astropy
import statsmodels

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Initialize the app with both Bootstrap and Font Awesome
external_stylesheets = [
    dbc.themes.LUX,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"
]

app = Dash(__name__, 
           external_stylesheets=external_stylesheets, 
           use_pages=False,
           pages_folder="",
           suppress_callback_exceptions=True)

# CRITICAL: Make server accessible at module level for deployment
server = app.server

# Main app layout matching your original structure
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    # Hero Section with Header
    dbc.Row(
        dbc.Col(
            html.Div([
                html.H1(
                    [
                        html.I(className="fas fa-city me-3"),
                        "Los Angeles Megacity",
                        html.Br(),
                        "Greenhouse Gas Information System"
                    ],
                    className="text-center text-white mb-4",
                    style={"fontSize": "2.5rem", "fontWeight": "600"}
                ),
                html.P(
                    "Monitoring and analyzing greenhouse gas emissions across the Los Angeles metropolitan area",
                    className="text-center text-white mb-0",
                    style={"fontSize": "1.2rem", "opacity": "0.9"}
                )
            ], className="hero-section")
        ),
        className="mb-4"
    ),
    
    # Status indicator
    dbc.Row(
        dbc.Col(
            html.Div([
                html.I(className="fas fa-check-circle text-success me-2"),
                html.Span("Phase 4A: All dependencies loaded successfully", className="small")
            ], className="text-center mb-2")
        )
    ),
    
    # Horizontal Navigation Tabs
    dbc.Row(
        dbc.Col(
            dbc.Nav([
                dbc.NavLink([
                    html.I(className="fas fa-bars me-2"),
                    "Overview"
                ], href="/page-1", active="exact", className="px-4"),
                
                dbc.NavLink([
                    html.I(className="fas fa-satellite me-2"),
                    "OCO-3 Observations"
                ], href="/page-2", active="exact", className="px-4"),
                
                dbc.NavLink([
                    html.I(className="fas fa-tower-observation me-2"),
                    "Surface Observations"
                ], href="/page-3", active="exact", className="px-4"),
                
                dbc.NavLink([
                    html.I(className="fas fa-chart-line me-2"),
                    "Emissions (Hindcast/Nowcast)"
                ], href="/page-4", active="exact", className="px-4"),
                
                dbc.NavLink([
                    html.I(className="fas fa-chart-line me-2"),
                    "Emissions (Forecast)"
                ], href="/page-5", active="exact", className="px-4"),
            ],
            pills=True,
            className="nav-pills-horizontal mb-4"),
            className="d-flex justify-content-center"
        )
    ),
    
    # Main Content - Full Width
    dbc.Row([
        dbc.Col([
            html.Div(id="page-content")
        ], width=12)
    ])
], className="container-fluid px-4 py-3")

# Update the CSS styles (same as your original)
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>LA Megacity GHG System</title>
        {%favicon%}
        {%css%}
        <!-- Add Font Awesome CSS -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
        <style>
            .nav-pills-horizontal {
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                padding: 0.5rem;
            }
            .nav-pills-horizontal .nav-link {
                color: #495057;
                transition: all 0.3s ease;
                border-radius: 5px;
                margin: 0 0.25rem;
                padding: 0.75rem 1.5rem;
                font-size: 1rem;
            }
            .nav-pills-horizontal .nav-link:hover {
                background-color: #e9ecef;
                transform: translateY(-2px);
            }
            .nav-pills-horizontal .nav-link.active {
                background-color: #6c757d;
                color: white;
            }
            .hero-section {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                padding: 2rem;
                border-radius: 0 0 20px 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .content-card {
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
            }
            .content-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Callback to handle page routing
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_page_content(pathname):
    try:
        logger.info(f"Rendering page: {pathname}")
        
        if pathname == "/page-1" or pathname == "/" or not pathname:
            return get_overview_placeholder()
        elif pathname == "/page-2":
            return get_oco3_placeholder()
        elif pathname == "/page-3":
            return get_surface_placeholder()
        elif pathname == "/page-4":
            return get_hindcast_placeholder()
        elif pathname == "/page-5":
            return get_forecast_placeholder()
        else:
            return html.Div([
                html.H1("404: Not found", className="text-danger"),
                html.P(f"The pathname {pathname} was not recognised...")
            ])
    except Exception as e:
        logger.error(f"Error rendering page {pathname}: {str(e)}")
        return html.Div([
            html.H1("Error", className="text-danger"),
            html.P(f"An error occurred: {str(e)}")
        ])

# Placeholder page functions - identical structure to your original but with simple content
def get_overview_placeholder():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Overview Dashboard", className="mb-4"),
                dbc.Alert([
                    html.H4("Phase 4A: Base Structure Working", className="alert-heading"),
                    html.P("All dependencies loaded successfully. This page will contain:"),
                    html.Ul([
                        html.Li("System status indicators"),
                        html.Li("Recent data summary"),
                        html.Li("Quick navigation cards"),
                        html.Li("Key metrics overview")
                    ])
                ], color="success"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Dependencies Status", className="card-title"),
                                html.P("✅ All scientific libraries loaded"),
                                html.P("✅ Geospatial libraries working"),
                                html.P("✅ Visualization ready")
                            ])
                        ])
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Next Steps", className="card-title"),
                                html.P("Ready to add real functionality"),
                                html.P("Will enable pages incrementally"),
                                html.P("Phase 4B: Enable this page next")
                            ])
                        ])
                    ], width=6)
                ])
            ])
        ])
    ])

def get_oco3_placeholder():
    return dbc.Container([
        html.H2("OCO-3 Satellite Observations", className="mb-4"),
        dbc.Alert([
            html.H4("Phase 4A: Placeholder Page"),
            html.P("This page will contain:"),
            html.Ul([
                html.Li("OCO-3 satellite data visualization"),
                html.Li("CO2 concentration maps"),
                html.Li("Time series analysis"),
                html.Li("Data filtering and selection tools")
            ])
        ], color="info")
    ])

def get_surface_placeholder():
    return dbc.Container([
        html.H2("Surface Observations", className="mb-4"),
        dbc.Alert([
            html.H4("Phase 4A: Placeholder Page"),
            html.P("This page will contain:"),
            html.Ul([
                html.Li("Ground-based measurement data"),
                html.Li("Station network visualization"),
                html.Li("Real-time monitoring displays"),
                html.Li("Data quality indicators")
            ])
        ], color="info")
    ])

def get_hindcast_placeholder():
    return dbc.Container([
        html.H2("Emissions Analysis (Hindcast/Nowcast)", className="mb-4"),
        dbc.Alert([
            html.H4("Phase 4A: Placeholder Page"),
            html.P("This page will contain:"),
            html.Ul([
                html.Li("Historical emissions analysis"),
                html.Li("Current emissions estimates"),
                html.Li("Flux calculation results"),
                html.Li("Uncertainty analysis")
            ])
        ], color="info")
    ])

def get_forecast_placeholder():
    return dbc.Container([
        html.H2("Emissions Forecast", className="mb-4"),
        dbc.Alert([
            html.H4("Phase 4A: Placeholder Page"),
            html.P("This page will contain:"),
            html.Ul([
                html.Li("Future emissions projections"),
                html.Li("Scenario modeling"),
                html.Li("Forecast confidence intervals"),
                html.Li("Model validation metrics")
            ])
        ], color="info")
    ])

# URL redirect callback
@app.callback(
    Output("url", "pathname"),
    Input("url", "pathname")
)
def init_pathname(pathname):
    if pathname == "/" or not pathname:
        return "/page-1"
    return pathname

# Health check endpoint
@server.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'phase': 'phase-4a-base-structure',
        'pages': ['overview', 'oco3', 'surface', 'hindcast', 'forecast'],
        'dependencies_loaded': True
    }, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Phase 4A base structure on port {port}")
    logger.info("All dependencies loaded successfully")
    
    app.run_server(
        host='0.0.0.0',
        port=port,
        debug=False,
        dev_tools_hot_reload=False,
        dev_tools_ui=False,
        dev_tools_props_check=False
    )
