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

# Import overview module (from Phase 4B)
try:
    from pages.overview import init as init_overview
    from pages.overview import get_layout as get_overview_layout
    from pages.overview import register_callbacks as register_overview_callbacks
    OVERVIEW_MODULE_LOADED = True
    OVERVIEW_ERROR = None
except Exception as e:
    OVERVIEW_MODULE_LOADED = False
    OVERVIEW_ERROR = str(e)

# Import flux_forecast module (from Phase 4D)
try:
    from pages.flux_forecast import init as init_flux_forecast
    from pages.flux_forecast import get_layout as get_flux_forecast_layout
    from pages.flux_forecast import register_callbacks as register_flux_forecast_callbacks
    FLUX_FORECAST_MODULE_LOADED = True
    FLUX_FORECAST_ERROR = None
except Exception as e:
    FLUX_FORECAST_MODULE_LOADED = False
    FLUX_FORECAST_ERROR = str(e)

# Import surface_observations module - FROM PHASE 4F
try:
    from pages.surface_observations import init as init_surface_observations
    from pages.surface_observations import get_layout as get_surface_observations_layout
    from pages.surface_observations import register_callbacks as register_surface_observations_callbacks
    SURFACE_OBSERVATIONS_MODULE_LOADED = True
    SURFACE_OBSERVATIONS_ERROR = None
except Exception as e:
    SURFACE_OBSERVATIONS_MODULE_LOADED = False
    SURFACE_OBSERVATIONS_ERROR = str(e)

# Import emissions module - NEW FOR THIS UPDATE
try:
    from pages.emissions import init as init_emissions
    from pages.emissions import get_layout as get_emissions_layout
    from pages.emissions import register_callbacks as register_emissions_callbacks
    EMISSIONS_MODULE_LOADED = True
    EMISSIONS_ERROR = None
except Exception as e:
    EMISSIONS_MODULE_LOADED = False
    EMISSIONS_ERROR = str(e)
    
# Import flux_hindcast module - ADDED FOR HINDCAST/NOWCAST
try:
    from pages.flux_hindcast import init as init_flux_hindcast
    from pages.flux_hindcast import get_layout as get_flux_hindcast_layout
    from pages.flux_hindcast import register_callbacks as register_flux_hindcast_callbacks
    FLUX_HINDCAST_MODULE_LOADED = True
    FLUX_HINDCAST_ERROR = None
except Exception as e:
    FLUX_HINDCAST_MODULE_LOADED = False
    FLUX_HINDCAST_ERROR = str(e)

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
           suppress_callback_exceptions=True,
           assets_folder='assets')

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

    # Status indicator - Updated to include all modules including flux_hindcast
    dbc.Row(
        dbc.Col(
            html.Div([
                html.I(className=f"fas fa-{'check-circle text-success' if (OVERVIEW_MODULE_LOADED and FLUX_FORECAST_MODULE_LOADED and SURFACE_OBSERVATIONS_MODULE_LOADED and EMISSIONS_MODULE_LOADED and FLUX_HINDCAST_MODULE_LOADED) else 'exclamation-triangle text-warning'} me-2"),
                html.Span(f"System Status: Overview {'✓' if OVERVIEW_MODULE_LOADED else '✗'} | Flux Forecast {'✓' if FLUX_FORECAST_MODULE_LOADED else '✗'} | Surface Obs {'✓' if SURFACE_OBSERVATIONS_MODULE_LOADED else '✗'} | Emissions {'✓' if EMISSIONS_MODULE_LOADED else '✗'} | Flux Hindcast {'✓' if FLUX_HINDCAST_MODULE_LOADED else '✗'}", className="small")
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

# CSS styles (with all necessary styles for surface observations)
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>LA Megacity GHG System</title>
        {%favicon%}
        {%css%}
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
            .panel-transition {
                transition: all 0.3s ease-in-out;
            }
            .expand-control {
                display: flex;
                flex-direction: column;
                gap: 10px;
                align-items: center;
                justify-content: center;
                height: 100%;
            }
            .expand-button {
                background: #007bff;
                border: none;
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            }
            .expand-button:hover {
                background: #0056b3;
            }
            .restore-button {
                background: #28a745;
                border: none;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }
            .restore-button:hover {
                background: #218838;
            }
            .time-control-overlay {
                position: absolute;
                top: 10px;
                left: 10px;
                width: calc(100% - 20px);
                background: rgba(255, 255, 255, 0.95);
                border-radius: 8px;
                padding: 10px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                z-index: 1000;
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
            if OVERVIEW_MODULE_LOADED:
                return get_overview_layout()
            else:
                return get_overview_error_layout()
        elif pathname == "/page-2":
            if EMISSIONS_MODULE_LOADED:
                return get_emissions_layout()
            else:
                return get_emissions_error_layout()
        elif pathname == "/page-3":
            if SURFACE_OBSERVATIONS_MODULE_LOADED:
                return get_surface_observations_layout()
            else:
                return get_surface_observations_error_layout()
        elif pathname == "/page-4":
            if FLUX_HINDCAST_MODULE_LOADED:
                return get_flux_hindcast_layout()
            else:
                return get_flux_hindcast_error_layout()
        elif pathname == "/page-5":
            if FLUX_FORECAST_MODULE_LOADED:
                return get_flux_forecast_layout()
            else:
                return get_flux_forecast_error_layout()
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

# Error layout if overview module fails to load
def get_overview_error_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Overview Page - Module Load Error", className="mb-4"),
                dbc.Alert([
                    html.H4("Overview Module Import Failed", className="alert-heading"),
                    html.P(f"Error loading overview module: {OVERVIEW_ERROR}"),
                    html.Hr(),
                    html.P("This indicates an issue with the overview module import structure.")
                ], color="danger")
            ])
        ])
    ])

# Error layout if flux_hindcast module fails to load
def get_flux_hindcast_error_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Emissions (Hindcast/Nowcast) - Module Load Error", className="mb-4"),
                dbc.Alert([
                    html.H4("Flux Hindcast Module Import Failed", className="alert-heading"),
                    html.P(f"Error loading flux_hindcast module: {FLUX_HINDCAST_ERROR}"),
                ], color="danger")
            ])
        ])
    ])

# Error layout if flux_forecast module fails to load
def get_flux_forecast_error_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Emissions Forecast - Module Load Error", className="mb-4"),
                dbc.Alert([
                    html.H4("Flux Forecast Module Import Failed", className="alert-heading"),
                    html.P(f"Error loading flux_forecast module: {FLUX_FORECAST_ERROR}"),
                    html.Hr(),
                    html.P("This indicates an issue with cross-module imports or data access.")
                ], color="danger")
            ])
        ])
    ])

# Error layout if surface_observations module fails to load
def get_surface_observations_error_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("Surface Observations - Module Load Error", className="mb-4"),
                dbc.Alert([
                    html.H4("Surface Observations Module Import Failed", className="alert-heading"),
                    html.P(f"Error loading surface_observations module: {SURFACE_OBSERVATIONS_ERROR}"),
                    html.Hr(),
                    html.P("This indicates an issue with:"),
                    html.Ul([
                        html.Li("Missing utility modules (conc_func, fig_surface_obs)"),
                        html.Li("HDF5 data file access"),
                        html.Li("Complex dependency chains"),
                        html.Li("Geometry data requirements")
                    ]),
                    html.P("Check the logs for specific import/initialization errors.")
                ], color="danger")
            ])
        ])
    ])

# Error layout if emissions module fails to load
def get_emissions_error_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("OCO-3 Observations - Module Load Error", className="mb-4"),
                dbc.Alert([
                    html.H4("Emissions Module Import Failed", className="alert-heading"),
                    html.P(f"Error loading emissions module: {EMISSIONS_ERROR}"),
                    html.Hr(),
                    html.P("This indicates an issue with:"),
                    html.Ul([
                        html.Li("GCS data access permissions"),
                        html.Li("Missing shapefile dependencies"),
                        html.Li("OCO-3 data file availability"),
                        html.Li("GeoPandas/spatial analysis dependencies")
                    ]),
                    html.P("Check the logs for specific import/initialization errors.")
                ], color="danger"),

                # Show module status
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Module Status Summary:"),
                        html.P(f"• Overview Module: {'✓ Loaded' if OVERVIEW_MODULE_LOADED else '✗ Failed'}"),
                        html.P(f"• Emissions Module: {'✓ Loaded' if EMISSIONS_MODULE_LOADED else '✗ Failed'}"),
                        html.P(f"• Surface Observations Module: {'✓ Loaded' if SURFACE_OBSERVATIONS_MODULE_LOADED else '✗ Failed'}"),
                        html.P(f"• Flux Hindcast Module: {'✓ Loaded' if FLUX_HINDCAST_MODULE_LOADED else '✗ Failed'}"),
                        html.P(f"• Flux Forecast Module: {'✓ Loaded' if FLUX_FORECAST_MODULE_LOADED else '✗ Failed'}"),
                        html.Hr(),
                        html.H6("Error Details:"),
                        html.Pre(EMISSIONS_ERROR or "No error details available",
                                style={'fontSize': '12px', 'backgroundColor': '#f8f9fa', 'padding': '10px'})
                    ])
                ])
            ])
        ])
    ])

# Initialize and register callbacks
def init_app():
    """Initialize all necessary components"""
    try:
        if OVERVIEW_MODULE_LOADED:
            logger.info("Initializing overview module")
            init_overview()
            register_overview_callbacks(app)
            logger.info("Successfully initialized overview module")
        else:
            logger.error(f"Overview module not loaded: {OVERVIEW_ERROR}")

        if FLUX_FORECAST_MODULE_LOADED:
            logger.info("Initializing flux_forecast module")
            init_flux_forecast()
            register_flux_forecast_callbacks(app)
            logger.info("Successfully initialized flux_forecast module")
        else:
            logger.error(f"Flux forecast module not loaded: {FLUX_FORECAST_ERROR}")

        if SURFACE_OBSERVATIONS_MODULE_LOADED:
            logger.info("Initializing surface_observations module")
            init_surface_observations()
            register_surface_observations_callbacks(app)
            logger.info("Successfully initialized surface_observations module")
        else:
            logger.error(f"Surface observations module not loaded: {SURFACE_OBSERVATIONS_ERROR}")
            
        if FLUX_HINDCAST_MODULE_LOADED:
            logger.info("Initializing flux_hindcast module")
            init_flux_hindcast()
            register_flux_hindcast_callbacks(app)
            logger.info("Successfully initialized flux_hindcast module")
        else:
            logger.error(f"Flux hindcast module not loaded: {FLUX_HINDCAST_ERROR}")

        if EMISSIONS_MODULE_LOADED:
            logger.info("Initializing emissions module")
            init_emissions()
            register_emissions_callbacks(app)
            logger.info("Successfully initialized emissions module")
        else:
            logger.error(f"Emissions module not loaded: {EMISSIONS_ERROR}")
    except Exception as e:
        logger.error(f"Error initializing modules: {e}")

# URL redirect callback
@app.callback(
    Output("url", "pathname"),
    Input("url", "pathname")
)
def init_pathname(pathname):
    if pathname == "/" or not pathname:
        return "/page-1"
    return pathname

# Health check endpoint - Updated to include all modules including flux_hindcast
@server.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'phase': 'updated-with-all-modules',
        'overview_module_loaded': OVERVIEW_MODULE_LOADED,
        'emissions_module_loaded': EMISSIONS_MODULE_LOADED,
        'surface_observations_module_loaded': SURFACE_OBSERVATIONS_MODULE_LOADED,
        'flux_forecast_module_loaded': FLUX_FORECAST_MODULE_LOADED,
        'flux_hindcast_module_loaded': FLUX_HINDCAST_MODULE_LOADED,
        'overview_error': OVERVIEW_ERROR,
        'emissions_error': EMISSIONS_ERROR,
        'surface_observations_error': SURFACE_OBSERVATIONS_ERROR,
        'flux_forecast_error': FLUX_FORECAST_ERROR,
        'flux_hindcast_error': FLUX_HINDCAST_ERROR,
        'dependencies_loaded': True
    }, 200

# Initialize the app when module is imported
init_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting LA Megacity Dashboard with all modules on port {port}")
    logger.info(f"Overview module loaded: {OVERVIEW_MODULE_LOADED}")
    logger.info(f"Emissions module loaded: {EMISSIONS_MODULE_LOADED}")
    logger.info(f"Surface observations module loaded: {SURFACE_OBSERVATIONS_MODULE_LOADED}")
    logger.info(f"Flux hindcast module loaded: {FLUX_HINDCAST_MODULE_LOADED}")
    logger.info(f"Flux forecast module loaded: {FLUX_FORECAST_MODULE_LOADED}")

    if OVERVIEW_ERROR:
        logger.error(f"Overview module error: {OVERVIEW_ERROR}")
    if EMISSIONS_ERROR:
        logger.error(f"Emissions module error: {EMISSIONS_ERROR}")
    if SURFACE_OBSERVATIONS_ERROR:
        logger.error(f"Surface observations module error: {SURFACE_OBSERVATIONS_ERROR}")
    if FLUX_HINDCAST_ERROR:
        logger.error(f"Flux hindcast module error: {FLUX_HINDCAST_ERROR}")
    if FLUX_FORECAST_ERROR:
        logger.error(f"Flux forecast module error: {FLUX_FORECAST_ERROR}")

    app.run_server(
        host='0.0.0.0',
        port=port,
        debug=False,
        dev_tools_hot_reload=False,
        dev_tools_ui=False,
        dev_tools_props_check=False
    )