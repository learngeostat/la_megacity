from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import dash
import sys
import os
import logging
logging.getLogger("fiona").disabled = True

# Force socket to use a standard hostname
#socket.gethostbyname = lambda x: '127.0.0.1'

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the local modules
from la_megacity.pages.emissions import init as init_emissions
from la_megacity.pages.emissions import get_layout as get_emissions_layout
from la_megacity.pages.emissions import register_callbacks as register_emissions_callbacks

# Import surface observations module
from la_megacity.pages.surface_observations import init as init_surface_observations
from la_megacity.pages.surface_observations import get_layout as get_surface_observations_layout
from la_megacity.pages.surface_observations import register_callbacks as register_surface_observations_callbacks

# Import flux hindcast/nowcast module
from la_megacity.pages.flux_hindcast import init as init_flux_hindcast
from la_megacity.pages.flux_hindcast import get_layout as get_flux_hindcast_layout
from la_megacity.pages.flux_hindcast import register_callbacks as register_flux_hindcast_callbacks

# Import flux forecast module
from la_megacity.pages.flux_forecast import init as init_flux_forecast
from la_megacity.pages.flux_forecast import get_layout as get_flux_forecast_layout
from la_megacity.pages.flux_forecast import register_callbacks as register_flux_forecast_callbacks

from la_megacity.pages.overview import init as init_overview

from la_megacity.pages.overview import get_layout as get_overview_layout
from la_megacity.pages.overview import register_callbacks as register_overview_callbacks

# Rest of your app.py code remains the same...

# Initialize the app with both Bootstrap and Font Awesome
external_stylesheets = [
    dbc.themes.LUX,
    "href=https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"
]

app = dash.Dash(__name__, 
                external_stylesheets=external_stylesheets, 
                use_pages=False,
                pages_folder="",
                suppress_callback_exceptions=True)  # Add this line

# Main app layout
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

# Update the CSS styles
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
    if pathname == "/page-1":
        return get_overview_layout()
    elif pathname == "/page-2":
        return get_emissions_layout()
    elif pathname == "/page-3":
        return get_surface_observations_layout()
    elif pathname == "/page-4":
        return get_flux_hindcast_layout()  # New hindcast/nowcast layout
    elif pathname == "/page-5":
        return get_flux_forecast_layout()  # New forecast layout
    else:
        # If the user tries to reach a different page, return a 404 message
        return html.Div([
            html.H1("404: Not found", className="text-danger"),
            html.P(f"The pathname {pathname} was not recognised...")
        ])

def init_app():
    """Initialize all necessary components"""
    try:
        # Initialize emissions page
        # Initialize overview page
        init_overview()
        register_overview_callbacks(app)
        
        init_emissions()
        register_emissions_callbacks(app)
        
        # Initialize surface observations page
        init_surface_observations()
        register_surface_observations_callbacks(app)
        
        # Initialize flux analysis pages
        init_flux_hindcast()
        register_flux_hindcast_callbacks(app)
        
        init_flux_forecast()
        register_flux_forecast_callbacks(app)
        
        print("Successfully initialized all components")
    except Exception as e:
        print(f"Error initializing components: {e}")

@app.callback(
    Output("url", "pathname"),
    Input("url", "pathname")
)
def init_pathname(pathname):
    if pathname == "/" or not pathname:
        return "/page-1"  # Changed from "/page-2" to land on overview page
    return pathname

# if __name__ == '__main__':
#     init_app()  # Initialize all components
#     app.server.config['SERVER_NAME'] = 'localhost:8050'
#     app.run_server(debug=True, port=8050)


if __name__ == '__main__':
    init_app()  # Initialize all components
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # # Get direct access to the underlying Flask server
    # flask_app = app.server
    
    # # Run Flask with debug=True but disable the auto-reloader
    # flask_app.run(host='localhost', port=8050, debug=True, use_reloader=False)
    
    app.run_server(debug=True, port=8058)
