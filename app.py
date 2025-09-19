from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import logging

# Keep the working geospatial libraries
import geopandas as gpd
import fiona
import pyproj
import shapely
from shapely.geometry import Point
import rasterio

# Test scientific data libraries - this is the new critical test
scientific_libs = {}
scientific_errors = {}

try:
    import xarray as xr
    scientific_libs['xarray'] = xr.__version__
except Exception as e:
    scientific_errors['xarray'] = str(e)

try:
    import netCDF4 as nc4
    scientific_libs['netcdf4'] = nc4.__version__
except Exception as e:
    scientific_errors['netcdf4'] = str(e)

try:
    import h5py
    scientific_libs['h5py'] = h5py.__version__
except Exception as e:
    scientific_errors['h5py'] = str(e)

try:
    import tables
    scientific_libs['tables'] = tables.__version__
except Exception as e:
    scientific_errors['tables'] = str(e)

try:
    import scipy
    scientific_libs['scipy'] = scipy.__version__
except Exception as e:
    scientific_errors['scipy'] = str(e)

try:
    import astropy
    scientific_libs['astropy'] = astropy.__version__
except Exception as e:
    scientific_errors['astropy'] = str(e)

try:
    import statsmodels
    scientific_libs['statsmodels'] = statsmodels.__version__
except Exception as e:
    scientific_errors['statsmodels'] = str(e)

try:
    import gcsfs
    scientific_libs['gcsfs'] = gcsfs.__version__
except Exception as e:
    scientific_errors['gcsfs'] = str(e)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Test scientific data functionality
SCIENTIFIC_DATA_READY = False
SCIENTIFIC_DATA_ERROR = None

try:
    if 'xarray' in scientific_libs and 'numpy' in [pkg for pkg in ['numpy']]:
        # Create sample xarray dataset
        time = pd.date_range('2023-01-01', periods=12, freq='M')
        lat = np.linspace(33.5, 34.5, 10)  # LA area latitudes
        lon = np.linspace(-118.8, -117.8, 10)  # LA area longitudes
        
        # Create synthetic CO2 concentration data
        np.random.seed(42)
        data = np.random.normal(415, 5, (12, 10, 10))  # CO2 concentrations around 415 ppm
        
        ds = xr.Dataset({
            'co2_concentration': (['time', 'lat', 'lon'], data)
        }, coords={
            'time': time,
            'lat': lat,
            'lon': lon
        })
        
        SCIENTIFIC_DATA_READY = True
        sample_dataset = ds
except Exception as e:
    SCIENTIFIC_DATA_ERROR = str(e)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("LA Megacity - Phase 3E (Scientific Data)", className="text-center mb-4"),
                html.P("Testing scientific data libraries: xarray, netcdf4, h5py, tables, scipy, astropy, statsmodels, gcsfs", 
                       className="text-center text-muted mb-4")
            ])
        ]),
        
        # Status indicators
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H5("Scientific Libraries Status:", className="mb-2"),
                    html.P(f"Loaded: {len(scientific_libs)} / {len(scientific_libs) + len(scientific_errors)}", className="mb-1"),
                    html.P(f"Data Ready: {SCIENTIFIC_DATA_READY}", className="mb-1"),
                    html.P(f"Errors: {len(scientific_errors)}", className="mb-0")
                ], color="success" if len(scientific_errors) == 0 else "warning")
            ])
        ], className="mb-4"),
        
        dbc.Nav([
            dbc.NavLink("Status", href="/", active="exact"),
            dbc.NavLink("Library Details", href="/details", active="exact"),
            dbc.NavLink("Data Test", href="/datatest", active="exact"),
            dbc.NavLink("Memory Info", href="/memory", active="exact"),
        ], pills=True, className="justify-content-center mb-4"),
        
        html.Div(id="page-content")
    ])
])

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    logger.info(f"Rendering page: {pathname}")
    
    if pathname == "/" or pathname is None:
        return dbc.Card([
            dbc.CardBody([
                html.H3("Scientific Data Libraries Test", className="card-title"),
                html.Hr(),
                html.H5("Successfully Loaded:"),
                html.Div([
                    html.P(f"• {lib}: {version}") for lib, version in scientific_libs.items()
                ] if scientific_libs else [html.P("No libraries loaded successfully")]),
                html.Hr(),
                html.H5("Failed to Load:"),
                html.Div([
                    html.P(f"• {lib}: {error}", className="text-danger") for lib, error in scientific_errors.items()
                ] if scientific_errors else [html.P("All libraries loaded successfully", className="text-success")]),
                html.Hr(),
                html.H5("Geospatial Libraries (from Phase 3D):"),
                html.P("• All geospatial libraries still working", className="text-success"),
                html.Hr(),
                html.P(f"Scientific data processing: {'Working' if SCIENTIFIC_DATA_READY else 'Failed'}")
            ])
        ])
    
    elif pathname == "/details":
        return dbc.Card([
            dbc.CardBody([
                html.H3("Detailed Library Information"),
                
                html.H5("Working Libraries:", className="mt-4"),
                dash_table.DataTable(
                    data=[{"Library": lib, "Version": version, "Status": "Success"} 
                          for lib, version in scientific_libs.items()],
                    columns=[{"name": "Library", "id": "Library"}, 
                            {"name": "Version", "id": "Version"},
                            {"name": "Status", "id": "Status"}],
                    style_cell={'textAlign': 'left'}
                ) if scientific_libs else html.P("No working libraries"),
                
                html.H5("Failed Libraries:", className="mt-4"),
                dash_table.DataTable(
                    data=[{"Library": lib, "Error": error[:100] + "..." if len(error) > 100 else error} 
                          for lib, error in scientific_errors.items()],
                    columns=[{"name": "Library", "id": "Library"}, 
                            {"name": "Error", "id": "Error"}],
                    style_cell={'textAlign': 'left', 'whiteSpace': 'normal'}
                ) if scientific_errors else html.P("No failed libraries", className="text-success")
            ])
        ])
    
    elif pathname == "/datatest":
        if not SCIENTIFIC_DATA_READY:
            return dbc.Alert(f"Scientific data processing failed: {SCIENTIFIC_DATA_ERROR}", color="danger")
        
        # Show xarray dataset info
        return dbc.Card([
            dbc.CardBody([
                html.H3("XArray Dataset Test"),
                html.P(f"Dataset dimensions: {dict(sample_dataset.dims)}"),
                html.P(f"Data variables: {list(sample_dataset.data_vars)}"),
                html.P(f"Coordinates: {list(sample_dataset.coords)}"),
                html.Hr(),
                html.H5("Sample CO2 Data (first time slice):"),
                dash_table.DataTable(
                    data=sample_dataset.co2_concentration.isel(time=0).to_pandas().reset_index().head(20).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in ['lat', 'lon', 'co2_concentration']],
                    style_cell={'textAlign': 'left'}
                ),
                html.Hr(),
                html.P(f"Mean CO2 concentration: {float(sample_dataset.co2_concentration.mean()):.2f} ppm"),
                html.P(f"Dataset size: {sample_dataset.nbytes / 1024:.1f} KB")
            ])
        ])
    
    elif pathname == "/memory":
        import psutil
        import gc
        
        # Get memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return dbc.Card([
            dbc.CardBody([
                html.H3("Memory Usage Information"),
                html.P(f"RSS Memory: {memory_info.rss / 1024 / 1024:.1f} MB"),
                html.P(f"VMS Memory: {memory_info.vms / 1024 / 1024:.1f} MB"),
                html.P(f"Available System Memory: {psutil.virtual_memory().available / 1024 / 1024:.1f} MB"),
                html.P(f"Memory Usage %: {psutil.virtual_memory().percent:.1f}%"),
                html.Hr(),
                html.P(f"Loaded scientific libraries: {len(scientific_libs)}"),
                html.P(f"Failed libraries: {len(scientific_errors)}"),
                html.Hr(),
                html.Small("This helps identify if memory issues are causing import failures")
            ])
        ])
    
    else:
        return dbc.Alert(f"Page '{pathname}' not found", color="danger")

@server.route('/health')
def health():
    return {
        'status': 'healthy', 
        'phase': 'phase-3e-scientific',
        'scientific_libs_loaded': len(scientific_libs),
        'scientific_libs_failed': len(scientific_errors),
        'scientific_data_ready': SCIENTIFIC_DATA_READY,
        'failed_libraries': list(scientific_errors.keys())
    }, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Phase 3E scientific libraries test on port {port}")
    logger.info(f"Scientific libraries loaded: {len(scientific_libs)}")
    logger.info(f"Scientific libraries failed: {len(scientific_errors)}")
    if scientific_errors:
        logger.error(f"Failed libraries: {list(scientific_errors.keys())}")
    app.run_server(host='0.0.0.0', port=port, debug=False)
