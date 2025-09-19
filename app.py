from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import logging

# Geospatial imports - this is the critical test
try:
    import geopandas as gpd
    import fiona
    import pyproj
    import shapely
    from shapely.geometry import Point, Polygon
    import rasterio
    GEOSPATIAL_LOADED = True
    GEOSPATIAL_ERROR = None
except Exception as e:
    GEOSPATIAL_LOADED = False
    GEOSPATIAL_ERROR = str(e)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Test geospatial data if libraries loaded successfully
if GEOSPATIAL_LOADED:
    try:
        # Create sample geospatial data for LA area
        la_cities = pd.DataFrame({
            'city': ['Los Angeles', 'Long Beach', 'Anaheim', 'Santa Ana', 'Riverside'],
            'population': [3900000, 462000, 346000, 334000, 314000],
            'co2_emissions': [45.2, 8.1, 6.2, 5.8, 5.4],
            'latitude': [34.0522, 33.7701, 33.8366, 33.7455, 33.9533],
            'longitude': [-118.2437, -118.1937, -117.9143, -117.8677, -117.3962]
        })
        
        # Convert to GeoDataFrame
        geometry = [Point(xy) for xy in zip(la_cities.longitude, la_cities.latitude)]
        gdf = gpd.GeoDataFrame(la_cities, geometry=geometry, crs='EPSG:4326')
        
        # Test coordinate transformation
        gdf_utm = gdf.to_crs('EPSG:32611')  # UTM Zone 11N for Southern California
        
        GEODATA_READY = True
        GEODATA_ERROR = None
    except Exception as e:
        GEODATA_READY = False
        GEODATA_ERROR = str(e)
else:
    GEODATA_READY = False
    GEODATA_ERROR = "Geospatial libraries not loaded"

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("LA Megacity - Phase 3D (Geospatial)", className="text-center mb-4"),
                html.P("Testing geospatial libraries: geopandas, fiona, pyproj, shapely, rasterio", 
                       className="text-center text-muted mb-4")
            ])
        ]),
        
        # Status indicators
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H5("Geospatial Libraries Status:", className="mb-2"),
                    html.P(f"✅ Loaded: {GEOSPATIAL_LOADED}", className="mb-1"),
                    html.P(f"✅ Data Ready: {GEODATA_READY}", className="mb-1"),
                    html.P(f"Error: {GEOSPATIAL_ERROR or GEODATA_ERROR or 'None'}", className="mb-0")
                ], color="success" if GEOSPATIAL_LOADED and GEODATA_READY else "danger")
            ])
        ], className="mb-4"),
        
        dbc.Nav([
            dbc.NavLink("Status", href="/", active="exact"),
            dbc.NavLink("Library Info", href="/info", active="exact"),
            dbc.NavLink("Geo Data", href="/geodata", active="exact"),
            dbc.NavLink("Map Test", href="/map", active="exact"),
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
                html.H3("Geospatial Libraries Test", className="card-title"),
                html.Hr(),
                html.H5("Import Status:"),
                html.P(f"• geopandas: {'✅ Success' if GEOSPATIAL_LOADED else '❌ Failed'}"),
                html.P(f"• fiona: {'✅ Success' if GEOSPATIAL_LOADED else '❌ Failed'}"),
                html.P(f"• pyproj: {'✅ Success' if GEOSPATIAL_LOADED else '❌ Failed'}"),
                html.P(f"• shapely: {'✅ Success' if GEOSPATIAL_LOADED else '❌ Failed'}"),
                html.P(f"• rasterio: {'✅ Success' if GEOSPATIAL_LOADED else '❌ Failed'}"),
                html.Hr(),
                html.H5("Data Processing:"),
                html.P(f"• Coordinate systems: {'✅ Working' if GEODATA_READY else '❌ Failed'}"),
                html.P(f"• Geometry creation: {'✅ Working' if GEODATA_READY else '❌ Failed'}"),
                html.Hr(),
                html.P(f"Error details: {GEOSPATIAL_ERROR or GEODATA_ERROR or 'No errors'}", 
                       className="text-muted small")
            ])
        ])
    
    elif pathname == "/info":
        if not GEOSPATIAL_LOADED:
            return dbc.Alert("Geospatial libraries failed to load", color="danger")
        
        return dbc.Card([
            dbc.CardBody([
                html.H3("Library Information"),
                html.P(f"GeoPandas version: {gpd.__version__}"),
                html.P(f"Fiona version: {fiona.__version__}"),
                html.P(f"PyProj version: {pyproj.__version__}"),
                html.P(f"Shapely version: {shapely.__version__}"),
                html.P(f"Rasterio version: {rasterio.__version__}"),
                html.Hr(),
                html.H5("Available CRS Systems:"),
                html.P("Testing EPSG:4326 (WGS84) and EPSG:32611 (UTM 11N)")
            ])
        ])
    
    elif pathname == "/geodata":
        if not GEODATA_READY:
            return dbc.Alert(f"Geodata processing failed: {GEODATA_ERROR}", color="danger")
        
        # Show the geodataframe as a table
        display_df = gdf.drop('geometry', axis=1)  # Remove geometry column for display
        
        return dbc.Card([
            dbc.CardBody([
                html.H3("LA Area Cities - Geospatial Data"),
                dash_table.DataTable(
                    data=display_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in display_df.columns],
                    style_cell={'textAlign': 'left'}
                ),
                html.Hr(),
                html.P(f"Original CRS: {gdf.crs}"),
                html.P(f"Transformed CRS: {gdf_utm.crs}"),
                html.P(f"Geometry type: {type(gdf.geometry.iloc[0]).__name__}")
            ])
        ])
    
    elif pathname == "/map":
        if not GEODATA_READY:
            return dbc.Alert(f"Map data not available: {GEODATA_ERROR}", color="danger")
        
        # Create a simple scatter plot on map using plotly
        fig = px.scatter_mapbox(
            la_cities, 
            lat="latitude", 
            lon="longitude",
            size="population",
            color="co2_emissions",
            hover_name="city",
            hover_data=["population", "co2_emissions"],
            color_continuous_scale="Reds",
            title="LA Area Cities - Population and CO2 Emissions",
            mapbox_style="open-street-map",
            zoom=8,
            height=600
        )
        
        return dbc.Card([
            dbc.CardBody([
                html.H3("Geospatial Visualization Test"),
                dcc.Graph(figure=fig)
            ])
        ])
    
    else:
        return dbc.Alert(f"Page '{pathname}' not found", color="danger")

@server.route('/health')
def health():
    return {
        'status': 'healthy', 
        'phase': 'phase-3d-geospatial',
        'geospatial_loaded': GEOSPATIAL_LOADED,
        'geodata_ready': GEODATA_READY,
        'error': GEOSPATIAL_ERROR or GEODATA_ERROR
    }, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Phase 3D geospatial test on port {port}")
    logger.info(f"Geospatial libraries loaded: {GEOSPATIAL_LOADED}")
    if not GEOSPATIAL_LOADED:
        logger.error(f"Geospatial import error: {GEOSPATIAL_ERROR}")
    app.run_server(host='0.0.0.0', port=port, debug=False)
