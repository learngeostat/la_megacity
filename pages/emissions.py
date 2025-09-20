"""
Created on Sat Dec 14 16:02:48 2024
Emissions analysis page for the LA Megacity Dashboard
"""

from dash import html, dcc, Input, Output, State, callback_context as ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import geopandas as gpd
import os
import math
import dash

import gcsfs
import zipfile
import io
import tempfile

#import shutil




# Modified import statement
#from la_megacity.utils.constants import SHAPEFILE_PATH, OCO3_DATA_PATH  # Assuming the constants are in this path
import logging
logging.getLogger("fiona").disabled = True


# These would be defined at the top of your file
SHAPEFILE_PATH = "gs://la-megacity-dashboard-data-1/data/shapefiles/"
OCO3_DATA_PATH = "gs://la-megacity-dashboard-data-1/data/csv/"


# Global variables for data storage
census_tracts = None
zip_codes = None
data = None
socab_boundary = None
available_dates = []

# Add after existing imports
# Constants for animation
MIN_ANIMATION_SPEED = 600  # milliseconds
MAX_ANIMATION_SPEED = 5000  # milliseconds
DEFAULT_ANIMATION_SPEED = 1000  # milliseconds

def create_restore_button():
    """Create the restore button"""
    return html.Button(
        "Restore Panels", 
        id="restore-button", 
        className="restore-button",
        style={'display': 'none'}
    )


def update_map_section(map_section):
    """Wrap the map section in a column with expansion classes"""
    return dbc.Col(
        map_section,
        id="left-panel",
        width=6,
        className="panel-transition pe-1",
        style={'display': 'block'}  # Ensure block display
    )

def create_expansion_controls():
    """Create the expansion control column"""
    return dbc.Col([
        html.Div([
            html.Button("→", id="expand-left", className="expand-button"),
            html.Button("←", id="expand-right", className="expand-button")
        ], className="expand-control")
    ], width="auto", className="px-0", style={'display': 'block'})

def update_time_series_section(time_series_section):
    """Wrap the time series section in a column with expansion classes"""
    return dbc.Col(
        time_series_section,
        id="right-panel",
        width=6,
        className="panel-transition ps-1",
        style={'display': 'block'}  # Ensure block display
    )


# Add after existing functions
def load_geodata():
    """Load geographical data with proper error handling"""
    try:
        census_tracts = gpd.read_file(os.path.join(SHAPEFILE_PATH, 'census_tract_clipped.shp'))
        zip_codes = gpd.read_file(os.path.join(SHAPEFILE_PATH, 'zip_code_socab.shp'))
        socab_boundary = gpd.read_file(os.path.join(SHAPEFILE_PATH, 'socabbound.shp'))
        return census_tracts, zip_codes, socab_boundary
    except Exception:
        return None, None
    
def get_years_and_dates(dates):
    """Get unique years and organize dates"""    
    # Organize dates by year
    date_by_year = {}
    for date_str in dates:
        year = date_str.split('-')[0]
        if year not in date_by_year:
            date_by_year[year] = []
        date_by_year[year].append(date_str)
    
    # Create year options in ascending order
    year_options = [{'label': year, 'value': year} 
                   for year in sorted(date_by_year.keys())]
    
    return year_options, date_by_year

def load_oco3_data():
    """Load OCO-3 data from GCS with error handling"""
    try:
        # Construct the full GCS path to the CSV file
        csv_path = os.path.join(OCO3_DATA_PATH, 'clipped_oco3_obs.csv')
        
        # Read the CSV directly from GCS using pandas
        df = pd.read_csv(csv_path,
                        dtype={
                            'year': int,
                            'month': int,
                            'day': int,
                            'formatted_date': int,
                            'longitude': float,
                            'latitude': float,
                            'xco2': float
                        },
                        low_memory=False)
        
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']]).dt.strftime('%Y-%m-%d')
        return df
    except Exception as e:
        # If the file is not found or another error occurs, return an empty DataFrame
        print(f"Error loading OCO-3 data from GCS: {e}")
        return pd.DataFrame()

def aggregate_by_polygon(points_df, polygons_gdf):
    """Aggregate OCO-3 observations within polygons"""
    points_gdf = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df.longitude, points_df.latitude),
        crs=polygons_gdf.crs
    )
    joined = gpd.sjoin(points_gdf, polygons_gdf, how='inner', predicate='within')
    return joined.groupby('index_right')['xco2'].mean().reset_index()

def create_analysis_frame(date, view_type, selected_zip, stat_type, view_state=None, scale_type='variable', scale_min=400, scale_max=425):
    """Generate analysis view with map and time series"""
    map_fig = go.Figure()
    time_series_fig = go.Figure()
    title = "No data available"
    y_label = "XCO2 (ppm)"

    # Colorbar settings
    colorbar_settings = dict(
        orientation='v',
        thickness=20,
        len=0.9,
        y=0.5,
        yanchor='middle',
        x=1.02,
        xanchor='left',
        title=dict(
            text='XCO₂ (ppm)',
            side='right'
        ),
        tickfont=dict(size=10)
    )

    # Calculate bounds for initial view
    if not view_state:
        bounds = socab_boundary.geometry.total_bounds
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        
        # Calculate zoom level to fit the geometry
        lon_range = bounds[2] - bounds[0]
        lat_range = bounds[3] - bounds[1]
        zoom = min(
            math.log2(360 / lon_range) - 1,
            math.log2(180 / lat_range) - 1
        )
        zoom = max(min(zoom, 10), 7)  # Constrain zoom between 7 and 10
    else:
        center_lat = view_state['center']['lat']
        center_lon = view_state['center']['lon']
        zoom = view_state['zoom']
    
    # Add static boundary trace - positioned at southwest corner
    boundary_trace = go.Choroplethmapbox(
        geojson=socab_boundary.__geo_interface__,
        locations=socab_boundary.index,
        z=[1] * len(socab_boundary),
        colorscale=[[0, 'rgba(169, 169, 169, 0.4)'], [1, 'rgba(169, 169, 169, 0.4)']],
        showscale=False,
        marker=dict(
            line=dict(
                width=1,
                color='black'
            )
        ),
        hoverinfo='skip',
        name='LA Megacity Extent',
        showlegend=True
    )
    map_fig.add_trace(boundary_trace)

    # Base layouts with southwest legend position
    map_layout = dict(
        autosize=True,
        mapbox=dict(
            style='open-street-map',
            zoom=zoom,
            center=dict(lat=center_lat, lon=center_lon)
        ),
        margin=dict(l=0, r=70, t=0, b=0),
        uirevision='constant',
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.02,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
            font=dict(size=10),
            itemsizing='constant',
            orientation='h'
        )
    )

    time_series_layout = dict(
        autosize=True,
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        margin=dict(l=50, r=30, t=40, b=40),
        uirevision='constant',
        paper_bgcolor='white',
        plot_bgcolor='rgba(240, 242, 245, 0.8)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            gridwidth=1,
            showline=True,
            linewidth=1,
            linecolor='rgba(128, 128, 128, 0.8)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            gridwidth=1,
            showline=True,
            linewidth=1,
            linecolor='rgba(128, 128, 128, 0.8)'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Apply initial layouts
    map_fig.update_layout(map_layout)
    time_series_fig.update_layout(time_series_layout)

    if not date:
        return map_fig, time_series_fig

    # Filter data for selected date
    filtered_data = data[data['date'] == date]

    # Add traces based on view type
    if view_type == 'native':
        # Create marker settings with conditional scale
        marker_settings = dict(
            size=8,
            color=filtered_data['xco2'],
            colorscale='Turbo',
            showscale=True,
            colorbar=colorbar_settings,
            opacity=0.8
        )
        
        # Add fixed scale if selected
        if scale_type == 'fixed':
            marker_settings.update(dict(
                cmin=scale_min,
                cmax=scale_max
            ))
        
        map_fig.add_trace(go.Scattermapbox(
            lat=filtered_data['latitude'],
            lon=filtered_data['longitude'],
            mode='markers',
            marker=marker_settings,
            text=filtered_data['xco2'].round(2).astype(str),
            hovertemplate="XCO2: %{text} ppm<extra></extra>",
            showlegend=False
        ))
        
        # Time series for native view
        daily_stats = data.groupby('date').agg({
            'xco2': ['mean', 'std']
        }).reset_index()
        daily_stats.columns = ['date', 'mean', 'std']
        
        y_values = daily_stats['std'] if stat_type == 'std' else daily_stats['mean']
        y_label = 'XCO2 Standard Deviation (ppm)' if stat_type == 'std' else 'Mean XCO2 (ppm)'
        
        time_series_fig.add_trace(go.Scatter(
            x=daily_stats['date'],
            y=y_values,
            mode='lines+markers',
            name=f'Basin-wide {stat_type}'
        ))
        title = f"Basin-wide Daily {stat_type.capitalize()} XCO2"

    else:  # ZIP view
        aggregated = aggregate_by_polygon(filtered_data, zip_codes)
        aggregated = pd.merge(
            aggregated,
            zip_codes[['ZIP_CODE', 'PO_NAME']],
            left_on='index_right',
            right_index=True,
            how='left'
        )
        
        hover_text = [
            f"ZIP Code: {zip_code}<br>" +
            f"Post Office: {po_name}<br>" +
            f"Mean XCO2: {xco2:.2f} ppm"
            for zip_code, po_name, xco2 in 
            zip(aggregated['ZIP_CODE'], aggregated['PO_NAME'], aggregated['xco2'])
        ]
        
        # Create choropleth settings with conditional scale
        choropleth_settings = dict(
            geojson=zip_codes.__geo_interface__,
            locations=aggregated['index_right'],
            z=aggregated['xco2'],
            colorscale='Turbo',
            marker_opacity=0.7,
            showscale=True,
            colorbar=colorbar_settings,
            hovertemplate="%{text}<extra></extra>",
            text=hover_text
        )
        
        # Add fixed scale if selected
        if scale_type == 'fixed':
            choropleth_settings.update(dict(
                zmin=scale_min,
                zmax=scale_max
            ))
        
        map_fig.add_trace(go.Choroplethmapbox(**choropleth_settings))

        # Time series for ZIP view
        if selected_zip:
            points_gdf = gpd.GeoDataFrame(
                data,
                geometry=gpd.points_from_xy(data.longitude, data.latitude),
                crs=zip_codes.crs
            )
            zip_data = gpd.sjoin(points_gdf, zip_codes, how='inner', predicate='within')
            zip_data = zip_data[zip_data['ZIP_CODE'] == selected_zip]
        
            if len(zip_data) > 0:
                daily_stats = zip_data.groupby('date').agg({
                    'xco2': ['mean', 'std']
                }).reset_index()
                daily_stats.columns = ['date', 'mean', 'std']
                
                y_values = daily_stats['std'] if stat_type == 'std' else daily_stats['mean']
                y_label = 'XCO2 Standard Deviation (ppm)' if stat_type == 'std' else 'Mean XCO2 (ppm)'
                
                time_series_fig.add_trace(go.Scatter(
                    x=daily_stats['date'],
                    y=y_values,
                    mode='lines+markers',
                    name=f'ZIP {selected_zip} {stat_type}'
                ))
                title = f"ZIP Code {selected_zip} Daily {stat_type.capitalize()} XCO2"
            else:
                title = f"No data available for ZIP Code {selected_zip}"
        else:
            title = "Click on a ZIP code to view its time series"

    # Final layout updates
    time_series_fig.update_layout(
        title=title,
        yaxis_title=y_label
    )

    # Ensure legend stays in southwest position
    map_fig.update_layout(
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.02,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
            font=dict(size=10),
            itemsizing='constant',
            orientation='h'
        )
    )

    return map_fig, time_series_fig

def get_initial_view_state():
    """Get default view state from boundary"""
    bounds = socab_boundary.geometry.total_bounds
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    
    # Calculate zoom level to fit the geometry
    lon_range = bounds[2] - bounds[0]
    lat_range = bounds[3] - bounds[1]
    zoom = min(
        math.log2(360 / lon_range) - 1,
        math.log2(180 / lat_range) - 1
    )
    zoom = max(min(zoom, 10), 7)  # Constrain zoom between 7 and 10
    
    return {
        'center': {'lat': center_lat, 'lon': center_lon},
        'zoom': zoom
    }


def create_header_card():
    """Create the header card with text followed by icons"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [
                            html.I(className="fas fa-industry me-2"),
                            "OCO-3 XCO₂ Snapshot Area Mode",
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
                        ], color="secondary", className="me-2", id="oco3-doc-button"),
                        dbc.Button([
                            "Help ",
                            html.I(className="fas fa-question-circle")
                        ], color="secondary", className="me-2", id="oco3-help-button"),
                        dbc.Button([
                            "Reset ",
                            html.I(className="fas fa-redo")
                        ], color="secondary", id="oco3-restart-button")
                    ], className="d-flex justify-content-end")
                ], width=2)
            ], className="align-items-center")
        ]),
        dbc.Collapse(
            dbc.CardBody([
                html.P([
                    "Analyze XCO₂ concentrations across the Los Angeles region. ",
                    "Explore spatial distribution and temporal patterns."
                ], className="text-muted mb-0")
            ]),
            id="header-collapse",
            is_open=True,
        )
    ], className="mb-4 shadow-sm")

def create_analysis_controls_row1(year_options, initial_year, initial_date_options, initial_dates):
    """Create the first row of analysis controls"""
    return dbc.Row([
        # Year Selection
        dbc.Col([
            html.Label("Select Year", className="form-label fw-bold"),
            dcc.Dropdown(
                id='oco3-year-picker',
                options=year_options,
                value=initial_year,
                clearable=False,
            )
        ], md=3),
        
        # Date Selection
        dbc.Col([
            html.Label("Select Date", className="form-label fw-bold"),
            dcc.Dropdown(
                id='oco3-date-picker',
                options=initial_date_options,
                value=initial_dates[0] if initial_dates else None,
                clearable=False,
            )
        ], md=3),
        
        # View Selection
        dbc.Col([
            html.Label("View Type", className="form-label fw-bold"),
            dcc.Dropdown(
                id='oco3-sam-view-toggle',
                options=[
                    {'label': 'Observations', 'value': 'native'},
                    {'label': 'ZIP Code Aggregation', 'value': 'zip'}
                ],
                value='native',
                clearable=False,
            )
        ], md=3),
        
        # Analysis Metric
        dbc.Col([
            html.Label("Analysis Metric", className="form-label fw-bold"),
            dcc.Dropdown(
                id='oco3-sam-stat-toggle',
                options=[
                    {'label': 'Mean XCO₂', 'value': 'mean'},
                    {'label': 'Standard Deviation', 'value': 'std'}
                ],
                value='mean',
                clearable=False,
            )
        ], md=3)
    ])

def create_analysis_controls_row2():
    """Create the second row of analysis controls"""
    # Define common input style to match dropdown height
    input_style = {"height": "36px"}  # Standard dropdown height in most browsers
    
    return dbc.Row([
        # Animation Speed Column
        dbc.Col([
            html.Label("Animation Speed (ms)", className="form-label fw-bold"),
            dbc.Input(
                id="animation-speed",
                type="number",
                min=MIN_ANIMATION_SPEED,
                max=MAX_ANIMATION_SPEED,
                step=100,
                value=DEFAULT_ANIMATION_SPEED,
                style=input_style,
                persistence=True,  # Add persistence
                persistence_type="session",  # Store in session
                inputMode="numeric",  # Explicitly set input mode
            )
        ], md=3),
        
        # Scale Controls Group
        dbc.Col([
            dbc.Row([
                # Scale Type Dropdown
                dbc.Col([
                    html.Label("Color Scale", className="form-label fw-bold"),
                    dcc.Dropdown(
                        id="scale-type",
                        options=[
                            {"label": "Variable Scale", "value": "variable"},
                            {"label": "Fixed Scale", "value": "fixed"}
                        ],
                        value="variable",
                        clearable=False
                    )
                ], width=4),
                
                # Min Scale Input
                dbc.Col([
                    html.Label("Min Scale (ppm)", className="form-label fw-bold"),
                    dbc.Input(
                        id="scale-min",
                        type="number",
                        min=300,
                        max=600,
                        value=400,
                        disabled=True,
                        style=input_style
                    )
                ], width=4),
                
                # Max Scale Input
                dbc.Col([
                    html.Label("Max Scale (ppm)", className="form-label fw-bold"),
                    dbc.Input(
                        id="scale-max",
                        type="number",
                        min=300,
                        max=600,
                        value=425,
                        disabled=True,
                        style=input_style
                    )
                ], width=4)
            ])
        ], md=9)
    ], className="mt-3")

def create_map_section():
    """Create the map visualization section with contained time controls"""
    return dbc.Card([
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
                       dbc.Button(
                           [
                               html.I(className="fas fa-download me-2"),
                               "DATA"
                           ],
                           id="download-spatial-data",
                           color="light",
                           size="sm",
                           className="float-end",
                           title="Download spatial data including shapefiles"
                       ),
                       dcc.Download(id="download-spatial-data-content")
                   ], width=2),
               ], className="align-items-center"),
           ]),
        dbc.Collapse(
            dbc.CardBody([
                html.Div([
                    # Time Control Overlay - Made more compact
                    html.Div([
                        dbc.Row([
                            # Play/Pause Controls
                            dbc.Col([
                                dbc.ButtonGroup([
                                    dbc.Button(
                                        html.I(className="fas fa-play"),
                                        id="play-button",
                                        color="light",
                                        size="sm",
                                        className="me-1",
                                        disabled=False  # Add initial disabled state
                                    ),
                                    dbc.Button(
                                        html.I(className="fas fa-pause"),
                                        id="pause-button",
                                        color="light",
                                        size="sm",
                                        disabled=False  # Add initial disabled state
                                    )
                                ])
                            ], width="auto", className="pe-2"),
                            
                            # Time Slider Column with Date Display
                            dbc.Col([
                                # Date Display Above Slider
                                html.Div(
                                    id="slider-date-display",
                                    className="text-center fw-bold mb-2"
                                ),
                                # Time Slider
                                dcc.Slider(
                                    id='time-slider',
                                    min=0,
                                    max=100,
                                    step=1,
                                    value=0,
                                    marks=None,
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    disabled=False  # Add initial disabled state
                                )
                            ], className="px-0")
                        ], className="g-0 align-items-center mx-2")
                    ], className="time-control-overlay"),
                    
                    # Main Map
                    dcc.Graph(
                        id='oco3-map-plot',
                        config={
                            'modeBarButtonsToRemove': ['autoScale2d', 'select2d', 'lasso2d'],
                            'displaylogo': False,
                            'scrollZoom': True
                        },
                        style={'height': '65vh'}
                    )
                ], className="position-relative")
            ]),
            id="map-collapse",
            is_open=True,
        )
    ])

def create_time_series_section():
    """Create the time series visualization section"""
    return dbc.Card([
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
                           dbc.Button(
                               [
                                   html.I(className="fas fa-download me-2"),
                                   "DATA"
                               ],
                               id="download-timeseries-data",
                               color="light",
                               size="sm",
                               className="float-end",
                               title="Download time series data as CSV"
                           ),
                           dcc.Download(id="download-timeseries-data-content")
                       ], width=2),
                   ], className="align-items-center"),
               ]),
        dbc.Collapse(
            dbc.CardBody([
                dcc.Graph(
                    id='oco3-time-series',
                    style={'height': '65vh'}
                )
            ]),
            id="timeseries-collapse",
            is_open=True,
        )
    ])

def create_analysis_controls_card(year_options, initial_year, initial_date_options, initial_dates):
    """Create the analysis controls card with two rows"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Button(
                [
                    html.I(className="fas fa-sliders-h me-2"),
                    "Analysis Controls",
                    html.I(className="fas fa-chevron-down ms-2", id="analysis-controls-chevron"),
                ],
                color="link",
                id="analysis-controls-toggle",
                className="text-primary fs-4 text-decoration-none p-0 w-100 text-start",
                style={"box-shadow": "none"}
            )
        ]),
        dbc.Collapse(
            dbc.CardBody([
                create_analysis_controls_row1(year_options, initial_year, initial_date_options, initial_dates),
                create_analysis_controls_row2()
            ]),
            id="analysis-controls-collapse",
            is_open=True,
        )
    ], className="mb-4 shadow-sm")

def create_help_modal():
    """Create the help documentation modal"""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("OCO-3 Dashboard Help")),
            dbc.ModalBody([
                html.H5("Overview"),
                html.P("This dashboard visualizes OCO-3 XCO₂ observations across the Los Angeles region, allowing you to explore both spatial distributions and temporal patterns of carbon dioxide concentrations."),
                
                html.H5("Analysis Controls"),
                html.H6("Basic Controls:"),
                html.Ul([
                    html.Li([
                        html.Strong("Select Year & Date: "),
                        "Choose the time period for analysis. The date picker updates based on available data for the selected year."
                    ]),
                    html.Li([
                        html.Strong("View Type: "),
                        "Toggle between 'Observations' (individual data points) and 'ZIP Code Aggregation' (averaged by postal code)."
                    ]),
                    html.Li([
                        html.Strong("Analysis Metric: "),
                        "Choose between mean XCO₂ values or standard deviation for temporal analysis."
                    ])
                ]),
                
                html.H6("Advanced Controls:"),
                html.Ul([
                    html.Li([
                        html.Strong("Animation Speed: "),
                        "Control the playback speed (600-3000ms) when animating through dates. Only available in Observations view."
                    ]),
                    html.Li([
                        html.Strong("Color Scale: "),
                        "Choose between variable scale (auto-adjusted to data range) or fixed scale (user-defined range)."
                    ]),
                    html.Li([
                        html.Strong("Scale Range: "),
                        "When using fixed scale, set minimum and maximum XCO₂ values (ppm) for consistent color mapping."
                    ])
                ]),
                
                html.H5("Map Visualization"),
                html.Ul([
                    html.Li([
                        html.Strong("Navigation: "),
                        "Pan by dragging, zoom with mouse wheel or touchpad gestures. Double-click to reset view."
                    ]),
                    html.Li([
                        html.Strong("Animation Controls: "),
                        "Use play/pause buttons and slider to animate through available dates (Observations view only)."
                    ]),
                    html.Li([
                        html.Strong("Interaction: "),
                        "In ZIP Code view, click on areas to see their temporal patterns in the time series plot."
                    ])
                ]),
                
                html.H5("Time Series Plot"),
                html.Ul([
                    html.Li([
                        html.Strong("Basin-wide View: "),
                        "Shows overall temporal patterns across the entire region."
                    ]),
                    html.Li([
                        html.Strong("ZIP Code View: "),
                        "Displays temporal patterns for selected ZIP code areas when in ZIP Code view mode."
                    ]),
                    html.Li([
                        html.Strong("Interaction: "),
                        "Hover over points for detailed values, zoom by selecting regions, double-click to reset view."
                    ])
                ]),
                
                html.H5("Additional Features"),
                html.Ul([
                    html.Li([
                        html.Strong("Restart: "),
                        "Use the restart button to reset all controls to their default values."
                    ]),
                    html.Li([
                        html.Strong("Collapsible Sections: "),
                        "Click section headers to show/hide different components of the dashboard."
                    ])
                ])
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="help-close", className="ms-auto")
            ),
        ],
        id="help-modal",
        size="lg",
        is_open=False,
    )

# Add this to your callback registrations
def register_help_callback(app):
    @app.callback(
        Output("help-modal", "is_open"),
        [Input("oco3-help-button", "n_clicks"), 
         Input("help-close", "n_clicks")],
        [State("help-modal", "is_open")],
    )
    def toggle_help_modal(n1, n2, is_open):
        if n1 or n2:
            return not is_open
        return is_open

# Add the modal to your layout
# In the get_layout() function, add create_help_modal() to the Container children

# Modified get_layout function
# def get_layout():
#     """Return the page layout"""
#     year_options, dates_by_year = get_years_and_dates(available_dates)
#     initial_year = year_options[0]['value'] if year_options else None
#     initial_dates = sorted(dates_by_year.get(initial_year, []), reverse=True) if initial_year else []
#     initial_date_options = [{'label': date, 'value': date} for date in initial_dates]
    
#     return dbc.Container([
#         # Help Modal and stores remain the same
#         create_help_modal(),
#         dcc.Store(id="animation-state"),
#         dcc.Store(id="current-frame", data=0),
#         dcc.Store(id='oco3-map-view-state'),
#         dcc.Store(id='oco3-selected-zip', data=None),
#         dcc.Store(id='expansion-state', data='none'),
#         dcc.Interval(
#             id='animation-interval',
#             interval=DEFAULT_ANIMATION_SPEED,
#             disabled=True
#         ),
        
#         # Main layout sections
#         create_header_card(),
#         create_analysis_controls_card(year_options, initial_year, initial_date_options, initial_dates),
        
#         # Main Content - Using flex container
#         html.Div([
#             html.Div(
#                 create_map_section(),
#                 id="left-panel",
#                 className="panel-transition flex-grow-1",
#                 style={'flex': '1', 'minWidth': '0'}
#             ),
#             html.Div(
#                 create_expansion_controls(),
#                 style={'width': '30px'},  # Fixed width for expansion controls
#                 className="d-flex flex-column justify-content-center"
#             ),
#             html.Div(
#                 create_time_series_section(),
#                 id="right-panel",
#                 className="panel-transition flex-grow-1",
#                 style={'flex': '1', 'minWidth': '0'}
#             )
#         ], className="d-flex gap-2", style={'height': '65vh'}),
        
#         # Restore button
#         html.Button(
#             "Restore Panels",
#             id="restore-button",
#             className="restore-button",
#             style={'display': 'none'}
#         )
        
#     ], fluid=True, className="px-4 py-3")

# In emissions.py -> get_layout()

    def get_layout():
        """Return the page layout"""
        year_options, dates_by_year = get_years_and_dates(available_dates)
        initial_year = year_options[0]['value'] if year_options else None
        initial_dates = sorted(dates_by_year.get(initial_year, []), reverse=True) if initial_year else []
        initial_date_options = [{'label': date, 'value': date} for date in initial_dates]
        
        return dbc.Container([
            # Apply padding to an inner div, not the main container
            html.Div([
                # Help Modal and stores remain the same
                create_help_modal(),
                dcc.Store(id="animation-state"),
                dcc.Store(id="current-frame", data=0),
                dcc.Store(id='oco3-map-view-state'),
                dcc.Store(id='oco3-selected-zip', data=None),
                dcc.Store(id='expansion-state', data='none'),
                dcc.Interval(
                    id='animation-interval',
                    interval=DEFAULT_ANIMATION_SPEED,
                    disabled=True
                ),
                
                # Main layout sections
                create_header_card(),
                create_analysis_controls_card(year_options, initial_year, initial_date_options, initial_dates),
                
                # Main Content - Using flex container
                html.Div([
                    html.Div(
                        create_map_section(),
                        id="left-panel",
                        className="panel-transition flex-grow-1",
                        style={'flex': '1', 'minWidth': '0'}
                    ),
                    html.Div(
                        create_expansion_controls(),
                        style={'width': '30px'},  # Fixed width for expansion controls
                        className="d-flex flex-column justify-content-center"
                    ),
                    html.Div(
                        create_time_series_section(),
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
                )
            ], className="px-4 py-3") # <-- PADDING MOVED HERE
            
        ], fluid=True) # <-- PADDING REMOVED FROM HERE


def init():
    """Initialize data and return available dates"""
    global census_tracts, zip_codes, socab_boundary, data, available_dates
    census_tracts, zip_codes, socab_boundary = load_geodata()
    data = load_oco3_data()
    available_dates = sorted(data['date'].unique()) if not data.empty else []
    return available_dates

def register_callbacks(app):
        """Register callbacks for the emissions page"""
        
        @app.callback(
    [
        Output("play-button", "disabled"),
        Output("pause-button", "disabled"),
        Output("time-slider", "disabled"),
        Output("animation-speed", "disabled")
    ],
    Input("oco3-sam-view-toggle", "value")
    )
        
        def toggle_animation_controls(view_type):
            """Disable all animation controls when in ZIP code view"""
            is_disabled = view_type == 'zip'
            return is_disabled, is_disabled, is_disabled, is_disabled
        
        @app.callback(
        [Output("header-collapse", "is_open"),
         Output("header-chevron", "className")],
        [Input("header-toggle", "n_clicks")],
        [State("header-collapse", "is_open")]
        )
        
        def toggle_header(n_clicks, is_open):
            if n_clicks:
                return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
            return is_open, "fas fa-chevron-down ms-2"
        
        @app.callback(
        [Output("analysis-controls-collapse", "is_open"),
         Output("analysis-controls-chevron", "className")],
        [Input("analysis-controls-toggle", "n_clicks")],
        [State("analysis-controls-collapse", "is_open")]
        )
        
        
        def toggle_analysis_controls(n_clicks, is_open):
            if n_clicks:
                return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
            return is_open, "fas fa-chevron-down ms-2"
        
        @app.callback(
        [Output("map-collapse", "is_open"),
         Output("map-chevron", "className")],
        [Input("map-toggle", "n_clicks")],
        [State("map-collapse", "is_open")]
        )
        
        def toggle_map(n_clicks, is_open):
            if n_clicks:
                return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
            return is_open, "fas fa-chevron-down ms-2"

        @app.callback(
        [Output("timeseries-collapse", "is_open"),
         Output("timeseries-chevron", "className")],
        [Input("timeseries-toggle", "n_clicks")],
        [State("timeseries-collapse", "is_open")]
        )
        
        def toggle_timeseries(n_clicks, is_open):
            if n_clicks:
                return not is_open, "fas fa-chevron-up ms-2" if not is_open else "fas fa-chevron-down ms-2"
            return is_open, "fas fa-chevron-down ms-2"
            
        @app.callback(
        [Output('oco3-year-picker', 'value'),
         Output('oco3-date-picker', 'value'),
         Output('oco3-sam-view-toggle', 'value'),
         Output('oco3-sam-stat-toggle', 'value'),
         Output('animation-speed', 'value'),
         Output('scale-type', 'value'),
         Output('scale-min', 'value'),
         Output('scale-max', 'value'),
         Output('animation-state', 'data'),
         Output('current-frame', 'data'),
         Output('oco3-selected-zip', 'data'),
         Output('time-slider', 'value'),
         # Add panel layout outputs
         Output('left-panel', 'style',allow_duplicate=True),
         Output('right-panel', 'style',allow_duplicate=True),
         Output('restore-button', 'style',allow_duplicate=True),
         Output('expansion-state', 'data',allow_duplicate=True)],
        Input('oco3-restart-button', 'n_clicks'),
        prevent_initial_call=True
        )
        
        def reset_to_initial_state(n_clicks):
            year_options, dates_by_year = get_years_and_dates(available_dates)
            initial_year = year_options[0]['value'] if year_options else None
            initial_dates = sorted(dates_by_year.get(initial_year, []), reverse=True) if initial_year else []
            initial_date = initial_dates[0] if initial_dates else None
            
            # Define base style for panels
            base_style = {'flex': '1', 'minWidth': '0'}
            
            return (
                initial_year,          # year picker
                initial_date,          # date picker
                'native',             # view toggle
                'mean',               # stat toggle
                1000,                 # animation speed
                'variable',           # scale type
                400,                  # scale min
                425,                  # scale max
                {'playing': False},   # animation state
                0,                    # current frame
                None,                 # selected zip
                0,                    # slider value
                base_style,           # left panel style
                base_style,           # right panel style
                {'display': 'none'},  # restore button style
                'none'               # expansion state
            )

        @app.callback(
        Output("time-slider", "max"),
        Input("oco3-year-picker", "value")
        )
        
        def update_slider_range(selected_year):
            """Update the slider's maximum value based on total number of dates"""
            if not available_dates:
                return 0
                
            # Use all available dates for the complete time period
            return len(available_dates) - 1

        
    
        @app.callback(
        Output('oco3-selected-zip', 'data', allow_duplicate=True),
        [Input('oco3-map-plot', 'clickData'),
         Input('oco3-sam-view-toggle', 'value')],
        prevent_initial_call=True
        )
        def update_selected_zip(clickData, view_type):
            if view_type != 'zip' or not clickData:
                return None
                
            try:
                hover_text = clickData['points'][0]['text']
                zip_code = hover_text.split('<br>')[0].split(': ')[1]
                return zip_code
            except (IndexError, KeyError):
                return None

        @app.callback(
        [Output('oco3-date-picker', 'options'),
         Output('oco3-date-picker', 'value', allow_duplicate=True)],
        Input('oco3-year-picker', 'value'),
        prevent_initial_call='initial_duplicate'
    )
        def update_date_options(selected_year):
            if not selected_year:
                return [], None
                
            year_dates = sorted([date for date in available_dates 
                         if date.startswith(selected_year)])
            date_options = [{'label': date, 'value': date} 
                           for date in year_dates]
            
            return date_options, year_dates[0] if year_dates else None

        @app.callback(
            [Output("scale-min", "disabled"),
             Output("scale-max", "disabled")],
            Input("scale-type", "value")
        )
        def toggle_scale_inputs(scale_type):
            disabled = scale_type == "variable"
            return disabled, disabled
    
        @app.callback(
        [Output("animation-interval", "disabled"),
         Output("animation-interval", "interval"),
         Output("animation-state", "data", allow_duplicate=True),
         Output("animation-speed", "value", allow_duplicate=True)],
        [Input("play-button", "n_clicks"),
         Input("pause-button", "n_clicks"),
         Input("animation-speed", "value")],
        [State("animation-state", "data"),
         State("current-frame", "data")],
        prevent_initial_call='initial_duplicate'
    )
        def control_animation(play_clicks, pause_clicks, speed, animation_state, current_frame):
            ctx_msg = dash.callback_context
            button_id = ctx_msg.triggered[0]["prop_id"].split(".")[0] if ctx_msg.triggered else None
            
            # Convert speed to integer if possible
            try:
                speed = int(float(speed)) if speed is not None else DEFAULT_ANIMATION_SPEED
            except (ValueError, TypeError):
                speed = DEFAULT_ANIMATION_SPEED
            
            # Validate speed
            if speed < MIN_ANIMATION_SPEED:
                validated_speed = MIN_ANIMATION_SPEED
            elif speed > MAX_ANIMATION_SPEED:
                validated_speed = MAX_ANIMATION_SPEED
            else:
                validated_speed = speed
            
            # Initialize animation state if needed
            if current_frame is None:
                current_frame = 0
            
            new_animation_state = {
                "frame": current_frame,
                "playing": animation_state.get("playing", False) if animation_state else False,
                "was_playing": animation_state.get("playing", False) if animation_state else False
            }
            
            # Handle different button clicks
            if button_id == "play-button":
                new_animation_state["playing"] = True
                new_animation_state["was_playing"] = False
                return False, validated_speed, new_animation_state, validated_speed
            
            elif button_id == "pause-button":
                new_animation_state["playing"] = False
                new_animation_state["was_playing"] = True
                return True, validated_speed, new_animation_state, validated_speed
            
            elif button_id == "animation-speed":
                is_playing = new_animation_state["playing"]
                return not is_playing, validated_speed, new_animation_state, validated_speed
            
            # Default return
            return True, validated_speed, new_animation_state, validated_speed
        
            @app.callback(
                [Output("animation-start-date", "options"),
                 Output("animation-start-date", "value")],
                [Input("animation-state", "data"),
                 Input("current-frame", "data")],
                [State("animation-start-date", "value")],
                prevent_initial_call=True
            )
            def manage_start_date_selector(animation_state, current_frame, current_value):
                is_playing = animation_state and animation_state.get("playing", False)
                
                available_dates_until_now = available_dates[:current_frame + 1] if current_frame is not None else available_dates
                options = [{"label": date, "value": date} for date in available_dates_until_now]
                
                if current_value not in available_dates_until_now:
                    current_value = available_dates_until_now[0] if available_dates_until_now else None
                
                return options, current_value
    
        @app.callback(
        [Output("slider-date-display", "children"),
         Output("current-frame", "data", allow_duplicate=True),
         Output("time-slider", "value", allow_duplicate=True),
         Output("animation-state", "data", allow_duplicate=True)],  # Add this output
        [Input("animation-interval", "n_intervals"),
         Input("time-slider", "value")],
        [State("current-frame", "data"),
         State("animation-state", "data")],
        prevent_initial_call='initial_duplicate'
    )
        def update_frame_and_date(n_intervals, slider_value, current_frame, animation_state):
            if not available_dates:
                return "", 0, 0, {"playing": False}
            
            ctx_msg = dash.callback_context
            trigger = ctx_msg.triggered[0]["prop_id"].split(".")[0] if ctx_msg.triggered else None
            
            # Initialize current_frame if None
            if current_frame is None:
                current_frame = 0
            
            # Handle slider movement
            if trigger == "time-slider":
                if slider_value is not None:
                    current_frame = min(slider_value, len(available_dates) - 1)
                    current_date = available_dates[current_frame]
                    return current_date, current_frame, slider_value, animation_state
            
            # Handle animation state
            if not animation_state or not animation_state.get("playing"):
                current_date = available_dates[current_frame]
                return current_date, current_frame, current_frame, animation_state
            
            # Handle animation progression
            next_frame = current_frame + 1
            
            # Check if we've reached the end
            if next_frame >= len(available_dates):
                # Stop animation at the last frame
                new_animation_state = {
                    "playing": False,
                    "was_playing": True,
                    "frame": current_frame
                }
                current_date = available_dates[current_frame]
                return current_date, current_frame, current_frame, new_animation_state
            
            # Continue animation
            current_date = available_dates[next_frame]
            return current_date, next_frame, next_frame, animation_state

            
        @app.callback(
        [Output('oco3-map-plot', 'figure'),
         Output('oco3-time-series', 'figure'),
         Output('oco3-map-view-state', 'data')],
        [Input('oco3-date-picker', 'value'),
         Input('oco3-sam-view-toggle', 'value'),
         Input('oco3-selected-zip', 'data'),
         Input('oco3-sam-stat-toggle', 'value'),
         Input("animation-interval", "n_intervals"),
         Input("scale-type", "value"),
         Input("scale-min", "value"),
         Input("scale-max", "value"),
         Input("animation-state", "data")],
        [State('oco3-map-plot', 'relayoutData'),
         State('oco3-map-view-state', 'data'),
         State("current-frame", "data"),
         State("time-slider", "value")]
    )
        def update_dashboard(selected_date, view_type, selected_zip, stat_type,
                            n_intervals, scale_type, scale_min, scale_max,
                            animation_state, relayoutData, map_state, current_frame, slider_value):
            ctx = dash.callback_context
            trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
            
            # Update view state handling
            if relayoutData and 'mapbox.center' in relayoutData:
                view_state = {
                    'center': relayoutData['mapbox.center'],
                    'zoom': relayoutData['mapbox.zoom']
                }
            else:
                view_state = map_state if map_state else get_initial_view_state()
            
            # Determine which date to use based on the trigger and animation state
            is_playing = animation_state and animation_state.get("playing", False)
            was_playing = animation_state and animation_state.get("was_playing", False)
            
            # Priority decision tree for date selection
            if trigger == 'oco3-date-picker':
                display_date = selected_date
            elif is_playing and current_frame is not None:
                display_date = available_dates[current_frame] if current_frame < len(available_dates) else available_dates[-1]
            elif was_playing and slider_value is not None:
                display_date = available_dates[slider_value] if slider_value < len(available_dates) else available_dates[-1]
            else:
                display_date = selected_date
        
            # Ensure scale parameters are valid numbers
            scale_min = float(scale_min) if scale_min is not None else 400
            scale_max = float(scale_max) if scale_max is not None else 425
            
            # Create visualizations with all necessary parameters
            map_fig, time_series_fig = create_analysis_frame(
                display_date,
                view_type,
                selected_zip,
                stat_type,
                view_state,
                scale_type,
                scale_min,
                scale_max
            )
            
            return map_fig, time_series_fig, view_state
        
        @app.callback(
        Output("help-modal", "is_open"),
        [Input("oco3-help-button", "n_clicks"), 
         Input("help-close", "n_clicks")],
        [State("help-modal", "is_open")],
    )
        def toggle_help_modal(n1, n2, is_open):
            if n1 or n2:
                return not is_open
            return is_open
        
        @app.callback(
        Output("download-spatial-data-content", "data"),
        Input("download-spatial-data", "n_clicks"),
        prevent_initial_call=True
        )
        def download_spatial_data(n_clicks):
            """
            Downloads spatial data from GCS, zips it in memory,
            and serves it to the user.
            """
            if not n_clicks:
                return None
        
            # Initialize the GCS filesystem interface
            fs = gcsfs.GCSFileSystem()
        
            # Create a temporary directory to store files for zipping
            with tempfile.TemporaryDirectory() as temp_dir:
                shapefile_types = ['.shp', '.shx', '.dbf', '.prj']
                
                # --- Download ZIP code shapefiles ---
                zip_base_gcs = f"{SHAPEFILE_PATH.rstrip('/')}/zip_code_socab"
                zip_base_local = os.path.join(temp_dir, 'zip_code_socab')
                for ext in shapefile_types:
                    gcs_path = zip_base_gcs + ext
                    if fs.exists(gcs_path):
                        fs.get(gcs_path, zip_base_local + ext)
        
                # --- Download SOCAB boundary shapefiles ---
                socab_base_gcs = f"{SHAPEFILE_PATH.rstrip('/')}/socabbound"
                socab_base_local = os.path.join(temp_dir, 'socabbound')
                for ext in shapefile_types:
                    gcs_path = socab_base_gcs + ext
                    if fs.exists(gcs_path):
                        fs.get(gcs_path, socab_base_local + ext)
        
                # --- Download OCO-3 CSV file ---
                csv_path_gcs = f"{OCO3_DATA_PATH.rstrip('/')}/clipped_oco3_obs.csv"
                csv_path_local = os.path.join(temp_dir, 'oco3_observations.csv')
                if fs.exists(csv_path_gcs):
                    fs.get(csv_path_gcs, csv_path_local)
        
                # --- Zip the downloaded files from the temporary directory ---
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arc_name = os.path.relpath(file_path, temp_dir)
                            zip_file.write(file_path, arc_name)
        
                zip_buffer.seek(0)
                
                return dcc.send_bytes(
                    zip_buffer.getvalue(),
                    "oco3_spatial_data.zip"
                )
            
        @app.callback(
        Output("download-timeseries-data-content", "data"),
        Input("download-timeseries-data", "n_clicks"),
        [State('oco3-time-series', 'figure')],
        prevent_initial_call=True
        )
        def download_timeseries_data(n_clicks, figure):
            if not n_clicks:
                return None
            
            try:
                # Create empty DataFrame for consolidated data
                consolidated_df = pd.DataFrame()
                
                # Extract data from each trace in the figure
                for trace in figure['data']:
                    dates = trace['x']
                    values = trace['y']
                    trace_name = trace['name']
                    
                    # Create DataFrame for current trace
                    df = pd.DataFrame({
                        'Date': dates,
                        f"XCO2_{trace_name}": values  # Prefix with XCO2 for clarity
                    })
                    
                    # Merge with consolidated DataFrame
                    if consolidated_df.empty:
                        consolidated_df = df
                    else:
                        consolidated_df = pd.merge(consolidated_df, df, on='Date', how='outer')
                
                # Sort by date
                consolidated_df.sort_values('Date', inplace=True)
                
                # Create descriptive filename
                if not consolidated_df.empty:
                    start_date = consolidated_df['Date'].min()
                    end_date = consolidated_df['Date'].max()
                    filename = f"oco3_timeseries_{start_date}_{end_date}.csv"
                else:
                    filename = "oco3_timeseries.csv"
                
                return dcc.send_data_frame(consolidated_df.to_csv, filename, index=False)
                
            except Exception as e:
                print(f"Error generating time series CSV: {str(e)}")
                return None
            
            """Register callbacks for panel expansion functionality"""
        @app.callback(
            [Output('left-panel', 'style'),
             Output('right-panel', 'style'),
             Output('restore-button', 'style'),
             Output('expansion-state', 'data')],
            [Input('expand-left', 'n_clicks'),
             Input('expand-right', 'n_clicks'),
             Input('restore-button', 'n_clicks')],
            [State('expansion-state', 'data')]
        )
        def handle_expansion(left_clicks, right_clicks, restore_clicks, current_state):
            ctx = dash.callback_context
            if not ctx.triggered:
                return {'flex': '1', 'minWidth': '0'}, {'flex': '1', 'minWidth': '0'}, {'display': 'none'}, 'none'
            
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
            
            return {'flex': '1', **base_style}, {'flex': '1', **base_style}, {'display': 'none'}, 'none'