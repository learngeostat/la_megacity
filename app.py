from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Dash app with Bootstrap
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Expose Flask server
server = app.server

# Create sample data to test pandas
sample_data = pd.DataFrame({
    'City': ['Los Angeles', 'San Francisco', 'San Diego', 'Sacramento'],
    'Population': [3900000, 873000, 1400000, 525000],
    'CO2_Emissions': np.array([45.2, 12.8, 23.1, 8.7])
})

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("LA Megacity - Phase 3B (Data Processing)", className="text-center mb-4"),
                html.P("Testing pandas + numpy integration", className="text-center text-muted mb-4")
            ])
        ]),
        
        dbc.Nav([
            dbc.NavLink("Home", href="/", active="exact"),
            dbc.NavLink("Data Table", href="/data", active="exact"),
            dbc.NavLink("Stats", href="/stats", active="exact"),
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
                html.H3("Home - Data Processing Working!", className="card-title"),
                html.P("✅ pandas and numpy loaded successfully", className="text-success"),
                dbc.Alert(f"Sample data shape: {sample_data.shape}", color="info"),
                html.P(f"Data types working: {type(sample_data).__name__}")
            ])
        ])
    elif pathname == "/data":
        return dbc.Card([
            dbc.CardBody([
                html.H3("Sample Data Table"),
                dash_table.DataTable(
                    data=sample_data.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in sample_data.columns],
                    style_cell={'textAlign': 'left'}
                )
            ])
        ])
    elif pathname == "/stats":
        return dbc.Card([
            dbc.CardBody([
                html.H3("Data Statistics"),
                html.P(f"Total Population: {sample_data['Population'].sum():,}"),
                html.P(f"Average CO2 Emissions: {sample_data['CO2_Emissions'].mean():.1f}"),
                html.P(f"Max Emissions City: {sample_data.loc[sample_data['CO2_Emissions'].idxmax(), 'City']}")
            ])
        ])
    else:
        return dbc.Alert(f"Page '{pathname}' not found", color="danger")

@server.route('/health')
def health():
    return {'status': 'healthy', 'phase': 'phase-3b-data-processing'}, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Phase 3B data processing test on port {port}")
    app.run_server(host='0.0.0.0', port=port, debug=False)
