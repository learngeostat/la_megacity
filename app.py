from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Dash app with Bootstrap
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Expose Flask server
server = app.server

# Bootstrap-enhanced layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    # Bootstrap header
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("LA Megacity - Phase 3A (Bootstrap)", className="text-center mb-4"),
                html.P("Testing dash-bootstrap-components", className="text-center text-muted mb-4")
            ])
        ]),
        
        # Bootstrap navigation
        dbc.Nav([
            dbc.NavLink("Home", href="/", active="exact"),
            dbc.NavLink("Page 1", href="/page-1", active="exact"),
            dbc.NavLink("Page 2", href="/page-2", active="exact"),
        ], pills=True, className="justify-content-center mb-4"),
        
        # Content
        html.Div(id="page-content")
    ])
])

# Same callback as before
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    logger.info(f"Rendering page: {pathname}")
    
    if pathname == "/" or pathname is None:
        return dbc.Card([
            dbc.CardBody([
                html.H3("Home - Bootstrap Working!", className="card-title"),
                html.P("✅ dash-bootstrap-components loaded successfully", className="text-success"),
                dbc.Alert("Phase 3A: Bootstrap components test", color="info")
            ])
        ])
    elif pathname == "/page-1":
        return dbc.Card([dbc.CardBody([html.H3("Page 1"), html.P("Bootstrap styling active")])])
    elif pathname == "/page-2":
        return dbc.Card([dbc.CardBody([html.H3("Page 2"), html.P("All systems working")])])
    else:
        return dbc.Alert(f"Page '{pathname}' not found", color="danger")

# Health check
@server.route('/health')
def health():
    return {'status': 'healthy', 'phase': 'phase-3a-bootstrap'}, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Phase 3A Bootstrap test on port {port}")
    app.run_server(host='0.0.0.0', port=port, debug=False)
