from dash import Dash, html, dcc, Input, Output
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Dash app
app = Dash(__name__)

# IMPORTANT: Expose Flask server for gunicorn
server = app.server

# Simple Dash layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.H1("LA Megacity - Phase 2 (Dash Test)", style={'textAlign': 'center'}),
        html.P("Dash framework running on Cloud Run", style={'textAlign': 'center'}),
        html.Hr(),
        
        # Simple navigation
        html.Div([
            html.A("Home", href="/", style={'marginRight': '20px'}),
            html.A("Page 1", href="/page-1", style={'marginRight': '20px'}),
            html.A("Page 2", href="/page-2")
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # Content area
        html.Div(id="page-content")
    ], style={'padding': '20px'})
])

# Simple callback for routing
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    logger.info(f"Rendering page: {pathname}")
    
    if pathname == "/" or pathname is None:
        return html.Div([
            html.H2("Home Page"),
            html.P("✅ Dash is working on Cloud Run!"),
            html.P(f"Current path: {pathname or '/'}"),
            html.P(f"Port: {os.environ.get('PORT', 'not-set')}")
        ])
    elif pathname == "/page-1":
        return html.Div([
            html.H2("Page 1"),
            html.P("This is page 1 - Dash routing works!")
        ])
    elif pathname == "/page-2":
        return html.Div([
            html.H2("Page 2"),
            html.P("This is page 2 - Everything working!")
        ])
    else:
        return html.Div([
            html.H2("404 - Not Found"),
            html.P(f"Page '{pathname}' not found")
        ])

# Health check for Cloud Run
@server.route('/health')
def health():
    return {'status': 'healthy', 'phase': 'phase-2-dash'}, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Phase 2 Dash app on port {port}")
    app.run_server(host='0.0.0.0', port=port, debug=False)
