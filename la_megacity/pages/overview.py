#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:49:48 2025

@author: vyadav
"""

import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output

def init():
    """Initialize any required resources for the overview page"""
    pass

def get_layout():
    """Return the updated page layout for the Overview page."""
    
    # Store component to hold the scroll target
    scroll_store = dcc.Store(id='scroll-store')
    
    layout = dbc.Container([
        scroll_store,
        dbc.Row([
            # Left Navigation Sidebar
            dbc.Col([
                dbc.Card([
                    html.Div([
                        html.H5("Navigation", className="p-3"),
                        dbc.Nav([
                            dbc.NavLink("System Summary", 
                                      id="nav-system-summary",
                                      n_clicks=0,
                                      className="nav-link",
                                      active="exact"),
                            dbc.NavLink("Monitoring Network", 
                                      id="nav-monitoring-network",
                                      n_clicks=0,
                                      className="nav-link",
                                      active="exact"),
                            dbc.NavLink("Data Sources", 
                                      id="nav-data-sources",
                                      n_clicks=0,
                                      className="nav-link",
                                      active="exact"),
                            dbc.NavLink("Emission Sectors", 
                                      id="nav-emission-sectors",
                                      n_clicks=0,
                                      className="nav-link",
                                      active="exact"),
                            dbc.NavLink("Help & Documentation", 
                                      id="nav-help-docs",
                                      n_clicks=0,
                                      className="nav-link",
                                      active="exact")
                        ], vertical=True, pills=True, className="nav-pills flex-column")
                    ], className="p-2")
                ], className="h-100 border-0 shadow-sm")
            ], width=3),

            # Main Content Area
            dbc.Col([
                html.Div([
                    html.H2("System Summary", id="system-summary", className="mb-4"),
                    dbc.Card([
                        dbc.CardBody([
                            html.P("The Los Angeles Megacity Greenhouse Gas Information System provides comprehensive monitoring and analysis of greenhouse gas emissions across the Los Angeles metropolitan area."),
                            dbc.Row([
                                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Coverage Area"), html.P("Greater Los Angeles Metropolitan Area", className="text-muted")]), className="h-100"), md=4),
                                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Monitoring Stations"), html.P("12 Fixed Locations", className="text-muted")]), className="h-100"), md=4),
                                dbc.Col(dbc.Card(dbc.CardBody([html.H5("Update Frequency"), html.P("Real-time & Daily Aggregates", className="text-muted")]), className="h-100"), md=4)
                            ])
                        ])
                    ], className="mb-5 border-0 shadow-sm"),

                    html.H2("Monitoring Network", id="monitoring-network", className="mb-4"),
                    dbc.Card([
                        dbc.CardBody([
                            html.P("Our network consists of multiple monitoring stations equipped with state-of-the-art sensors for measuring greenhouse gas concentrations and related atmospheric parameters."),
                            html.Div([
                                html.H5("Key Measurements", className="mb-3"),
                                html.Ul([
                                    html.Li("CO2 and CH4 concentrations"),
                                    html.Li("Air quality parameters"),
                                    html.Li("Meteorological data")
                                ], className="list-unstyled")
                            ], className="border-start border-4 border-primary ps-4 mt-4")
                        ])
                    ], className="mb-5 border-0 shadow-sm"),

                    html.H2("Data Sources", id="data-sources", className="mb-4"),
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([html.H5("Ground Measurements"), html.P("Real-time sensor data from monitoring stations", className="text-muted")], md=6),
                                dbc.Col([html.H5("Satellite Data"), html.P("Remote sensing observations from multiple satellites", className="text-muted")], md=6)
                            ])
                        ])
                    ], className="mb-5 border-0 shadow-sm"),

                    html.H2("Emission Sectors", id="emission-sectors", className="mb-4"),
                    dbc.Card([
                        dbc.CardBody([
                            html.P("Detailed emission sector breakdown and trends."),
                            dbc.Row([
                                dbc.Col(html.Ul([html.Li("Transportation"), html.Li("Industrial"), html.Li("Residential")], className="list-unstyled"), md=6)
                            ])
                        ])
                    ], className="mb-5 border-0 shadow-sm"),

                    html.H2("Help & Documentation", id="help-docs", className="mb-4"),
                    dbc.Card([
                        dbc.CardBody([
                            html.P("Documentation and user guides for the system.")
                        ])
                    ], className="mb-5 border-0 shadow-sm"),
                ])
            ], width=9)
        ])
    ], fluid=True)

    return layout

def register_callbacks(app):
    """Register callbacks needed for the overview page"""
    
    app.clientside_callback(
        """
        function(n1, n2, n3, n4, n5) {
            const ctx = dash_clientside.callback_context;
            if (!ctx.triggered.length) return;
            
            const triggerId = ctx.triggered[0].prop_id.split('.')[0];
            const sectionMap = {
                'nav-system-summary': 'system-summary',
                'nav-monitoring-network': 'monitoring-network',
                'nav-data-sources': 'data-sources',
                'nav-emission-sectors': 'emission-sectors',
                'nav-help-docs': 'help-docs'
            };
            
            const targetId = sectionMap[triggerId];
            if (targetId) {
                const element = document.getElementById(targetId);
                if (element) {
                    element.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start',
                        inline: 'nearest'
                    });
                }
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('scroll-store', 'data'),
        [Input('nav-system-summary', 'n_clicks'),
         Input('nav-monitoring-network', 'n_clicks'),
         Input('nav-data-sources', 'n_clicks'),
         Input('nav-emission-sectors', 'n_clicks'),
         Input('nav-help-docs', 'n_clicks')],
        prevent_initial_call=True
    )

