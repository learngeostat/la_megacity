import pandas as pd
import plotly.express as px
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objs as go
import statsmodels.api as sm
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# %% plot auto correlation function
def figure_acf(acf_dataframe, measurement_sites, title_acf):
    fig = px.line(acf_dataframe, x='lags', y=measurement_sites,
                  title=title_acf, height=500)
    # width=1200, height=900)
    fig.update_layout(xaxis_title="Time (lags)", yaxis_title="correlation",
                      margin={"r": 10, "t": 50, "l": 0, "b": 0},
                      title_x=0.5,
                      legend_title="Sites",
                      font_family="Arial",
                      font_color="sandybrown",
                      font_size=20,
                      template='plotly_dark',
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      legend=dict(bgcolor='darkgrey', yanchor="top",
                                  y=0.99,
                                  x=0.30,
                                  orientation="h",
                                  font=dict(
                                      family="Arial",
                                      size=20,
                                      color="black"
                                  ),
                                  bordercolor="Black",
                                  borderwidth=2,

                                  ),
                      legend_title_font_color="maroon",
                      title_font=dict(size=25,
                                      color="seashell",
                                      family="Arial")
                      )

    fig.update_xaxes(showline=True, ticks='outside',
                     ticklen=10, tickcolor="sandybrown", tickwidth=2,
                     showgrid=True,
                     zeroline=False,
                     gridwidth=2,
                     gridcolor='lightskyblue')

    fig.update_yaxes(showline=True, tickformat='.1f', ticks='outside',
                     ticklen=10, tickcolor="sandybrown", tickwidth=2,
                     zeroline=True, zerolinecolor='lightskyblue', zerolinewidth=2, showgrid=True, gridwidth=2,
                     gridcolor='lightskyblue')
    fig.add_shape(
        # Rectangle with reference to the plot
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1.0,
        y1=1.0,
        line=dict(
            color="lightskyblue",
            width=1,
        )
    )
    fig.update_traces(line=dict(width=3))
    return fig
#%% Plot time-series data for a data stored in a dataframe
# Note time-series x axis would be its index
def plot_hourly_timeseries_with_index(df, col_name, title="Time Series"):
    """
    Optimized plotting for time series data where 'datetime_UTC' is the index.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the time-series data.
        col_name (str): The column name to plot.
        title (str): Title for the plot.
    """
    # Ensure the DataFrame index is datetime_UTC and drop missing values in the column
    temp_df = df.dropna(subset=[col_name])

    # Ensure the index is in datetime format
    if not isinstance(temp_df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot the time-series data
    ax.plot(temp_df.index, temp_df[col_name], 
            color='blue', 
            linewidth=0.8,  # Thin lines for better rendering
            alpha=0.8,      # Transparency for aesthetics
            rasterized=True # For large datasets
    )
    
    # Set major x-axis ticks for every 3 months
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=90)
    
    # Set x-axis and y-axis labels and the title
    ax.set_xlabel('Date (UTC)', fontsize=12)
    ax.set_ylabel('CO₂ (ppm)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Set tight layout for better spacing
    plt.tight_layout()
    
    # Show the plot
    plt.show()
#%% plot time_series for sites
def plot_time_series_for_sites(df, sites, plot_type="concentration", title="Time Series for Sites"):
    """
    Plot time-series for multiple sites where each site's data is in a separate column.

    Parameters:
        df (pd.DataFrame): DataFrame with time-series data.
        sites (list): List of site names to plot.
        plot_type (str): Type of data to plot, either "concentration" or "std".
        title (str): Title for the plot.
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    # Determine column naming pattern based on plot_type
    if plot_type == "concentration":
        col_name = lambda site: f"co2_ppm_{site}"
    elif plot_type == "std":
        col_name = lambda site: f"co2_SD_minutes_{site}"
    else:
        raise ValueError("Invalid plot_type. Choose either 'concentration' or 'std'.")

    # Filter DataFrame to keep only the relevant columns
    selected_columns = [col_name(site) for site in sites if col_name(site) in df.columns]
    if not selected_columns:
        raise ValueError("None of the specified site names match the column names in the DataFrame.")

    temp_df = df[selected_columns].dropna(how='all')  # Drop rows where all selected site columns are NaN

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot each site's data
    for site, col in zip(sites, selected_columns):
        ax.plot(temp_df.index, temp_df[col], label=site, linewidth=0.8, alpha=0.8)

    # Configure x-axis ticks and labels
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=90)

    # Add labels, title, and legend
    ax.set_xlabel('Date (UTC)', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(title="Sites", fontsize=10, title_fontsize=12, loc='upper left')

    # Add grid
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Set tight layout
    plt.tight_layout()

    # Show the plot
    plt.show()


# %% Plot Spatial Area of Dominance MAP

def site_ioami_map(full_geometry, half_geometry, selected_sites=None, relayoutData=None, map_state=None):
    """Create map with boundary and monitoring sites.
    
    Returns:
        tuple: (map_fig, current_map_state) where map_fig is the plotly figure and 
        current_map_state is the current map view state
    """
    # Initialize figure
    map_fig = go.Figure()
    current_map_state = None
    
    try:
        # Calculate bounds of full_geometry for initial view
        if full_geometry is not None and hasattr(full_geometry, 'geometry'):
            bounds = full_geometry.geometry.total_bounds
            center_lon = (bounds[0] + bounds[2]) / 2
            center_lat = (bounds[1] + bounds[3]) / 2
            
            # Calculate zoom level to fit the geometry
            lon_range = bounds[2] - bounds[0]
            lat_range = bounds[3] - bounds[1]
            zoom = min(
                math.log2(360 / lon_range) - 1,
                math.log2(180 / lat_range) - 1
            )
            zoom = max(min(zoom, 10), 7)
        else:
            center_lon, center_lat = -117.5, 34
            zoom = 8
        
        # Base layout settings for map with autosize
        map_layout = dict(
            autosize=True,
            mapbox=dict(
                style='open-street-map',
                zoom=zoom,
                center=dict(lat=center_lat, lon=center_lon)
            ),
            margin=dict(l=0, r=70, t=30, b=0),
            uirevision='constant',
            paper_bgcolor='white',
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1,
                font=dict(size=12),
                itemsizing='constant',
                orientation='h'
            )
        )
        
        # Add boundary choropleth
        if full_geometry is not None and hasattr(full_geometry, 'geometry'):
            map_fig.add_trace(go.Choroplethmapbox(
                geojson=full_geometry.geometry.__geo_interface__,
                locations=full_geometry.index,
                z=[1] * len(full_geometry),
                colorscale=[[0, 'rgba(169, 169, 169, 0.4)'], [1, 'rgba(169, 169, 169, 0.4)']],
                showscale=False,
                hoverinfo='skip',
                name='LA Megacity Extent',
                showlegend=True
            ))
        
        # Add monitoring sites if data is available
        if half_geometry is not None and not half_geometry.empty:
            # Create two separate traces for selected and unselected sites
            selected_mask = [site in (selected_sites or []) for site in half_geometry['tower_name']]
            
            # Selected sites
            selected_sites_data = half_geometry[selected_mask]
            if not selected_sites_data.empty:
                map_fig.add_trace(go.Scattermapbox(
                    lat=selected_sites_data['lat'],
                    lon=selected_sites_data['long'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='maroon',
                        opacity=0.8
                    ),
                    text=[f"{name} ({height}m)" for name, height in 
                          zip(selected_sites_data['Full Name'], selected_sites_data['agl'])],
                    customdata=selected_sites_data['tower_name'],
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    name='Selected Sites',
                    showlegend=True
                ))
            
            # Unselected sites
            unselected_sites_data = half_geometry[~np.array(selected_mask)]
            if not unselected_sites_data.empty:
                map_fig.add_trace(go.Scattermapbox(
                    lat=unselected_sites_data['lat'],
                    lon=unselected_sites_data['long'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='rgba(128, 128, 128, 0.6)',
                        opacity=0.8
                    ),
                    text=[f"{name} ({height}m)" for name, height in 
                          zip(unselected_sites_data['Full Name'], unselected_sites_data['agl'])],
                    customdata=unselected_sites_data['tower_name'],
                    hovertemplate="<b>%{text}</b><extra></extra>",
                    name='Unselected Sites',
                    showlegend=True
                ))
        
        # Update map view state based on user interaction or stored state
        if relayoutData and 'mapbox.center' in relayoutData:
            current_map_state = {
                'center': relayoutData['mapbox.center'],
                'zoom': relayoutData['mapbox.zoom']
            }
            map_layout['mapbox'].update(
                center=current_map_state['center'],
                zoom=current_map_state['zoom']
            )
        elif map_state:
            current_map_state = map_state
            map_layout['mapbox'].update(map_state)
        else:
            # Store initial map state if no updates
            current_map_state = {
                'center': map_layout['mapbox']['center'],
                'zoom': map_layout['mapbox']['zoom']
            }
        
        # Apply final layout
        map_fig.update_layout(map_layout)
        
    except Exception as e:
        print(f"Error in site_ioami_map: {str(e)}")
        import traceback
        traceback.print_exc()
        return map_fig, None
    
    return map_fig, current_map_state

# Example usage
# Assuming full_geometry and half_geometry are valid GeoDataFrames
# fig = site_ioami_map(full_geometry, half_geometry)
# fig.show()
def site_time_series(agg_results, selected_sites, title, units, analysis_type='mean', selected_gas='co2'):
    fig = go.Figure()
    
    # Dictionary for gas display names
    gas_display = {
        'co2': 'CO₂',
        'ch4': 'CH₄',
        'co': 'CO'
    }
    
    # Determine time aggregation and set format, tick angle, and calculate appropriate padding
    date_format = '%Y-%m'  # Default to monthly format for display
    tick_angle = 0
    padding_factor = 0.01  # Default padding factor (1% of range)
    
    # Initialize padding_unit with a default value before it's used
    # Using total time range as default in case calculations fail
    if len(agg_results) >= 2:
        total_range = pd.to_datetime(agg_results['datetime_UTC'].max()) - pd.to_datetime(agg_results['datetime_UTC'].min())
        padding_unit = total_range * padding_factor  # Default padding is ~1% of the total time range
        
        # Now try to determine a more specific padding based on data frequency
        try:
            time_delta = pd.to_datetime(agg_results['datetime_UTC'].iloc[1]) - pd.to_datetime(agg_results['datetime_UTC'].iloc[0])
            
            # Calculate padding based on data frequency
            if time_delta.days == 1:  # Daily data
                padding_unit = pd.Timedelta(days=1)
            elif time_delta.days >= 28:  # Monthly data
                padding_unit = pd.Timedelta(days=5)  # ~5 days padding for monthly data
            elif 6 <= time_delta.days <= 8:  # Weekly data
                padding_unit = pd.Timedelta(days=2)  # 2 days padding for weekly data
            elif time_delta.seconds <= 3600 and time_delta.days == 0:  # Hourly data
                padding_unit = pd.Timedelta(hours=1)
            # If none of the above conditions are met, we'll use the default padding_unit
        except Exception as e:
            print(f"Error calculating time delta: {e}. Using default padding.")
    else:
        # If we have fewer than 2 data points, use a standard padding
        padding_unit = pd.Timedelta(days=30)  # Default to 30 days if not enough data points

    # Base layout settings for time series with autosize
    time_series_layout = dict(
        autosize=True,
        title=title,
        xaxis_title="Date",
        margin=dict(l=50, r=30, t=40, b=40),  # Increased bottom margin for labels
        uirevision='constant',
        paper_bgcolor='white',
        plot_bgcolor='rgba(240, 242, 245, 0.8)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            gridwidth=1,
            showline=True,
            linewidth=1,
            linecolor='rgba(128, 128, 128, 0.8)',
            type='date',
            tickformat=date_format,  # This will show only month and year on axis
            hoverformat='%Y-%m-%d %H:%M',  # This will show full date and time on hover
            tickangle=tick_angle,
            tickfont=dict(size=12),
            ticklabelposition="outside",
            automargin=True,
            # Calculate appropriate padding based on data frequency
            range=[
                pd.to_datetime(agg_results['datetime_UTC'].min()) - padding_unit,  # First date minus padding
                pd.to_datetime(agg_results['datetime_UTC'].max()) + padding_unit   # Last date plus padding
            ] if not agg_results.empty else None  # Only set range if we have data
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
        ),
        hovermode='x unified'
    )
    
    try:
        # Check if we're dealing with ratio data
        is_ratio = ':' in selected_gas
        
        if is_ratio:
            # Convert display value to dictionary key (e.g., 'co2:ch4' to 'co2_ch4')
            ratio_key = selected_gas.replace(':', '_')
            
            # Plot ratio data for each site
            for site in selected_sites:
                col_name = f'{ratio_key}_ratio_{site}'
                
                if col_name not in agg_results.columns:
                    print(f"Warning: {col_name} not found in data columns: {agg_results.columns.tolist()}")
                    continue
                
                fig.add_trace(go.Scatter(
                    x=agg_results['datetime_UTC'],
                    y=agg_results[col_name],
                    mode='lines+markers',
                    name=site,
                    line=dict(width=2),
                    marker=dict(size=6),
                    hovertemplate="%{x}<br>" + f"{site}: %{{y:.3f}}<extra></extra>"
                ))
            
            # Set y-axis title for ratio plot
            y_axis_title = f"{selected_gas.replace(':', '/')} Ratio"
            
        else:
            # Original single gas plotting logic
            for site in selected_sites:
                if analysis_type == 'mean':
                    unit = 'ppm' if selected_gas == 'co2' else 'ppb'
                    col_name = f'{selected_gas}_{unit}_{site}'
                    y_axis_title = f"{gas_display.get(selected_gas, selected_gas)} Concentration ({units})"
                else:  # std
                    col_name = f'{selected_gas}_SD_minutes_{site}'
                    y_axis_title = f"{gas_display.get(selected_gas, selected_gas)} Standard Deviation ({units})"
                
                if col_name not in agg_results.columns:
                    print(f"Warning: {col_name} not found in data columns: {agg_results.columns.tolist()}")
                    continue
                
                fig.add_trace(go.Scatter(
                    x=agg_results['datetime_UTC'],
                    y=agg_results[col_name],
                    mode='lines+markers',
                    name=site,
                    line=dict(width=2),
                    marker=dict(size=6),
                    hovertemplate="%{x}<br>" + f"{site}: %{{y:.2f}} {units}<extra></extra>"
                ))
        
        # Update layout with y-axis title
        time_series_layout['yaxis_title'] = y_axis_title
        fig.update_layout(time_series_layout)
        
        return fig
    
    except Exception as e:
        print(f"Error in site_time_series: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return {
            'data': [],
            'layout': {
                'title': {'text': 'Error creating time series'},
                'annotations': [{
                    'text': str(e),
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'color': 'red'}
                }]
            }
        }



# def site_time_series(agg_results, selected_sites, title, units, analysis_type='mean', selected_gas='co2'):
#     fig = go.Figure()
    
#     # Dictionary for gas display names
#     gas_display = {
#         'co2': 'CO₂',
#         'ch4': 'CH₄',
#         'co': 'CO',
#         'ratio': 'Ratio'  # Add ratio type
#     }
    
#     # Determine time aggregation based on data frequency
#     if len(agg_results) >= 2:
#         time_delta = pd.to_datetime(agg_results['datetime_UTC'].iloc[1]) - pd.to_datetime(agg_results['datetime_UTC'].iloc[0])
#         if time_delta.days == 1:
#             date_format = '%Y-%m-%d'  # Daily
#         elif time_delta.days >= 28:
#             date_format = '%Y-%m'     # Monthly
#         else:
#             date_format = '%Y-%m-%d %H:%M'  # Hourly
#     else:
#         date_format = '%Y-%m-%d'  # Default to daily if can't determine
    
#     # Base layout settings for time series with autosize
#     time_series_layout = dict(
#         autosize=True,
#         title=title,
#         xaxis_title="Date",
#         margin=dict(l=50, r=30, t=40, b=40),
#         uirevision='constant',
#         paper_bgcolor='white',
#         plot_bgcolor='rgba(240, 242, 245, 0.8)',
#         xaxis=dict(
#             showgrid=True,
#             gridcolor='rgba(128, 128, 128, 0.2)',
#             gridwidth=1,
#             showline=True,
#             linewidth=1,
#             linecolor='rgba(128, 128, 128, 0.8)',
#             type='date',
#             tickformat=date_format,
#             hoverformat=date_format
#         ),
#         yaxis=dict(
#             showgrid=True,
#             gridcolor='rgba(128, 128, 128, 0.2)',
#             gridwidth=1,
#             showline=True,
#             linewidth=1,
#             linecolor='rgba(128, 128, 128, 0.8)'
#         ),
#         showlegend=True,
#         legend=dict(
#             yanchor="top",
#             y=0.99,
#             xanchor="left",
#             x=0.01
#         ),
#         hovermode='x unified'
#     )
    
#     try:
#         # Handle ratio plots differently
#         if analysis_type == 'ratio':
#             # For ratio plots, look for the ratio column directly
#             ratio_column = f'{title}_ratio'  # This should match how we named it in calculate_gas_ratio
#             if ratio_column not in agg_results.columns:
#                 print(f"Warning: {ratio_column} not found in data columns: {agg_results.columns.tolist()}")
#                 raise ValueError(f"Ratio column {ratio_column} not found in data")
            
#             # Add trace for ratio
#             fig.add_trace(go.Scatter(
#                 x=agg_results['datetime_UTC'],
#                 y=agg_results[ratio_column],
#                 mode='lines+markers',
#                 name=selected_sites[0],  # For ratio we only use one site
#                 line=dict(width=2),
#                 marker=dict(size=6),
#                 hovertemplate=f"{selected_sites[0]}: %{{y:.2f}}<extra></extra>"
#             ))
            
#             y_axis_title = f"{title} Ratio"
            
#         else:
#             # Original single gas plotting logic
#             for site in selected_sites:
#                 if analysis_type == 'mean':
#                     unit = 'ppm' if selected_gas == 'co2' else 'ppb'
#                     col_name = f'{selected_gas}_{unit}_{site}'
#                     y_axis_title = f"{gas_display[selected_gas]} Concentration ({units})"
#                 else:  # std
#                     col_name = f'{selected_gas}_SD_minutes_{site}'
#                     y_axis_title = f"{gas_display[selected_gas]} Standard Deviation ({units})"
                
#                 if col_name not in agg_results.columns:
#                     print(f"Warning: {col_name} not found in data columns: {agg_results.columns.tolist()}")
#                     continue
                
#                 fig.add_trace(go.Scatter(
#                     x=agg_results['datetime_UTC'],
#                     y=agg_results[col_name],
#                     mode='lines+markers',
#                     name=site,
#                     line=dict(width=2),
#                     marker=dict(size=6),
#                     hovertemplate=f"{site}: %{{y:.2f}} {units}<extra></extra>"
#                 ))
        
#         # Update layout with y-axis title
#         time_series_layout['yaxis_title'] = y_axis_title
#         fig.update_layout(time_series_layout)
        
#         return fig
    
#     except Exception as e:
#         print(f"Error in site_time_series: {str(e)}")
#         return {
#             'data': [],
#             'layout': {
#                 'title': {'text': 'Error creating time series'},
#                 'annotations': [{
#                     'text': str(e),
#                     'xref': 'paper',
#                     'yref': 'paper',
#                     'x': 0.5,
#                     'y': 0.5,
#                     'showarrow': False,
#                     'font': {'color': 'red'}
#                 }]
#             }
#         }


# def site_time_series(agg_results, selected_sites, title, units, analysis_type='mean', selected_gas='co2'):
#     fig = go.Figure()
    
#     # Dictionary for gas display names
#     gas_display = {
#         'co2': 'CO₂',
#         'ch4': 'CH₄',
#         'co': 'CO'
#     }
    
#     # Determine time aggregation based on data frequency
#     if len(agg_results) >= 2:
#         time_delta = pd.to_datetime(agg_results['datetime_UTC'].iloc[1]) - pd.to_datetime(agg_results['datetime_UTC'].iloc[0])
#         if time_delta.days == 1:
#             date_format = '%Y-%m-%d'  # Daily
#         elif time_delta.days >= 28:
#             date_format = '%Y-%m'     # Monthly
#         else:
#             date_format = '%Y-%m-%d %H:%M'  # Hourly
#     else:
#         date_format = '%Y-%m-%d'  # Default to daily if can't determine
    
#     # Base layout settings for time series with autosize
#     time_series_layout = dict(
#         autosize=True,
#         title=title,
#         xaxis_title="Date",
#         margin=dict(l=50, r=30, t=40, b=40),
#         uirevision='constant',
#         paper_bgcolor='white',
#         plot_bgcolor='rgba(240, 242, 245, 0.8)',
#         xaxis=dict(
#             showgrid=True,
#             gridcolor='rgba(128, 128, 128, 0.2)',
#             gridwidth=1,
#             showline=True,
#             linewidth=1,
#             linecolor='rgba(128, 128, 128, 0.8)',
#             type='date',
#             tickformat=date_format,  # Use determined format for ticks
#             hoverformat=date_format  # Use determined format for hover
#         ),
#         yaxis=dict(
#             showgrid=True,
#             gridcolor='rgba(128, 128, 128, 0.2)',
#             gridwidth=1,
#             showline=True,
#             linewidth=1,
#             linecolor='rgba(128, 128, 128, 0.8)'
#         ),
#         showlegend=True,
#         legend=dict(
#             yanchor="top",
#             y=0.99,
#             xanchor="left",
#             x=0.01
#         ),
#         hovermode='x unified'  # Show hover for all traces at x position
#     )
    
#     try:
#         for site in selected_sites:
#             # Define column name based on analysis type and gas
#             if analysis_type == 'mean':
#                 unit = 'ppm' if selected_gas == 'co2' else 'ppb'
#                 col_name = f'{selected_gas}_{unit}_{site}'
#                 y_axis_title = f"{gas_display[selected_gas]} Concentration ({units})"
#             else:  # std
#                 col_name = f'{selected_gas}_SD_minutes_{site}'
#                 y_axis_title = f"{gas_display[selected_gas]} Standard Deviation ({units})"
            
#             if col_name not in agg_results.columns:
#                 print(f"Warning: {col_name} not found in data columns: {agg_results.columns.tolist()}")
#                 continue
            
#             # Add trace for each site with custom hover template
#             fig.add_trace(go.Scatter(
#                 x=agg_results['datetime_UTC'],
#                 y=agg_results[col_name],
#                 mode='lines+markers',
#                 name=site,
#                 line=dict(width=2),
#                 marker=dict(size=6),
#                 hovertemplate=f"{site}: %{{y:.2f}} {units}<extra></extra>"  # Custom hover template
#             ))
        
#         # Update layout with y-axis title
#         time_series_layout['yaxis_title'] = y_axis_title
#         fig.update_layout(time_series_layout)
        
#         return fig
    
#     except Exception as e:
#         print(f"Error in site_time_series: {str(e)}")
#         return {
#             'data': [],
#             'layout': {
#                 'title': {'text': 'Error creating time series'},
#                 'annotations': [{
#                     'text': str(e),
#                     'xref': 'paper',
#                     'yref': 'paper',
#                     'x': 0.5,
#                     'y': 0.5,
#                     'showarrow': False,
#                     'font': {'color': 'red'}
#                 }]
#             }
#         }


# def site_time_series(agg_results, selected_sites, title, units, analysis_type='mean'):
#     fig = go.Figure()
    
#     # Base layout settings for time series with autosize
#     time_series_layout = dict(
#         autosize=True,
#         title=title,
#         xaxis_title="Date",
#         margin=dict(l=50, r=30, t=40, b=40),
#         uirevision='constant',
#         paper_bgcolor='white',
#         plot_bgcolor='rgba(240, 242, 245, 0.8)',
#         xaxis=dict(
#             showgrid=True,
#             gridcolor='rgba(128, 128, 128, 0.2)',
#             gridwidth=1,
#             showline=True,
#             linewidth=1,
#             linecolor='rgba(128, 128, 128, 0.8)',
#             type='date',
#             tickformat='%b %Y'
#         ),
#         yaxis=dict(
#             showgrid=True,
#             gridcolor='rgba(128, 128, 128, 0.2)',
#             gridwidth=1,
#             showline=True,
#             linewidth=1,
#             linecolor='rgba(128, 128, 128, 0.8)'
#         ),
#         showlegend=True,
#         legend=dict(
#             yanchor="top",
#             y=0.99,
#             xanchor="left",
#             x=0.01
#         )
#     )
    
#     try:
#         for site in selected_sites:
#             # Define column name based on analysis type
#             if analysis_type == 'mean':
#                 col_name = f'co2_ppm_{site}'
#                 y_axis_title = f"CO₂ Concentration ({units})"
#             else:  # std
#                 col_name = f'co2_SD_minutes_{site}'
#                 y_axis_title = f"CO₂ Standard Deviation ({units})"
            
#             if col_name not in agg_results.columns:
#                 print(f"Warning: {col_name} not found in data columns: {agg_results.columns.tolist()}")
#                 continue
            
#             # Add trace for each site
#             fig.add_trace(go.Scatter(
#                 x=agg_results['datetime_UTC'],
#                 y=agg_results[col_name],
#                 mode='lines+markers',
#                 name=site,
#                 line=dict(width=2),
#                 marker=dict(size=6)
#             ))
        
#         # Update layout with y-axis title
#         time_series_layout['yaxis_title'] = y_axis_title
#         fig.update_layout(time_series_layout)
        
#         return fig
    
#     except Exception as e:
#         print(f"Error in site_time_series: {str(e)}")
#         return {
#             'data': [],
#             'layout': {
#                 'title': {'text': 'Error creating time series'},
#                 'annotations': [{
#                     'text': str(e),
#                     'xref': 'paper',
#                     'yref': 'paper',
#                     'x': 0.5,
#                     'y': 0.5,
#                     'showarrow': False,
#                     'font': {'color': 'red'}
#                 }]
#             }
#         }


# %% Plot violin plot
def figure_violin(agg_results, filter_suffix, measurement_sites, title_, units_):
    column_name_keys = list(agg_results.filter(like=filter_suffix).columns)
    replacement_keys = dict(zip(column_name_keys, measurement_sites))
    agg_results.rename(columns=replacement_keys, inplace=True)

    filtered_data = agg_results.filter(measurement_sites, axis=1)
    filtered_data = filtered_data.melt()
    year_data = pd.concat([agg_results['year']] * len(measurement_sites), ignore_index=True)
    year_data = year_data.to_frame()
    col_data = pd.concat([year_data, filtered_data], axis=1)
    col_data.rename(columns={'value': units_, 'variable': 'site'}, inplace=True)
    # col_data=col_data.sort_values(by=['site'])

    col_data = col_data.dropna()

    fig = px.violin(col_data, y=units_, x='site', color='site', box=True, points='all',
                    title=title_, height=500)

    fig.update_layout(xaxis_title="measurement_sites", yaxis_title=units_,
                      margin={"r": 10, "t": 50, "l": 0, "b": 0},
                      title_x=0.5,
                      legend_title="measurement_sites",
                      font_family="Arial",
                      font_color="sandybrown",
                      font_size=20,
                      template='plotly_dark',
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      legend=dict(bgcolor='darkgrey', yanchor="top",
                                  y=0.99,
                                  x=0.30,
                                  orientation="h",
                                  font=dict(
                                      family="Arial",
                                      size=20,
                                      color="black"
                                  ),
                                  bordercolor="Black",
                                  borderwidth=2,

                                  ),
                      legend_title_font_color="maroon",
                      title_font=dict(size=25,
                                      color="seashell",
                                      family="Arial")
                      )

    fig.update_xaxes(showline=True, ticks='outside',
                     ticklen=10, tickcolor="sandybrown", tickwidth=2,
                     showgrid=True,
                     zeroline=False,
                     gridwidth=2,
                     gridcolor='lightskyblue')

    fig.update_yaxes(showline=True, tickformat='.1f', ticks='outside',
                     ticklen=10, tickcolor="sandybrown", tickwidth=2,
                     zeroline=True, zerolinecolor='lightskyblue', zerolinewidth=2, showgrid=True, gridwidth=2,
                     gridcolor='lightskyblue')
    fig.add_shape(
        # Rectangle with reference to the plot
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1.0,
        y1=1.0,
        line=dict(
            color="lightskyblue",
            width=1,
        )
    )
    fig.update_traces(line=dict(width=3), box_visible=True, meanline_visible=True,
                      points='all', jitter=0.5,
                      marker_line_color='rgba(0,0,0,0.5)',
                      marker_line_width=1)
    return fig


# %% Plot density plots
def figure_histogram(agg_results, filter_suffix, measurement_sites, title_, units_):
    if filter_suffix != None:
        column_name_keys = list(agg_results.filter(like=filter_suffix).columns)
        replacement_keys = dict(zip(column_name_keys, measurement_sites))
        agg_results.rename(columns=replacement_keys, inplace=True)
        filtered_data = agg_results.filter(measurement_sites, axis=1)
    else:
        filtered_data=agg_results
    # find total bins from freedman diaconis rule
    filtered_data=filtered_data.dropna()
    tbins=[]
    for i in measurement_sites:
        third_quartile = np.quantile(filtered_data[i], 0.75)
        first_quartile = np.quantile(filtered_data[i], 0.25)
        inter_quartile_range = third_quartile - first_quartile
        freedman_denominator = np.cbrt(len(filtered_data.index)) # take cube root
        bin_width = 2 * (inter_quartile_range / freedman_denominator)
        data_min, data_max = filtered_data[i].min(), filtered_data[i].max()
        data_range = data_max - data_min
        total_bins = int((data_range / bin_width) + 1)
        tbins.append(total_bins)

    fig = px.histogram(data_frame=filtered_data, x=measurement_sites, marginal="box", nbins=max(tbins),
                       opacity=0.6, histnorm='probability', width=750, height=650,
                       color_discrete_sequence=px.colors.sequential.Hot, title=title_)
    fig.update_traces(marker_line_width=1, marker_line_color="white")

    fig.update_layout(xaxis_title=units_, yaxis_title="probability",
                      margin={"r": 10, "t": 50, "l": 0, "b": 0},
                      title_x=0.5,
                      legend_title="Sites",
                      font_family="Arial",
                      font_color="sandybrown",
                      font_size=20,
                      template='plotly_dark',
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      legend=dict(#yanchor="top",
                                  y=0.70,
                                  x=0.85,
                                  orientation="v",
                                  font={'family': "Arial", 'size': 20, 'color': "black"},
                                  bordercolor="Black",
                                  borderwidth=2,
                                  ),
                      legend_title_font_color="maroon",
                      title_font={'size': 18, 'color': "Black", 'family': "Arial"}
                      )

    fig.update_xaxes(showline=True, ticks='outside',
                     ticklen=10, tickcolor="sandybrown", tickwidth=2,
                     showgrid=True,
                     zeroline=False,
                     gridwidth=2,
                     gridcolor='lightskyblue')

    fig.update_yaxes(showline=True, tickformat='.2f', ticks='outside',
                     ticklen=10, tickcolor="sandybrown", tickwidth=2,
                     zeroline=True, zerolinecolor='lightskyblue', zerolinewidth=2, showgrid=True, gridwidth=2,
                     gridcolor='lightskyblue')

    fig.add_shape(
        # Rectangle with reference to the plot
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1.0,
        y1=1.0,
        line=dict(
            color="lightskyblue",
            width=1,
        )
    )

    return fig

# %% figure line graph
def figure_linegraph(agg_results, filter_suffix, measurement_sites, title_):
    if filter_suffix != None:
        column_name_keys = list(agg_results.filter(like=filter_suffix).columns)
        replacement_keys = dict(zip(column_name_keys, measurement_sites))
        agg_results.rename(columns=replacement_keys, inplace=True)

        filtered_data = agg_results.filter(measurement_sites, axis=1)
        x_name = measurement_sites[0]
        y_name = measurement_sites[1]
        fig = px.scatter(data_frame=filtered_data, x=x_name, y=y_name, width=750, height=650, trendline='ols',
                     trendline_color_override="red",
                     title=title_)
    else:
        x_name=str(measurement_sites[0])
        #x_name=str(measurement_sites[0])#'time_index'
        #agg_results['time_index']=list(agg_results.index)
        y_name = str(measurement_sites[1]) #'time_index'
        fig = px.scatter(data_frame=agg_results, x=x_name, y=y_name, width=750, height=650, trendline='ols',
                     trendline_color_override="red",
                     title=title_)

    fig.update_traces(marker=dict(size=6, color='darkseagreen',
                                  line=dict(width=1,
                                            color='blue')), selector=dict(mode='markers'))

    fig.update_layout(xaxis_title=x_name, yaxis_title=y_name,
                      margin={"r": 10, "t": 50, "l": 0, "b": 0},
                      title_x=0.5,
                      legend_title="Sites",
                      font_family="Arial",
                      font_color="sandybrown",
                      font_size=20,
                      template='plotly_dark',
                      plot_bgcolor='rgba(0, 0, 0, 0)',
                      paper_bgcolor='rgba(0, 0, 0, 0)',
                      legend=dict(yanchor="top",
                                  y=0.70,
                                  x=0.75,
                                  orientation="h",
                                  font={'family': "Arial", 'size': 20, 'color': "black"},
                                  bordercolor="Black",
                                  borderwidth=2,
                                  ),
                      legend_title_font_color="maroon",
                      title_font={'size': 18, 'color': "Black", 'family': "Arial"}
                      )

    fig.update_xaxes(showline=True, ticks='outside',
                     ticklen=10, tickcolor="sandybrown", tickwidth=2,
                     showgrid=True,
                     zeroline=False,
                     gridwidth=2,
                     gridcolor='lightskyblue')

    fig.update_yaxes(showline=True, tickformat='.2f', ticks='outside',
                     ticklen=10, tickcolor="sandybrown", tickwidth=2,
                     zeroline=True, zerolinecolor='lightskyblue', zerolinewidth=2, showgrid=True, gridwidth=2,
                     gridcolor='lightskyblue')

    fig.add_shape(
        # Rectangle with reference to the plot
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1.0,
        y1=1.0,
        line={'color': "lightskyblue", 'width': 1}
    )
    return fig

# %% Scatterplot Matrix
def figure_scatterplot_matrix(agg_results, filter_suffix, measurement_sites, title_):
    if filter_suffix != None:
        column_name_keys = list(agg_results.filter(like=filter_suffix).columns)
        replacement_keys = dict(zip(column_name_keys, measurement_sites))
        agg_results.rename(columns=replacement_keys, inplace=True)
        filtered_data = agg_results.filter(measurement_sites, axis=1)

        fig = ff.create_scatterplotmatrix(filtered_data, diag='histogram', title=title_,
                                          height=1300, width=1500)
    else:
        fig = ff.create_scatterplotmatrix(agg_results, diag='histogram', title=title_,
                                          height=1300, width=1500)
    fig.add_shape(
        # Rectangle with reference to the plot
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1.0,
        y1=1.0,
        line={'color': "lightskyblue", 'width': 1}
    )
    fig.update_layout(
        font_family="Arial",
        font_color="midnightblue",
        font_size=15,
        template='plotly_dark',
        legend=dict(yanchor="top",
                    y=0.70,
                    x=0.75,
                    orientation="h",
                    font={'family': "Arial", 'size': 20, 'color': "black"},
                    bordercolor="Black",
                    borderwidth=2,
                    ),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        legend_title_font_color="maroon",
        title_text=title_,
        title_x=0.5,
        title_font={'size': 25, 'color': "Black", 'family': "Arial"}
    )
    fig.update_xaxes(ticks='outside',
                     showline=True,
                     tickcolor="midnightblue",
                     showgrid=True,
                     zeroline=False,
                     gridwidth=1,
                     gridcolor='white')

    fig.update_yaxes(showline=True,
                     tickcolor="midnightblue",
                     ticks='outside',
                     showgrid=True,
                     gridcolor='white')
    return fig
