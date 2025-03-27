import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(layout="wide", page_title="F1 Driver DNA Analysis")

# Load your processed data (assuming you've saved results to files)
# If not, you'll need to run the analysis first
@st.cache_data
def load_data():
    # Load the data from the pickle file you just created
    with open('driver_dna_analysis.pkl', 'rb') as f:
        data = pickle.load(f)
    
    return {
        'overall_results': data['overall_results'],
        'track_type_results': data['track_type_results'], 
        'weather_results': data['weather_results'],
        'driver_keys': list(data['aggregated_overall'].keys())
    }

# Title and description
st.title("F1 Driver DNA - Qualifying Analysis")
st.markdown("""
This dashboard visualizes F1 driver styles extracted from **qualifying session telemetry**. 
These patterns may differ from race driving styles, where factors like fuel management, 
traffic, and strategy come into play.
""")

# Load data
try:
    data = load_data()
    
    # Main layout with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Driver DNA Fingerprints", 
        "Style Comparisons", 
        "Track Type Adaptability",
        "Weather Adaptability"
    ])
    
    # TAB 1: DRIVER DNA FINGERPRINTS
    with tab1:
        st.header("Driver DNA Fingerprint")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Driver selection
            all_drivers = sorted(list(set([k.split('_')[0] for k in data['driver_keys']])))
            selected_driver = st.selectbox("Select Driver:", all_drivers)
            
            # Year selection - show available years for this driver
            available_years = sorted([k.split('_')[1] for k in data['driver_keys'] 
                                    if k.split('_')[0] == selected_driver])
            
            selected_year = st.selectbox("Select Year:", available_years)
            
            # Display driver style info
            driver_key = f"{selected_driver}_{selected_year}"
            if driver_key in data['overall_results']['feature_df'].index:
                cluster = data['overall_results']['feature_df'].loc[driver_key, 'cluster']
                style = data['overall_results']['style_names'][cluster]
                
                st.subheader("Overall Driving Style:")
                st.markdown(f"**{style}**")
                
                # Show description
                desc = data['overall_results']['style_descriptions'][cluster]
                st.markdown(f"*{desc}*")
                
                # Track type styles
                st.subheader("Track-Specific Styles:")
                for track_type, results in data['track_type_results'].items():
                    if driver_key in results['feature_df'].index:
                        track_cluster = results['feature_df'].loc[driver_key, 'cluster']
                        track_style = results['style_names'][track_cluster]
                        st.markdown(f"**{track_type.title()}**: {track_style}")
        
        with col2:
            if driver_key in data['overall_results']['feature_df'].index:
                # Create radar chart
                feature_cols = [col for col in data['overall_results']['feature_df'].columns 
                               if col not in ['driver', 'year', 'cluster']]
                
                driver_values = data['overall_results']['feature_df'].loc[driver_key, feature_cols].values
                
                # Format feature names for display
                display_features = [f.replace('_', ' ').title() for f in feature_cols]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=driver_values,
                    theta=display_features,
                    fill='toself',
                    name=f"{selected_driver} ({selected_year})",
                    line=dict(color='rgb(31, 119, 180)', width=2),
                    fillcolor='rgba(31, 119, 180, 0.3)'
                ))
                
                # Add average of driver's style cluster for comparison
                cluster = data['overall_results']['feature_df'].loc[driver_key, 'cluster']
                cluster_avg = data['overall_results']['cluster_analysis'].loc[cluster].values
                
                fig.add_trace(go.Scatterpolar(
                    r=cluster_avg,
                    theta=display_features,
                    fill='toself',
                    name=f"Style Average",
                    line=dict(color='rgba(255, 99, 71, 0.8)', width=1, dash='dot'),
                    fillcolor='rgba(255, 99, 71, 0.1)'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[-3, 3]
                        )),
                    title=f"{selected_driver} DNA Fingerprint ({selected_year})",
                    showlegend=True,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation of key metrics
                st.subheader("Understanding the Metrics")
                st.markdown("""
                * **Avg Brake Intensity**: How aggressively the driver applies brakes (higher = more aggressive)
                * **Avg Throttle Intensity**: How aggressively the driver applies throttle (higher = more aggressive)
                * **Speed Variability**: How much the driver varies their speed throughout the lap
                * **Path Smoothness**: How smooth and consistent the racing line is (lower = more aggressive line)
                * **Gear Changes**: Frequency of gear changes per kilometer
                * **Short Shift Ratio**: Percentage of upshifts that occur before peak RPM
                * **Entry Exit Bias**: Ratio of entry to exit speeds (>1 = prioritizes entry, <1 = prioritizes exit)
                * **Avg Corner Speed Reduction**: How much speed is reduced in corners (higher = more reduction)
                """)
    
    # TAB 2: STYLE COMPARISONS
    with tab2:
        st.header("Driver Style Comparisons")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Select two drivers to compare
            driver1 = st.selectbox("Select First Driver:", all_drivers, key="driver1")
            year1 = st.selectbox("Year:", [k.split('_')[1] for k in data['driver_keys'] 
                                        if k.split('_')[0] == driver1], key="year1")
            
        with col2:
            driver2 = st.selectbox("Select Second Driver:", all_drivers, key="driver2")
            year2 = st.selectbox("Year:", [k.split('_')[1] for k in data['driver_keys'] 
                                        if k.split('_')[0] == driver2], key="year2")
        
        # Create comparison visualization
        key1 = f"{driver1}_{year1}"
        key2 = f"{driver2}_{year2}"
        
        if key1 in data['overall_results']['feature_df'].index and key2 in data['overall_results']['feature_df'].index:
            feature_cols = [col for col in data['overall_results']['feature_df'].columns 
                           if col not in ['driver', 'year', 'cluster']]
            
            values1 = data['overall_results']['feature_df'].loc[key1, feature_cols].values
            values2 = data['overall_results']['feature_df'].loc[key2, feature_cols].values
            
            display_features = [f.replace('_', ' ').title() for f in feature_cols]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values1,
                theta=display_features,
                fill='toself',
                name=f"{driver1} ({year1})",
                line=dict(color='rgb(31, 119, 180)', width=2),
                fillcolor='rgba(31, 119, 180, 0.3)'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=values2,
                theta=display_features,
                fill='toself',
                name=f"{driver2} ({year2})",
                line=dict(color='rgb(255, 99, 71)', width=2),
                fillcolor='rgba(255, 99, 71, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[-3, 3]
                    )),
                title=f"Driving Style Comparison: {driver1} vs {driver2}",
                showlegend=True,
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show style information
            col1, col2 = st.columns(2)
            
            with col1:
                cluster1 = data['overall_results']['feature_df'].loc[key1, 'cluster']
                style1 = data['overall_results']['style_names'][cluster1]
                desc1 = data['overall_results']['style_descriptions'][cluster1]
                
                st.subheader(f"{driver1}'s Style ({year1})")
                st.markdown(f"**{style1}**")
                st.markdown(f"*{desc1}*")
                
                # Show top 3 distinctive features
                st.markdown("**Key distinctive traits:**")
                cluster_avg = data['overall_results']['cluster_analysis'].loc[cluster1]
                top_features = cluster_avg.abs().sort_values(ascending=False).head(3).index
                
                for feature in top_features:
                    value = cluster_avg[feature]
                    direction = "High" if value > 0 else "Low"
                    st.markdown(f"- {direction} {feature.replace('_', ' ')}: {value:.2f}")
            
            with col2:
                cluster2 = data['overall_results']['feature_df'].loc[key2, 'cluster']
                style2 = data['overall_results']['style_names'][cluster2]
                desc2 = data['overall_results']['style_descriptions'][cluster2]
                
                st.subheader(f"{driver2}'s Style ({year2})")
                st.markdown(f"**{style2}**")
                st.markdown(f"*{desc2}*")
                
                # Show top 3 distinctive features
                st.markdown("**Key distinctive traits:**")
                cluster_avg = data['overall_results']['cluster_analysis'].loc[cluster2]
                top_features = cluster_avg.abs().sort_values(ascending=False).head(3).index
                
                for feature in top_features:
                    value = cluster_avg[feature]
                    direction = "High" if value > 0 else "Low"
                    st.markdown(f"- {direction} {feature.replace('_', ' ')}: {value:.2f}")
    
    # TAB 3: TRACK TYPE ADAPTABILITY
    with tab3:
        st.header("Driver Adaptability Across Track Types")
        
        # Get all drivers with data for all track types
        adaptable_drivers = []
        for driver_key in data['driver_keys']:
            driver = driver_key.split('_')[0]
            year = driver_key.split('_')[1]
            
            # Check if driver has data for all track types
            has_all_types = True
            for track_type in data['track_type_results']:
                if driver_key not in data['track_type_results'][track_type]['feature_df'].index:
                    has_all_types = False
                    break
            
            if has_all_types:
                adaptable_drivers.append(driver_key)
        
        # Calculate adaptability scores
        if adaptable_drivers:
            adaptability_data = []
            
            for driver_key in adaptable_drivers:
                driver = driver_key.split('_')[0]
                year = driver_key.split('_')[1]
                
                styles = {}
                for track_type, result in data['track_type_results'].items():
                    cluster = result['feature_df'].loc[driver_key, 'cluster']
                    style = result['style_names'][cluster]
                    styles[track_type] = style
                
                # Count unique styles
                unique_styles = len(set(styles.values()))
                
                # Calculate adaptability (percentage of track types with different styles)
                adaptability = ((unique_styles - 1) / (len(data['track_type_results']) - 1) 
                               if len(data['track_type_results']) > 1 else 0)
                
                adaptability_data.append({
                    'driver': driver,
                    'year': year,
                    'driver_key': driver_key,
                    'adaptability': adaptability * 100,
                    'unique_styles': unique_styles,
                    'styles': styles
                })
            
            # Sort by adaptability
            adaptability_data = sorted(adaptability_data, key=lambda x: x['adaptability'], reverse=True)
            
            # Create adaptability chart
            drivers = [f"{d['driver']} ({d['year']})" for d in adaptability_data]
            values = [d['adaptability'] for d in adaptability_data]
            
            fig = px.bar(
                x=drivers, 
                y=values,
                title="Driver Adaptability Across Track Types",
                labels={'x': 'Driver', 'y': 'Adaptability Score (%)'},
                color=values,
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Allow user to select a driver to see details
            selected_adaptable_driver = st.selectbox(
                "Select driver to see track type styles:", 
                [f"{d['driver']} ({d['year']})" for d in adaptability_data]
            )
            
            # Find the selected driver data
            selected_data = next(d for d in adaptability_data 
                               if f"{d['driver']} ({d['year']})" == selected_adaptable_driver)
            
            # Display track specific styles
            st.subheader(f"{selected_data['driver']}'s Styles Across Track Types ({selected_data['year']})")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**High Speed Tracks:**")
                st.markdown(f"{selected_data['styles']['high_speed']}")
            
            with col2:
                st.markdown("**Technical Tracks:**")
                st.markdown(f"{selected_data['styles']['technical']}")
            
            with col3:
                st.markdown("**Street Circuits:**")
                st.markdown(f"{selected_data['styles']['street']}")
            
            # Show trait variation across track types
            st.subheader("Trait Variation Across Track Types")
            
            # Get feature data for each track type
            feature_cols = [col for col in data['track_type_results']['high_speed']['feature_df'].columns 
                           if col not in ['driver', 'year', 'cluster']]
            
            track_values = {}
            for track_type in data['track_type_results']:
                if selected_data['driver_key'] in data['track_type_results'][track_type]['feature_df'].index:
                    track_values[track_type] = data['track_type_results'][track_type]['feature_df'].loc[
                        selected_data['driver_key'], feature_cols].values
            
            # Create comparison plot
            fig = go.Figure()
            
            colors = {
                'high_speed': 'rgb(31, 119, 180)',
                'technical': 'rgb(255, 99, 71)', 
                'street': 'rgb(50, 205, 50)'
            }
            
            for track_type, values in track_values.items():
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=[f.replace('_', ' ').title() for f in feature_cols],
                    fill='toself',
                    name=f"{track_type.title()}",
                    line=dict(color=colors[track_type], width=2),
                    fillcolor=f"rgba{colors[track_type][3:-1]}, 0.3)"
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[-3, 3]
                    )),
                title=f"{selected_data['driver']}'s Trait Variation by Track Type ({selected_data['year']})",
                showlegend=True,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: WEATHER ADAPTABILITY
    with tab4:
        st.header("Driving Style Adaptation to Weather Conditions")
        
        # Get drivers with data in both conditions
        weather_adaptable = []
        for driver_key in data['driver_keys']:
            driver = driver_key.split('_')[0]
            year = driver_key.split('_')[1]
            
            if (driver_key in data['weather_results']['dry']['feature_df'].index and 
                driver_key in data['weather_results']['wet']['feature_df'].index):
                weather_adaptable.append(driver_key)
        
        if weather_adaptable:
            # Allow user to select driver
            selected_driver = st.selectbox(
                "Select driver to see weather adaptation:", 
                [f"{key.split('_')[0]} ({key.split('_')[1]})" for key in weather_adaptable]
            )
            
            driver, year = selected_driver.split(' ')[0], selected_driver.split('(')[1][:-1]
            driver_key = f"{driver}_{year}"
            
            if (driver_key in data['weather_results']['dry']['feature_df'].index and 
                driver_key in data['weather_results']['wet']['feature_df'].index):
                
                # Get styles in each condition
                dry_cluster = data['weather_results']['dry']['feature_df'].loc[driver_key, 'cluster']
                dry_style = data['weather_results']['dry']['style_names'][dry_cluster]
                
                wet_cluster = data['weather_results']['wet']['feature_df'].loc[driver_key, 'cluster']
                wet_style = data['weather_results']['wet']['style_names'][wet_cluster]
                
                # Display the styles
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Dry Conditions Style:")
                    st.markdown(f"**{dry_style}**")
                    
                    desc = data['weather_results']['dry']['style_descriptions'][dry_cluster]
                    st.markdown(f"*{desc}*")
                
                with col2:
                    st.subheader("Wet Conditions Style:")
                    st.markdown(f"**{wet_style}**")
                    
                    desc = data['weather_results']['wet']['style_descriptions'][wet_cluster]
                    st.markdown(f"*{desc}*")
                
                # Create a comparison visualization
                feature_cols = [col for col in data['weather_results']['dry']['feature_df'].columns 
                               if col not in ['driver', 'year', 'cluster']]
                
                dry_values = data['weather_results']['dry']['feature_df'].loc[driver_key, feature_cols].values
                wet_values = data['weather_results']['wet']['feature_df'].loc[driver_key, feature_cols].values
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=dry_values,
                    theta=[f.replace('_', ' ').title() for f in feature_cols],
                    fill='toself',
                    name="Dry Conditions",
                    line=dict(color='rgb(255, 165, 0)', width=2),
                    fillcolor='rgba(255, 165, 0, 0.3)'
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=wet_values,
                    theta=[f.replace('_', ' ').title() for f in feature_cols],
                    fill='toself',
                    name="Wet Conditions",
                    line=dict(color='rgb(30, 144, 255)', width=2),
                    fillcolor='rgba(30, 144, 255, 0.3)'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[-3, 3]
                        )),
                    title=f"{driver}'s Driving Style: Dry vs. Wet Conditions ({year})",
                    showlegend=True,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate key differences
                differences = np.abs(dry_values - wet_values)
                feature_diffs = list(zip(feature_cols, differences))
                feature_diffs.sort(key=lambda x: x[1], reverse=True)
                
                st.subheader("Biggest Changes in Wet Conditions")
                
                for i, (feature, diff) in enumerate(feature_diffs[:3]):
                    dry_val = dry_values[feature_cols.index(feature)]
                    wet_val = wet_values[feature_cols.index(feature)]
                    
                    change = "increases" if wet_val > dry_val else "decreases"
                    
                    st.markdown(f"**{i+1}. {feature.replace('_', ' ').title()}**: {change} by {diff:.2f} standard deviations")
                
                # Show trait stability metric
                trait_stability = 100 - (np.mean(differences) / 6 * 100)  # 6 is max possible difference (from -3 to +3)
                
                st.metric("Overall Trait Stability", f"{trait_stability:.1f}%", 
                         help="Higher percentage means more consistent driving traits across weather conditions")
                
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.markdown("""
    ### Data Loading Error
    
    This dashboard requires the analysis results from your driver DNA notebook.
    Please run the analysis code first and save the results.
    
    You can modify the `load_data()` function to load your specific results.
    """)

st.markdown("---")
st.caption("""
This application is unofficial and is not associated in any way with the Formula 1 companies. 
F1, FORMULA ONE, FORMULA 1, FIA FORMULA ONE WORLD CHAMPIONSHIP, GRAND PRIX and related marks 
are trade marks of Formula One Licensing B.V.
""")