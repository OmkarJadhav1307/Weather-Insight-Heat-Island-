import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os
import glob

# Set page configuration
st.set_page_config(
    page_title="Heatwave ML Prediction",
    page_icon="🔥",
    layout="wide"
)

# Title and description
st.title("Heatwave Prediction for Nagpur")
st.markdown("""
This page uses machine learning to predict future heatwave patterns in Nagpur based on historical data.
The model learns from temperature sensor data collected from 2017-2024 and projects future trends up to 2040.
""")

# Function to load and combine all data files
@st.cache_data
def load_all_data():
    # Path to data files
    data_path = "data"
    all_files = glob.glob(os.path.join(data_path, "nagpur_*.csv"))
    
    all_data = []
    for file in all_files:
        # Extract year from filename
        year = int(file.split('_')[-1].split('.')[0])
        
        # Read data
        df = pd.read_csv(file)
        df['year'] = year
        
        # Calculate average sensor value for the year
        avg_sensor_value = df['sensor_value'].mean()
        
        all_data.append({
            'year': year,
            'avg_sensor_value': avg_sensor_value,
            'data': df
        })
    
    # Create a dataframe with yearly averages
    yearly_data = pd.DataFrame([{'year': d['year'], 'avg_sensor_value': d['avg_sensor_value']} for d in all_data])
    yearly_data = yearly_data.sort_values('year')
    
    return yearly_data, all_data

# Load all data
yearly_data, all_data_list = load_all_data()

# Display historical data
st.subheader("Historical Temperature Data (2017-2024)")
col1, col2 = st.columns([2, 1])

with col1:
    # Plot historical data
    fig = px.line(
        yearly_data, 
        x='year', 
        y='avg_sensor_value',
        markers=True,
        labels={
            'year': 'Year',
            'avg_sensor_value': 'Average Temperature (°C)'
        },
        title='Average Annual Temperature in Nagpur (2017-2024)'
    )
    fig.update_layout(
        xaxis=dict(dtick=1),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.dataframe(
        yearly_data.rename(columns={'year': 'Year', 'avg_sensor_value': 'Avg. Temperature (°C)'}),
        hide_index=True,
        use_container_width=True
    )

# Train the prediction model
st.subheader("Temperature Prediction Model")

# Create features and target for training
X = yearly_data[['year']].values
y = yearly_data['avg_sensor_value'].values

# Sidebar for model selection and parameters
with st.sidebar:
    st.header("Model Parameters")
    model_type = st.selectbox("Select Model Type", ["Linear Regression", "."])
    
    if model_type == "Polynomial Regression":
        poly_degree = st.slider("Polynomial Degree", min_value=2, max_value=5, value=2)
    
    prediction_year = st.slider("Prediction Year", min_value=2025, max_value=2040, value=2030)
    
    # User can toggle to see data for specific years
    st.header("Data Visualization")
    selected_year = st.selectbox("View data for specific year", sorted([d['year'] for d in all_data_list]))

# Train the selected model
if model_type == "Linear Regression":
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate predictions for future years
    future_years = np.array(range(2017, prediction_year + 1)).reshape(-1, 1)
    predicted_temps = model.fit(X, y).predict(future_years)
    
    # Create prediction dataframe
    pred_df = pd.DataFrame({
        'year': future_years.flatten(),
        'predicted_temperature': predicted_temps,
        'is_prediction': future_years.flatten() > 2024
    })
    
    # Add confidence intervals (simple approximation)
    residuals = y - model.predict(X)
    std_residuals = np.std(residuals)
    pred_df['upper_bound'] = pred_df['predicted_temperature'] + 1.96 * std_residuals
    pred_df['lower_bound'] = pred_df['predicted_temperature'] - 1.96 * std_residuals
    
    # Display model equation
    coefficient = model.coef_[0]
    intercept = model.intercept_
    equation = f"Temperature = {coefficient:.4f} × Year + {intercept:.4f}"
    
else:  # Polynomial Regression
    poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Generate predictions for future years
    future_years = np.array(range(2017, prediction_year + 1)).reshape(-1, 1)
    future_years_poly = poly_features.transform(future_years)
    predicted_temps = model.predict(future_years_poly)
    
    # Create prediction dataframe
    pred_df = pd.DataFrame({
        'year': future_years.flatten(),
        'predicted_temperature': predicted_temps,
        'is_prediction': future_years.flatten() > 2024
    })
    
    # Add confidence intervals (simple approximation)
    residuals = y - model.predict(X_poly)
    std_residuals = np.std(residuals)
    pred_df['upper_bound'] = pred_df['predicted_temperature'] + 1.96 * std_residuals
    pred_df['lower_bound'] = pred_df['predicted_temperature'] - 1.96 * std_residuals
    
    # Display model equation
    coefficients = model.coef_[0]
    intercept = model.intercept_
    terms = [f"{coefficients[i]:.4f} × Year^{i+1}" for i in range(poly_degree)]
    equation = f"Temperature = {' + '.join(terms)} + {intercept:.4f}"

# Display model information
st.markdown(f"**Model Type:** {model_type}")
st.markdown(f"**Model Equation:** {equation}")

# Plot predictions
st.subheader(f"Temperature Predictions (2017-{prediction_year})")

# Create the prediction plot
fig = px.line(
    pred_df, 
    x='year', 
    y='predicted_temperature',
    labels={
        'year': 'Year',
        'predicted_temperature': 'Temperature (°C)'
    },
    title=f'Predicted Average Annual Temperature in Nagpur (2017-{prediction_year})'
)

# Add the historical data points
fig.add_scatter(
    x=yearly_data['year'],
    y=yearly_data['avg_sensor_value'],
    mode='markers',
    name='Historical Data',
    marker=dict(size=10, color='red')
)

# Add confidence interval
fig.add_scatter(
    x=pred_df['year'],
    y=pred_df['upper_bound'],
    mode='lines',
    line=dict(width=0),
    showlegend=False
)

fig.add_scatter(
    x=pred_df['year'],
    y=pred_df['lower_bound'],
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(0,100,80,0.2)',
    name='95% Confidence Interval'
)

# Add a vertical line at 2024 to separate historical from predictions
fig.add_vline(x=2024.5, line_dash="dash", line_color="gray", annotation_text="Predictions Start")

fig.update_layout(
    xaxis=dict(dtick=2),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# Display prediction results for specific years
col1, col2 = st.columns(2)

with col1:
    # Show predictions for key years
    prediction_data = pred_df[pred_df['year'] > 2024].copy()
    prediction_data = prediction_data[['year', 'predicted_temperature', 'lower_bound', 'upper_bound']]
    prediction_data.columns = ['Year', 'Predicted Temp (°C)', 'Lower Bound (°C)', 'Upper Bound (°C)']
    prediction_data = prediction_data.round(2)
    
    # Filter to show only every 5 years for cleaner display
    years_to_show = [2025, 2030, 2035, 2040]
    filtered_prediction_data = prediction_data[prediction_data['Year'].isin(years_to_show)]
    
    st.markdown("### Predicted Temperatures for Key Years")
    st.dataframe(filtered_prediction_data, hide_index=True, use_container_width=True)

with col2:
    # Calculate the increase from 2024 to prediction_year
    temp_2024 = pred_df[pred_df['year'] == 2024]['predicted_temperature'].values[0]
    temp_pred_year = pred_df[pred_df['year'] == prediction_year]['predicted_temperature'].values[0]
    increase = temp_pred_year - temp_2024
    percent_increase = (increase / temp_2024) * 100
    
    st.markdown("### Projected Temperature Change")
    st.markdown(f"From 2024 to {prediction_year}:")
    st.markdown(f"- Temperature increase: **{increase:.2f}°C**")
    st.markdown(f"- Percentage increase: **{percent_increase:.2f}%**")
    
    # Add a warning if the temperature increase is significant
    if increase > 1.5:
        st.warning(f"⚠️ The projected temperature increase of {increase:.2f}°C is significant and may lead to more frequent and intense heatwaves.")

# Display heatmap for a specific year
st.subheader(f"Temperature Heatmap for {selected_year}")

# Get data for the selected year
selected_data = next((d['data'] for d in all_data_list if d['year'] == selected_year), None)

if selected_data is not None:
    # Display temperature heatmap using Folium for the selected year
    st.markdown("### Folium Heatmap Visualization")
    
    # Create a folium map centered on Nagpur
    m = folium.Map(
        location=[21.1458, 79.0882],  # Nagpur Coordinates
        tiles="cartodbpositron",
        zoom_start=12,
    )
    
    # Create heatmap data
    heat_data = [[row.latitude, row.longitude, row.sensor_value] for _, row in selected_data.iterrows()]
    
    # Add the heatmap to the map
    HeatMap(
        heat_data, 
        radius=25, 
        blur=15, 
        min_opacity=0.6, 
        max_zoom=13, 
        gradient={
            "0.4": 'blue', 
            "0.6": 'lime', 
            "0.8": 'yellow', 
            "1.0": 'red'
        }
    ).add_to(m)
    
    # Display the folium map in Streamlit
    folium_static(m, width=800, height=500)
    
    # Also add the scatter mapbox for comparison
    st.markdown("### Interactive Sensor Map")
    fig = px.scatter_mapbox(
        selected_data,
        lat="latitude",
        lon="longitude",
        color="sensor_value",
        size="sensor_value",
        color_continuous_scale="Viridis",
        size_max=15,
        zoom=11,
        mapbox_style="carto-positron",
        title=f"Temperature Sensor Readings in Nagpur ({selected_year})",
        labels={"sensor_value": "Temperature (°C)"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display the raw data
    with st.expander("View Sensor Data"):
        st.dataframe(selected_data, use_container_width=True)
else:
    st.error(f"No data available for {selected_year}")

# Add a predictive heatmap slider visualization
st.markdown("---")
st.subheader("Temperature Heatmap Prediction Slider")
st.markdown("""
Use the slider below to visualize predicted temperature patterns across Nagpur for future years.
This visualization combines our ML model predictions with spatial distribution patterns.
""")

# Create a year slider for future predictions
prediction_slider_year = st.slider(
    "Select Year to View Predicted Temperature Patterns", 
    min_value=2017, 
    max_value=2040, 
    value=2030,
    step=1,
    format="%d",
    key="heatmap_slider"
)

# Check if selected year is in the past (we have actual data) or future (prediction)
is_future_year = prediction_slider_year > 2024

# Get actual data if it's a past year
if not is_future_year:
    prediction_data = next((d['data'] for d in all_data_list if d['year'] == prediction_slider_year), None)
    
    if prediction_data is not None:
        # Display statistics for the selected year
        avg_temp = prediction_data['sensor_value'].mean()
        max_temp = prediction_data['sensor_value'].max()
        min_temp = prediction_data['sensor_value'].min()
        
        # Show statistics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Temperature (°C)", f"{avg_temp:.2f}")
        with col2:
            st.metric("Maximum Temperature (°C)", f"{max_temp:.2f}")
        with col3:
            st.metric("Minimum Temperature (°C)", f"{min_temp:.2f}")
            
        # Create a folium map with actual data
        m = folium.Map(
            location=[21.1458, 79.0882],
            tiles="cartodbpositron",
            zoom_start=12,
        )
        
        # Create heatmap data from actual values
        heat_data = [[row.latitude, row.longitude, row.sensor_value] 
                    for _, row in prediction_data.iterrows()]
        
        # Add the heatmap to the map
        HeatMap(
            heat_data, 
            radius=25, 
            blur=15, 
            min_opacity=0.6, 
            max_zoom=13, 
            gradient={
                "0.4": 'blue', 
                "0.6": 'lime', 
                "0.8": 'yellow', 
                "1.0": 'red'
            }
        ).add_to(m)
        
        # Display the map
        st.markdown(f"#### Actual Temperature Heatmap for {prediction_slider_year}")
        folium_static(m, width=800, height=500)
        
    else:
        st.error(f"No data available for {prediction_slider_year}")
        
else:  # Future year prediction
    # Get the year 2024 data as a base
    base_data = next((d['data'] for d in all_data_list if d['year'] == 2024), None)
    
    if base_data is not None:
        # Get the predicted temperature for the selected future year
        if model_type == "Linear Regression":
            # Use the linear model to predict temperature for the future year
            future_year_array = np.array([[prediction_slider_year]])
            predicted_temp = model.predict(future_year_array)[0]
        else:  # Polynomial Regression
            # Use the polynomial model to predict temperature
            poly_future = poly_features.transform(np.array([[prediction_slider_year]]))
            predicted_temp = model.predict(poly_future)[0]
        
        # Calculate the scaling factor based on the difference between 2024 and predicted year
        temp_2024 = next((d['avg_sensor_value'] for d in all_data_list if d['year'] == 2024), None)
        if temp_2024 is not None:
            scaling_factor = predicted_temp / temp_2024
            
            # Create a copy of the 2024 data and adjust the temperature values
            future_data = base_data.copy()
            future_data['sensor_value'] = future_data['sensor_value'] * scaling_factor
            
            # Display statistics for the predicted year
            avg_temp = future_data['sensor_value'].mean()
            max_temp = future_data['sensor_value'].max()
            min_temp = future_data['sensor_value'].min()
            
            # Show statistics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Avg Temp (°C)", f"{avg_temp:.2f}")
            with col2:
                st.metric("Predicted Max Temp (°C)", f"{max_temp:.2f}")
            with col3:
                st.metric("Predicted Min Temp (°C)", f"{min_temp:.2f}")
                
            # Create a folium map for the prediction
            m = folium.Map(
                location=[21.1458, 79.0882],
                tiles="cartodbpositron",
                zoom_start=12,
            )
            
            # Create heatmap data from predicted values
            heat_data = [[row.latitude, row.longitude, row.sensor_value] 
                        for _, row in future_data.iterrows()]
            
            # Add the heatmap to the map
            HeatMap(
                heat_data, 
                radius=25, 
                blur=15, 
                min_opacity=0.6, 
                max_zoom=13, 
                gradient={
                    "0.4": 'blue', 
                    "0.65": 'lime', 
                    "0.8": 'yellow', 
                    "1.0": 'red'
                }
            ).add_to(m)
            
            # Add title with warning about prediction
            st.markdown(f"#### Predicted Temperature Heatmap for {prediction_slider_year}")
            if prediction_slider_year > 2030:
                st.warning("⚠️ Long-term predictions have higher uncertainty. This visualization is based on the ML model projections and historical spatial distribution patterns.")
            
            # Display the map
            folium_static(m, width=800, height=500)
            
            # Show what changed in a note
            temp_increase = avg_temp - temp_2024
            st.info(f"Predicted temperature increase from 2024 to {prediction_slider_year}: **{temp_increase:.2f}°C**")
        else:
            st.error("Error calculating temperature predictions. Base year (2024) data not available.")
    else:
        st.error("Error loading base data for prediction visualization.")

# Add interpretation and implications
st.markdown("---")
st.subheader("Analysis and Implications")

st.markdown("""
### What does this prediction tell us?

The model suggests a clear warming trend in Nagpur over the coming years. This has several potential implications:

1. **Increased Heatwave Frequency:** Higher average temperatures are likely to correlate with more frequent and intense heatwave events.

2. **Public Health Concerns:** Rising temperatures may lead to increased heat-related illnesses, particularly among vulnerable populations.

3. **Energy Demands:** Higher temperatures will likely increase cooling demands and electricity consumption during summer months.

4. **Water Resources:** Warming may impact local water resources through increased evaporation and changes to precipitation patterns.

### Limitations of this model:

- The prediction is based on limited historical data (2017-2024).
- It doesn't account for potential climate policy changes or mitigation efforts.
- Complex climate feedback mechanisms aren't fully represented in this simplified model.
- The confidence interval widens for predictions further into the future.

For more accurate climate predictions, comprehensive climate models that incorporate multiple variables and feedback mechanisms should be consulted.
""")

# Footer
st.markdown("---")
st.markdown("### Made with ❤️ for understanding climate change in Nagpur")
st.markdown("Data last updated: 2024")