import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Clean Air For India",
    page_icon="🌬️",
    layout="wide"
)

st.title('AIR QUALITY INDEX')

# Add an expandable section with multiple subsections
with st.expander('More Information'):
    # Add an "About" subsection
    st.markdown('### About The Project')
    st.write('This project is an interactive visualization of air quality data for cities in India. The goal of this project is to provide an accessible and informative way for people to explore air quality data and learn more about public policies.')
    
    # Add a subsection on air quality in India
    st.markdown('### Air Quality in India')
    st.write('Air quality in India is a vital issue due to its severe health impacts, including respiratory diseases and increased healthcare costs, economic productivity losses, and environmental degradation. India\'s commitment to global agreements and the interconnectedness of air pollution and climate change underline the urgency of improving air quality.\n\nUrbanization, industrialization, and regulatory needs demand attention. Enhancing air quality ensures better quality of life, sustains long-term development, and fosters a healthier population. Despite progress, sustained efforts are crucial to mitigate the broad spectrum of issues stemming from poor air quality.')
    
    # Add a subsection on Clean Air for India
    st.markdown('### Clean Air for India')
    st.write('By providing accessible air quality information, we aim to enhance awareness and create a way for people to collaborate effectively. This approach fosters informed actions, empowering individuals to collectively contribute to improving air quality and aligns with the website\'s focus on enhancing public engagement for cleaner air.')
    
   
    # Add a subsection on the data
    st.markdown('### Data')
    st.write('The data originates from "Air Quality Data in India (2015 - 2020)" available at: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india?select=city_day.csv')

# Load the air quality dataset with error handling
try:
    # Attempt to load the data from the expected path
    df = pd.read_csv('C:\\Users\\lenovo\\Downloads\\WeatherNexus\\WeatherNexus\\air_quality_data.csv')
    
    # Remove rows with missing values in the date column
    df = df.dropna(subset=['Date'])
    
    # Convert the date column to a datetime object
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'air_quality_data.csv' exists in the current directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# ------------- City Selection -----------------
# Get unique cities from the data
cities = sorted(df['City'].unique())

# Create a selectbox widget to allow the user to select the city
selected_city = st.selectbox('Select city', cities)

# Filter the data to only include rows for the selected city
city_data = df[df['City'] == selected_city]

if city_data.empty:
    st.warning(f"No data available for {selected_city}.")
    st.stop()

# ------------- Pollutants Diagram -----------------
st.markdown(f"## Air Quality Pollutants in {selected_city}")

# Create a multiselect widget to allow the user to select the pollutants to display
available_pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
pollutants = st.multiselect('Select pollutants', available_pollutants)

# Create a scatter plot to display the selected pollutants over time
if pollutants:
    # Filter out rows where all selected pollutants are NaN
    valid_pollutant_data = city_data.dropna(subset=pollutants, how='all')
    
    if not valid_pollutant_data.empty:
        chart_data = valid_pollutant_data.melt(
            id_vars=['Date', 'City'], 
            value_vars=pollutants, 
            var_name='pollutant', 
            value_name='level'
        )
        
        # Create an interactive scatter plot
        fig = px.scatter(
            chart_data, 
            x='Date', 
            y='level', 
            color='pollutant',
            title=f'Air Quality of {selected_city} - Selected Pollutants Over Time',
            labels={'level': 'Concentration', 'Date': 'Date'},
            hover_data=['pollutant', 'level', 'Date']
        )
        
        # Update the layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Concentration Level',
            legend_title='Pollutant',
            height=500
        )
        
        # Show the plot
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data available for the selected pollutants in {selected_city}.")
else:
    st.info('Please select at least one pollutant to display the chart.')

# ------------- Indicators Acceptable Levels -----------------
st.markdown(f'## Level of Air Pollutants in {selected_city}')

# Create a DataFrame with the acceptable levels of various air pollutants
data = {
    'Pollutant': ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'],
    'Acceptable Level': [12, 50, 53, 100, 100, 100, 9, 75, 70, 5, 7.5, 150]
}
acceptable_levels = pd.DataFrame(data)

# Set the index of the acceptable_levels DataFrame to the 'Pollutant' column
acceptable_levels = acceptable_levels.set_index('Pollutant')

# Group the data by city and year
city_data['Year'] = city_data['Date'].dt.year
grouped = city_data.groupby(['City', 'Year'])

try:
    # Calculate the mean of each pollutant column
    annual_averages = grouped[available_pollutants].mean().round(1)
    
    # Reset the index to move the group labels into columns
    annual_averages = annual_averages.reset_index()
    
    # Melt the annual_averages DataFrame to create a long format table
    long_table = annual_averages.melt(
        id_vars=['City', 'Year'], 
        var_name='Pollutant', 
        value_name='Value'
    )
    
    # Filter the data to only include rows for the selected city
    long_table = long_table[long_table['City'] == selected_city]
    
    if not long_table.empty:
        # Pivot the long_table DataFrame to create a wide format table with columns for each year
        pollutant_table = long_table.pivot_table(
            index='Pollutant', 
            columns='Year', 
            values='Value'
        )
        
        # Reindex the pollutant_table DataFrame to match the order of pollutants in the acceptable_levels DataFrame
        pollutant_table = pollutant_table.reindex(acceptable_levels.index)
        
        # Add a column for the acceptable levels
        pollutant_table.insert(0, 'Acceptable Level', acceptable_levels['Acceptable Level'])
        
        # Style the table to highlight values that exceed the acceptable levels
        def highlight_exceeds(val, acceptable):
            if pd.isna(val) or pd.isna(acceptable):
                return ''
            return 'background-color: #ffcccc' if val > acceptable else ''
        
        # Apply the styling function
        styled_table = pollutant_table.style.apply(
            lambda row: [highlight_exceeds(val, row['Acceptable Level']) 
                        for val in row], 
            axis=1
        )
        
        # Display the styled table
        st.dataframe(styled_table, use_container_width=True)
    else:
        st.warning(f"No annual averages available for {selected_city}.")
        
except Exception as e:
    st.error(f"Error generating the pollutant table: {str(e)}")

# Add some spacing
st.markdown("---")

# Add information about air quality policies
st.markdown("## Air Quality Policies")
st.write("""
India has implemented several policies to combat air pollution:

1. **National Clean Air Programme (NCAP)**: Launched in 2019, it aims to reduce particulate matter pollution by 20-30% by 2024.

2. **Bharat Stage Emission Standards**: These are automotive emission standards to regulate air pollutants from vehicles.

3. **Pradhan Mantri Ujjwala Yojana**: A scheme to provide clean cooking fuel to women below the poverty line.

4. **National Air Quality Index (AQI)**: A tool to communicate air quality status to people in simple terms.
""")

