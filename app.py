import streamlit as st
import os

# Check if data files exist, if not, generate them
if not os.path.exists('data') or len(os.listdir('data')) < 8:
    st.info("Generating data files. This will take a moment...")
    import data_generator
    st.rerun()

# Set page configuration
st.set_page_config(
    page_title="WeatherInsight",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main content
st.title("WeatherInsight! ☁️🌡️")
st.subheader("Your Gateway to Smarter Weather & Air Quality Intelligence")

# Add some information about the application
st.markdown("""
### Welcome to WeatherInsight!

This application provides comprehensive weather and air quality monitoring tools:

#### 🌡️ **Nagpur Temperature Heatmaps**
- View temperature distribution across Nagpur from 2017-2024
- Analyze historical climate trends with interactive visualizations
- Understand temperature patterns and urban heat islands

#### 📊 **IoT Sensor Dashboard**
- Real-time temperature and humidity monitoring
- Powered by ESP32-based sensors
- Data storage and retrieval via Supabase

#### Getting Started
Use the sidebar navigation to explore the different features of WeatherInsight.
""")

# Application features in columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Nagpur Temperature Heatmaps")
    st.markdown("""
    - Interactive temperature visualization
    - Historical data from 2017-2024
    - Detailed heatmap analysis
    - Urban heat island detection
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Weather_heat-wave_heatwave.svg/1200px-Weather_heat-wave_heatwave.svg.png")

with col2:
    st.markdown("### IoT Sensor Dashboard")
    st.markdown("""
    - Real-time temperature monitoring
    - Humidity tracking with time-series visualization
    - Sensor data analytics
    - Cloud-based data storage with Supabase
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Dht22.jpg/440px-Dht22.jpg")

# Footer
st.markdown("---")
st.markdown("© 2024 WeatherInsight - Providing weather intelligence for informed decisions")
