from datetime import datetime
from meteostat import Point, Daily
import pandas as pd

# Set time period for 2023
start = datetime(2023, 1, 1)
end = datetime(2023, 12, 31)

# Create Point for Pasig, Philippines (coordinates for Pasig City)
pasig = Point(14.5869, 121.0614, 10)

# Get daily data for 2023
data_2023 = Daily(pasig, start, end)
data_2023 = data_2023.fetch()

# Export data to CSV
data_2023.to_csv('pasig_weather_data_2023.csv')
