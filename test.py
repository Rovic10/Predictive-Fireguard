from flask import Flask, jsonify, request, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import euclidean_distances
import datetime
import requests

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://ronald1:123@localhost/fire_prediction'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
CORS(app)

class Pinagbuhatan(db.Model):
    __tablename__ = 'pinagbuhatan_records'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    location = db.Column(db.String(255), nullable=False)
    cause = db.Column(db.String(255), nullable=False)

# Fire Prediction Model Training Function
def train_model():
    df1 = pd.read_csv('firerecords.csv')
    df2 = pd.read_csv('nofire.csv')
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.dropna(subset=['Fire Occurred'])

    # Select features and labels
    X = df[['Year', 'Month', 'Day', 'Human Error', 'Electrical', 'Temperature', 'Wind Speed', 'Precipitation']]
    y = df['Fire Occurred']

    # Handle missing values with imputation
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Check model accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

    return clf, imputer

# Load historical data for similarity analysis
def load_historical_data():
    fire_df = pd.read_csv('firerecords.csv')
    no_fire_df = pd.read_csv('nofire.csv')
    fire_df['Fire Occurred'] = 1
    no_fire_df['Fire Occurred'] = 0
    combined_df = pd.concat([fire_df, no_fire_df], ignore_index=True)
    return combined_df

# Compute averages for fire and no-fire days
def compute_averages(df):
    fire_grouped = df.groupby('Fire Occurred').mean()
    fire_avg = fire_grouped.loc[1]
    no_fire_avg = fire_grouped.loc[0]
    return fire_avg, no_fire_avg

# Calculate similarity between today's weather and historical averages
def calculate_similarity(today_weather, fire_avg, no_fire_avg):
    today_array = np.array(today_weather).reshape(1, -1)
    fire_avg_array = fire_avg[['Temperature', 'Wind Speed', 'Precipitation']].values.reshape(1, -1)
    no_fire_avg_array = no_fire_avg[['Temperature', 'Wind Speed', 'Precipitation']].values.reshape(1, -1)

    fire_distance = euclidean_distances(today_array, fire_avg_array)[0][0]
    no_fire_distance = euclidean_distances(today_array, no_fire_avg_array)[0][0]

    return fire_distance, no_fire_distance

# Train model and initialize global variables
clf, imputer = train_model()  # Initialize classifier and imputer globally
historical_df = load_historical_data()
fire_avg, no_fire_avg = compute_averages(historical_df)

@app.route('/')
def home():
    return render_template('server.html')  # Ensure 'server.html' exists in the templates folder

@app.route('/api/predict_today', methods=['GET'])
def predict_today():
    try:
        # Fetch weather data from an external API
        weather_api_key = "a36039c9d2288f7a4abe69090750734f"  # Replace with your actual API key
        location = "Pasig"  # Specify your location
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_api_key}&units=metric"
        
        response = requests.get(weather_url)
        weather_data = response.json()

        # Extract relevant weather information for today's prediction
        today_weather = {
            'Temperature': weather_data['main']['temp'],
            'Wind Speed': weather_data['wind']['speed'],
            'Precipitation': 0  # Assume no precipitation for now; you could update this
        }
        
        # Get today's weather values in a list
        today_weather_values = [today_weather['Temperature'], today_weather['Wind Speed'], today_weather['Precipitation']]
        
        # Compute similarity to fire and no-fire historical averages
        fire_distance, no_fire_distance = calculate_similarity(today_weather_values, fire_avg, no_fire_avg)

        # Return average weather for fire and no-fire days
        average_weather_on_fire_days = fire_avg[['Temperature', 'Wind Speed', 'Precipitation']].to_dict()
        average_weather_on_non_fire_days = no_fire_avg[['Temperature', 'Wind Speed', 'Precipitation']].to_dict()
        
        # Make a prediction using the trained model
        today_data = {
            'Year': [datetime.datetime.now().year],
            'Month': [datetime.datetime.now().month],
            'Day': [datetime.datetime.now().day],
            'Human Error': [1],  # You could adjust this
            'Electrical': [0],  # Adjust based on your analysis
            'Temperature': [today_weather['Temperature']],
            'Wind Speed': [today_weather['Wind Speed']],
            'Precipitation': [today_weather['Precipitation']]
        }

        today_df = pd.DataFrame(today_data)
        today_df_imputed = imputer.transform(today_df)
        outbreak_today_prob = clf.predict_proba(today_df_imputed)[0][1] * 100
        
        # Combine both machine learning prediction and historical similarity
        return jsonify({
            'percentage_chance_of_fire_outbreak_today': f'{outbreak_today_prob:.2f}%',
            'similarity_to_fire_days': fire_distance,
            'similarity_to_non_fire_days': no_fire_distance,
            'average_weather_on_fire_days': average_weather_on_fire_days,
            'average_weather_on_non_fire_days': average_weather_on_non_fire_days
        })
    except Exception as e:
        print(f"Error predicting today's fire outbreak: {e}")
        return jsonify({"error": "Unable to make prediction"}), 500



if __name__ == '__main__':
    app.run(port=5001, debug=True)
