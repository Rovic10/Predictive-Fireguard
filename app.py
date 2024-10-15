from flask import Flask, jsonify, request, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import datetime
import requests  # Import requests to fetch weather data

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

# Fire Prediction Logic
def train_model():
    df1 = pd.read_csv('firerecords.csv')
    df2 = pd.read_csv('nofire.csv')
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.dropna(subset=['Fire Occurred'])
    X = df[['Year', 'Month', 'Day', 'Human Error', 'Electrical', 'Temperature', 'Wind Speed', 'Precipitation']]
    y = df['Fire Occurred']
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
    return clf, imputer

# Initialize the model and imputer
clf, imputer = train_model()

@app.route('/')
def home():
    return render_template('server.html')

@app.route('/api/pinagbuhatan_records', methods=['GET', 'POST'])
def manage_all_pinagbuhatan_records():
    if request.method == 'GET':
        try:
            date_query = request.args.get('date')
            if date_query:
                incidents = Pinagbuhatan.query.filter(Pinagbuhatan.date == date_query).order_by(Pinagbuhatan.date.desc()).all()
            else:
                incidents = Pinagbuhatan.query.order_by(Pinagbuhatan.date.desc()).all()
            results = [
                {
                    "id": incident.id,
                    "date": incident.date.strftime('%Y-%m-%d'),
                    "location": incident.location,
                    "cause": incident.cause
                } for incident in incidents
            ]
            return jsonify(results)
        except Exception as e:
            print(f"Error fetching incidents: {e}")
            return jsonify({"error": "Unable to fetch data"}), 500

    elif request.method == 'POST':
        try:
            data = request.json
            new_incident = Pinagbuhatan(
                date=data['date'],
                location=data['location'],
                cause=data['cause']
            )
            db.session.add(new_incident)
            db.session.commit()
            return jsonify({"status": "success"}), 201
        except Exception as e:
            print(f"Error adding new incident: {e}")
            return jsonify({"status": "error", "message": "Unable to add data"}), 500

@app.route('/api/pinagbuhatan_records/<int:id>', methods=['GET', 'PUT', 'DELETE'])
def manage_pinagbuhatan_record(id):
    if request.method == 'GET':
        try:
            incident = Pinagbuhatan.query.get(id)
            if not incident:
                return jsonify({"error": "Record not found"}), 404
            result = {
                "id": incident.id,
                "date": incident.date.strftime('%Y-%m-%d'),
                "location": incident.location,
                "cause": incident.cause
            }
            return jsonify(result)
        except Exception as e:
            print(f"Error fetching incident: {e}")
            return jsonify({"error": "Unable to fetch data"}), 500

    elif request.method == 'PUT':
        try:
            data = request.json
            incident = Pinagbuhatan.query.get(id)
            if not incident:
                return jsonify({"error": "Record not found"}), 404

            incident.date = data['date']
            incident.location = data['location']
            incident.cause = data['cause']
            db.session.commit()
            return jsonify({"status": "success"}), 200
        except Exception as e:
            print(f"Error updating incident: {e}")
            return jsonify({"status": "error", "message": "Unable to update data"}), 500

    elif request.method == 'DELETE':
        try:
            incident = Pinagbuhatan.query.get(id)
            if not incident:
                return jsonify({"error": "Record not found"}), 404

            db.session.delete(incident)
            db.session.commit()
            return jsonify({"status": "success"}), 200
        except Exception as e:
            print(f"Error deleting incident: {e}")
            return jsonify({"status": "error", "message": "Unable to delete data"}), 500

@app.route('/api/predict_today', methods=['GET'])
def predict_today():
    try:
        # Fetch weather data from an external API
        weather_api_key = "a36039c9d2288f7a4abe69090750734f"  # Replace with your actual API key
        location = "Pasig"  # Specify your location
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_api_key}&units=metric"
        
        response = requests.get(weather_url)
        response.raise_for_status()  # Raise an error for bad responses
        weather_data = response.json()

        # Extract relevant weather information
        temperature = weather_data['main']['temp']
        wind_speed = weather_data['wind']['speed']
        precipitation = 0  # This could be modified based on actual data

        today_data = {
            'Year': [datetime.datetime.now().year],
            'Month': [datetime.datetime.now().month],
            'Day': [datetime.datetime.now().day],
            'Human Error': [1],  # Adjust based on your analysis
            'Electrical': [0],  # Adjust based on your analysis
            'Temperature': [temperature],
            'Wind Speed': [wind_speed],
            'Precipitation': [precipitation]
        }

        today_df = pd.DataFrame(today_data)
        today_df_imputed = imputer.transform(today_df)
        outbreak_today_prob = clf.predict_proba(today_df_imputed)[0][1] * 100
        
        return jsonify({'percentage_chance_of_fire_outbreak_today': f'{outbreak_today_prob:.2f}%'})
    except Exception as e:
        # Detailed error logging
        print(f"Error predicting today's fire outbreak: {e}")
        return jsonify({"error": f"Unable to make prediction: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(port=5001, debug=True)
