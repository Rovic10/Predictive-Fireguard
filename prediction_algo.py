import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import datetime

# Step 1: Load your datasets (replace 'dataset1.csv' and 'dataset2.csv' with your file paths)
df1 = pd.read_csv('firerecords.csv')
df2 = pd.read_csv('nofire.csv')

# Step 2: Combine the datasets
df = pd.concat([df1, df2], ignore_index=True)

# Step 3: Drop rows with missing target variable (y), 'Fire Occurred'
df = df.dropna(subset=['Fire Occurred'])

# Step 4: Define features (X) and labels (y)
X = df[['Year', 'Month', 'Day', 'Human Error', 'Electrical', 'Temperature', 'Wind Speed', 'Precipitation']]
y = df['Fire Occurred']  # Target variable (binary: 1 for fire, 0 for no fire)

# Step 5: Handle missing values (impute with mean for numerical data)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Step 7: Train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 8: Predict and calculate accuracy on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Step 9: Predict for future months and current day
# Generate predictions for future dates, assuming a fixed barangay and dummy data for future

# Example: Let's predict for future months in the years 2023 and 2024 for two barangays
future_dates = []
for year in [2024]:
    for month in range(1, 13):
        for barangay in ['Pinagbuhatan']:
            # Dummy data for future prediction (adjust temperature, wind, etc. as needed)
            future_dates.append({
                'Year': year,
                'Month': month,
                'Day': 1,  # Example day, can be varied
                'Human Error': 1,  # You can change this to 0 for no human error
                'Electrical': 0,   # You can change this based on the cause
                'Temperature': 30,  # Sample temperature, adjust as needed
                'Wind Speed': 10,   # Sample wind speed
                'Precipitation': 0  # Sample precipitation
            })

# Convert future data to DataFrame
future_df = pd.DataFrame(future_dates)

# Step 10: Impute missing values for future data if any (important step)
future_features = future_df[['Year', 'Month', 'Day', 'Human Error', 'Electrical', 'Temperature', 'Wind Speed', 'Precipitation']]
future_features_imputed = imputer.transform(future_features)

# Predict using the trained classifier
future_df['predicted_outbreak'] = clf.predict(future_features_imputed)

# You can also calculate the probability of outbreak
future_df['percentage_increase'] = clf.predict_proba(future_features_imputed)[:, 1] * 100  # Probability of class 1 (fire)

# Step 11: Display the future predictions
print(f"\nPredicted Fire Outbreaks for Future Months by Barangay:\n")
print(future_df[['Year', 'Month', 'predicted_outbreak', 'percentage_increase']])

# Step 12: Real-time prediction for today's data
today_data = {
    'Year': [datetime.datetime.now().year],
    'Month': [datetime.datetime.now().month],
    'Day': [datetime.datetime.now().day],  # Use the current day
    'Human Error': [1],  # Replace based on current data
    'Electrical': [0],   # Replace based on current data
    'Temperature': [35],  # Replace with real-time temperature
    'Wind Speed': [12],  # Replace with real-time wind speed
    'Precipitation': [0]  # Replace with real-time precipitation
}

today_df = pd.DataFrame(today_data)
today_df_imputed = imputer.transform(today_df)

# Predict chance of fire outbreak today
outbreak_today_prob = clf.predict_proba(today_df_imputed)[0][1] * 100
print(f'\nPercentage chance of fire outbreak today: {outbreak_today_prob:.2f}%')
