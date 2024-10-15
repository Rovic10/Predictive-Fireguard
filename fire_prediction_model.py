import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load and Inspect Your Data
data = pd.read_csv('C:/Games/location/dataset1.csv')

# Print column names to verify
print("Column names in the DataFrame:")
print(data.columns)

# Display the first few rows of the data
print("\nFirst few rows of the data:")
print(data.head())

# Display basic statistics of the data
print("\nData statistics:")
print(data.describe())

# Check for missing values
print("\nMissing values in the data:")
print(data.isnull().sum())

# Step 3: Preprocess the Data
# 3.1: Convert 'Date' to datetime format
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y')

# 3.2: Handle missing values for numeric columns only
data.fillna(data.select_dtypes(include=[np.number]).mean(), inplace=True)

# 3.3: Convert categorical variables using one-hot encoding
if 'Cause of Fire ' in data.columns:
    data = pd.get_dummies(data, columns=['Cause of Fire '], drop_first=True)

# 3.4: Drop the 'Date' column if it's not needed in the model
if 'Date' in data.columns:
    data.drop('Date', axis=1, inplace=True)

# 3.5: Rename columns if needed to ensure consistency
data.columns = data.columns.str.strip()  # Remove leading/trailing spaces from column names

# Separate features (X) and target (y)
if 'Fire_Occurred' in data.columns:
    X = data.drop('Fire_Occurred', axis=1)  # Features
    y = data['Fire_Occurred']  # Target
else:
    raise KeyError("'Fire_Occurred' column is not found in the DataFrame. Please check the column names.")

# Step 4: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make Predictions and Evaluate the Model on Test Data
if len(y_train.unique()) > 1:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate the Model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nAccuracy: {accuracy:.2f}')

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f'ROC AUC: {roc_auc:.2f}')

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()
    plt.show()
else:
    print("Cannot compute ROC AUC score because there is only one class in the training set.")

# Step 5: Predict Fire Occurrence on New Data

# Example new_data format (without the target 'Fire_Occurred')
# Replace this with actual new data
new_data = pd.DataFrame({
    'Temperature': [30],  # Example value
    'Humidity': [40],     # Example value
    'WindSpeed': [15],    # Example value
    # Add other features as per your dataset
    # Ensure the same preprocessing steps (encoding, scaling) are applied
})

# Apply the same preprocessing steps to new_data (e.g., one-hot encoding, missing value handling)
new_data_processed = new_data.copy()

# Predict the outcome using the trained model
fire_prediction = model.predict(new_data_processed)

# Output the prediction
if fire_prediction[0] == 1:
    print("Fire is likely to occur.")
else:
    print("Fire is unlikely to occur.")

# You can also check the probability of the fire occurring
fire_prob = model.predict_proba(new_data_processed)[:, 1]
print(f"Probability of a fire outbreak: {fire_prob[0]:.2f}")
