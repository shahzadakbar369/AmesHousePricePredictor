import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv('../dataset/AmesHousing.csv')

# Select features and target
features = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Built']
target = 'SalePrice'
X = df[features]
y = df[target]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=features)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Save model and imputer
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/house_price_model.pkl')
joblib.dump(imputer, 'model/imputer.pkl')

# Plot and save feature importance
importances = model.feature_importances_
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("House Price Predictor - Feature Importance")

# Create output directory if not exists
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/feature_importance.png')
plt.close()
