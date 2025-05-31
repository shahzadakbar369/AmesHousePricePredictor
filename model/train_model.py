import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv('../dataset/AmesHousing.csv')

features = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area', 'Total Bsmt SF', 'Full Bath', 'Year Built']
target = 'SalePrice'
X = df[features]
y = df[target]

imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/house_price_model.pkl')
joblib.dump(imputer, 'model/imputer.pkl')

importances = model.feature_importances_
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("House Price Predictor - Feature Importance")

os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/feature_importance.png')
plt.close()
