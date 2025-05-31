import os
import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, '..', 'model', 'house_price_model.pkl')
imputer_path = os.path.join(BASE_DIR, '..', 'model', 'model', 'imputer.pkl')

model = joblib.load(model_path)
imputer = joblib.load(imputer_path)

root = tk.Tk()
root.title("House Price Predictor")

entries = {}
fields = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

for idx, field in enumerate(fields):
    tk.Label(root, text=field).grid(row=idx, column=0)
    entry = tk.Entry(root)
    entry.grid(row=idx, column=1)
    entries[field] = entry

def predict_price():
    try:
        values = [float(entries[field].get()) for field in fields]
        values = np.array(values).reshape(1, -1)
        values = imputer.transform(values)
        price = model.predict(values)[0]
        messagebox.showinfo("Predicted Price", f"Estimated House Price: ${price:,.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Please enter valid numbers for all fields.\n\nDetails:\n{e}")

tk.Button(root, text="Predict Price", command=predict_price).grid(row=len(fields), columnspan=2)
root.mainloop()
