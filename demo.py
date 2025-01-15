from fastapi import FastAPI, File, UploadFile
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from io import BytesIO
from fastapi.responses import JSONResponse
from sklearn.preprocessing import StandardScaler
import numpy as np

app = FastAPI()

# Load the pre-trained model (update the path as necessary)
model = joblib.load('c:/Users/ADMIN/Documents/PythonData1200/AI_dev/xgb_model.pkl')

# List of features expected after feature engineering
required_features = [
    "net_income_to_stockholder's_equity",
    'borrowing_dependency',
    'continuous_interest_rate_(after_tax)',
    'roa(c)_before_interest_and_depreciation_before_interest',
    'persistent_eps_in_the_last_four_seasons',
    'total_debt/total_net_worth',
    'interest-bearing_debt_interest_rate',
    'non-industry_income_and_expenditure/revenue',
    'operating_profit_per_person',
    'net_value_growth_rate',
    'net_profit_before_tax/paid-in_capital',
    'quick_ratio',
    'roa(b)_before_interest_and_depreciation_after_tax',
    'cash_flow_to_equity',
    'pre-tax_net_interest_rate',
    'debt_ratio_%',
    'operating_profit/paid-in_capital',
    'inventory/working_capital',
    'realized_sales_gross_profit_growth_rate',
    'roa(a)_before_interest_and_%_after_tax'
]

@app.post("/convert_csv_to_json_and_predict/")
async def convert_csv_to_json_and_predict(file: UploadFile = File(...)):
    try:
        # Read the CSV data from the uploaded file
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))  # Convert bytes data to DataFrame

        # Ensure the CSV has only the required features
        missing_columns = set(required_features) - set(df.columns)
        if missing_columns:
            return JSONResponse(status_code=400, content={"message": f"Missing columns: {', '.join(missing_columns)}"})

        # Preprocess: Add missing columns with default values (e.g., 0 or NaN)
        for col in missing_columns:
            df[col] = np.nan  # Use np.nan or 0, depending on the expected behavior

        # Filter the DataFrame to include only the required columns
        df_filtered = df[required_features]

        # Handle scaling (if needed) - Apply scaling using the same scaler used during training
        # Assuming you used StandardScaler during model training:
        scaler = StandardScaler()

        # Fit the scaler on the filtered training data during the training phase (not shown here).
        # Apply the scaler to the incoming data
        df_scaled = pd.DataFrame(scaler.fit_transform(df_filtered), columns=df_filtered.columns)

        # Make predictions using the pre-trained model
        predictions = model.predict(df_scaled)  # Replace with actual prediction logic

        # Add the predictions as a new column to the original DataFrame
        df['predictions'] = predictions

        # Convert the DataFrame to a dictionary or JSON format
        predictions_json = df.to_dict(orient="records")  # Converts to list of dicts

        # Return the modified DataFrame as JSON
        return JSONResponse(content={"data": predictions_json})

    except Exception as e:
        return JSONResponse(status_code=400, content={"message": str(e)})
