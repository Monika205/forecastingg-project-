
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import joblib

# =========================================================
# 1. INITIALIZE API
# =========================================================
app = FastAPI(
    title="Sales Forecasting API",
    version="0.2.0"
)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "sales_forecasting_bundle.joblib"
)


# =========================================================
# 2. LOAD MODEL BUNDLE AND HISTORICAL SALES
# =========================================================
try:
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    monthly_sales = bundle["monthly_sales"]
    feature_columns = list(bundle["feature_columns"])

    if isinstance(monthly_sales, pd.DataFrame):
        if monthly_sales.shape[1] != 1:
            raise ValueError(
                "monthly_sales must contain exactly one sales column."
            )
        monthly_sales = monthly_sales.iloc[:, 0]

    monthly_sales = pd.Series(monthly_sales).copy()

    monthly_sales.index = pd.to_datetime(
        monthly_sales.index,
        errors="coerce"
    )

    monthly_sales = monthly_sales[
        ~monthly_sales.index.isna()
    ]

    monthly_sales = monthly_sales.astype(float).dropna()

    monthly_sales.index = (
        monthly_sales.index.to_period("M").to_timestamp()
    )

    monthly_sales = (
        monthly_sales.groupby(level=0).sum().sort_index()
    )

    if len(monthly_sales) < 12:
        raise ValueError(
            "At least 12 historical months are required."
        )

except Exception as error:
    raise RuntimeError(
        f"Could not load model bundle: {error}"
    )


first_historical_month = monthly_sales.index.min()
last_historical_month = monthly_sales.index.max()


# =========================================================
# 3. FORECAST REQUEST SCHEMA
# =========================================================
class ForecastRequest(BaseModel):
    year: int = Field(ge=2011, le=2014)
    month: int = Field(ge=1, le=12)
    horizon: int = Field(default=3, ge=1, le=3)


# =========================================================
# 4. GENERATE FORECAST
# =========================================================
def generate_forecast(start_date, horizon):
    history = monthly_sales[
        monthly_sales.index < start_date
    ].copy()

    history = history.astype(float).sort_index()

    if len(history) < 12:
        raise ValueError(
            "At least 12 months of historical data before "
            "the selected forecast month are required. "
            "Please select a later month."
        )

    expected_features = [
        "month",
        "year_number",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_12",
        "rolling_mean_3",
        "rolling_mean_12"
    ]

    missing_features = [
        feature for feature in expected_features
        if feature not in feature_columns
    ]

    unexpected_features = [
        feature for feature in feature_columns
        if feature not in expected_features
    ]

    if missing_features or unexpected_features:
        raise ValueError(
            f"Model feature mismatch. "
            f"Missing: {missing_features}; "
            f"unexpected: {unexpected_features}"
        )

    predictions = []

    for _ in range(horizon):
        next_month = (
            history.index.max() + pd.offsets.MonthBegin(1)
        )

        features = {
            "month": next_month.month,
            "year_number": (
                next_month.year - first_historical_month.year
            ),
            "lag_1": history.iloc[-1],
            "lag_2": history.iloc[-2],
            "lag_3": history.iloc[-3],
            "lag_12": history.iloc[-12],
            "rolling_mean_3": history.iloc[-3:].mean(),
            "rolling_mean_12": history.iloc[-12:].mean()
        }

        input_data = pd.DataFrame(
            [[features[column] for column in feature_columns]],
            columns=feature_columns
        )

        predicted_sales = float(
            model.predict(input_data)[0]
        )

        predicted_sales = max(0.0, predicted_sales)

        history.loc[next_month] = predicted_sales

        predictions.append({
            "month": next_month.strftime("%Y-%m-%d"),
            "predicted_sales": predicted_sales
        })

    return predictions


# =========================================================
# 5. HOME ENDPOINT
# =========================================================
@app.get("/")
def home():
    return {
        "message": "Sales Forecasting API is running",
        "status": "healthy"
    }


# =========================================================
# 6. HEALTH CHECK ENDPOINT
# =========================================================
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =========================================================
# 7. HISTORICAL SALES ENDPOINT
# =========================================================
@app.get("/history")
def history():
    return {
        "first_historical_month": (
            first_historical_month.strftime("%Y-%m-%d")
        ),
        "last_historical_month": (
            last_historical_month.strftime("%Y-%m-%d")
        ),
        "total_historical_months": len(monthly_sales),
        "historical_sales": [
            {
                "Month": date.strftime("%Y-%m-%d"),
                "Sales": float(sales)
            }
            for date, sales in monthly_sales.items()
        ]
    }


# =========================================================
# 8. FORECAST ENDPOINT
# =========================================================
@app.post("/forecast")
def forecast(request: ForecastRequest):
    try:
        start_date = pd.Timestamp(
            year=request.year,
            month=request.month,
            day=1
        )

        results = generate_forecast(
            start_date=start_date,
            horizon=request.horizon
        )

        return {
            "forecast_start": results[0]["month"],
            "forecast_end": results[-1]["month"],
            "horizon": request.horizon,
            "forecasts": results
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {error}"
        )
