
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "https://newproject-gvg1.onrender.com"
HISTORY_API_URL = f"{API_BASE_URL}/history"
FORECAST_API_URL = f"{API_BASE_URL}/forecast"


# =========================================================
# 2. LOAD HISTORICAL SALES FROM BACKEND
# =========================================================
@st.cache_data(ttl=3600)
def load_historical_sales():
    response = requests.get(
        HISTORY_API_URL,
        timeout=60
    )
    response.raise_for_status()

    data = response.json()

    if "historical_sales" not in data:
        raise ValueError(
            "The backend response does not contain historical_sales."
        )

    history_df = pd.DataFrame(data["historical_sales"])

    if history_df.empty:
        raise ValueError("The backend returned no historical sales data.")

    required_columns = {"Month", "Sales"}

    if not required_columns.issubset(history_df.columns):
        raise ValueError(
            "Historical data must contain Month and Sales columns."
        )

    history_df["Month"] = pd.to_datetime(
        history_df["Month"],
        errors="coerce"
    )
    history_df["Sales"] = pd.to_numeric(
        history_df["Sales"],
        errors="coerce"
    )

    history_df = history_df.dropna(
        subset=["Month", "Sales"]
    )

    history_df["Month"] = (
        history_df["Month"].dt.to_period("M").dt.to_timestamp()
    )

    history = (
        history_df.groupby("Month")["Sales"]
        .sum()
        .sort_index()
    )

    if len(history) < 12:
        raise ValueError(
            "At least 12 months of historical sales are required."
        )

    return history


# =========================================================
# 3. PLOTLY CHART HELPER
# =========================================================
def create_sales_chart(
    historical_data,
    forecast_data=None,
    title="Monthly Sales",
    height=450
):
    fig = go.Figure()

    if historical_data is not None and not historical_data.empty:
        hist = historical_data.sort_index()

        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=hist.values,
                mode="lines+markers",
                name="Historical Sales",
                line=dict(color="#1687E8", width=2.5),
                marker=dict(size=5),
                hovertemplate=(
                    "Month: %{x|%b %Y}<br>"
                    "Historical Sales: %{y:,.2f}"
                    "<extra></extra>"
                )
            )
        )

    if forecast_data is not None and not forecast_data.empty:
        forecast = forecast_data.sort_values("Month")

        fig.add_trace(
            go.Scatter(
                x=forecast["Month"],
                y=forecast["Predicted Sales"],
                mode="lines+markers",
                name="Forecast Sales",
                line=dict(color="#FF9F43", width=3, dash="dash"),
                marker=dict(size=9, symbol="diamond"),
                hovertemplate=(
                    "Month: %{x|%b %Y}<br>"
                    "Forecast Sales: %{y:,.2f}"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=65, b=30),
        xaxis=dict(
            title="Month",
            type="date",
            tickformat="%b %Y",
            dtick="M1",
            tickangle=-45,
            showgrid=False,
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            title="Sales",
            tickformat=",.0f",
            rangemode="tozero",
            gridcolor="rgba(150,150,150,0.2)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    return fig


# =========================================================
# 4. DASHBOARD HEADER
# =========================================================
st.title("📈 Sales Forecasting Dashboard")

st.write(
    "Analyze historical monthly sales and generate forecasts "
    "using a Ridge Regression model served through a FastAPI backend."
)

try:
    with st.spinner("Loading historical sales data..."):
        monthly_sales = load_historical_sales()

except Exception as error:
    st.error(
        f"Unable to load historical sales from the backend: {error}"
    )
    st.info(
        "Check that the Render backend is running and that "
        "the /history endpoint is available."
    )
    st.stop()

first_historical_month = monthly_sales.index.min()
last_historical_month = monthly_sales.index.max()

# The first forecast month needs 12 preceding historical months.
earliest_forecast_month = (
    first_historical_month + pd.DateOffset(months=12)
)

st.info(
    f"Historical data: "
    f"{first_historical_month.strftime('%B %Y')} to "
    f"{last_historical_month.strftime('%B %Y')}. "
    f"Earliest eligible forecast month: "
    f"{earliest_forecast_month.strftime('%B %Y')}. "
    "Forecast values are estimates, not actual future sales."
)


# =========================================================
# 5. SIDEBAR FORECAST SETTINGS
# =========================================================
st.sidebar.header("⚙️ Forecast Settings")
st.sidebar.subheader("Select Forecast Start Date")

month_names = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

available_years = [2011, 2012, 2013, 2014]

selected_year = st.sidebar.selectbox(
    "Select Year",
    options=available_years,
    index=3,
    key="forecast_year"
)

selected_month = st.sidebar.selectbox(
    "Select Month",
    options=list(range(1, 13)),
    format_func=lambda month: month_names[month],
    index=0,
    key="forecast_month"
)

selected_forecast_start = pd.Timestamp(
    year=selected_year,
    month=selected_month,
    day=1
)

duration_days = st.sidebar.selectbox(
    "Forecast Duration",
    options=[30, 60, 90],
    index=0,
    format_func=lambda days: f"{days} days",
    key="forecast_duration"
)

horizon = duration_days // 30

st.sidebar.caption(
    "The model predicts monthly sales: "
    "30 days = 1 month, 60 days = 2 months, "
    "90 days = 3 months."
)

st.sidebar.divider()
st.sidebar.subheader("Dataset Information")

st.sidebar.write(
    f"First historical month: "
    f"**{first_historical_month.strftime('%B %Y')}**"
)

st.sidebar.write(
    f"Last historical month: "
    f"**{last_historical_month.strftime('%B %Y')}**"
)

st.sidebar.write(
    f"Historical months available: **{len(monthly_sales)}**"
)

st.sidebar.write(
    f"Selected forecast start: "
    f"**{selected_forecast_start.strftime('%B %Y')}**"
)

valid_forecast_date = (
    selected_forecast_start >= earliest_forecast_month
)

if not valid_forecast_date:
    st.sidebar.warning(
        "This month does not have 12 preceding historical months. "
        f"Select {earliest_forecast_month.strftime('%B %Y')} "
        "or a later month."
    )


# =========================================================
# 6. HISTORICAL SALES OVERVIEW
# =========================================================
st.divider()
st.subheader("📊 Historical Sales Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Historical Sales",
    f"{monthly_sales.sum():,.2f}"
)
col2.metric(
    "Average Monthly Sales",
    f"{monthly_sales.mean():,.2f}"
)
col3.metric(
    "Historical Months",
    f"{len(monthly_sales)}"
)

st.subheader("Historical Monthly Sales")

historical_fig = create_sales_chart(
    historical_data=monthly_sales,
    title="Historical Monthly Sales",
    height=450
)

st.plotly_chart(
    historical_fig,
    use_container_width=True,
    config={"displaylogo": False}
)

with st.expander("View Historical Sales Data"):
    historical_table = pd.DataFrame({
        "Month": monthly_sales.index.strftime("%B %Y"),
        "Historical Sales": monthly_sales.values.round(2)
    })

    st.dataframe(
        historical_table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 7. FORECAST THROUGH RENDER API
# =========================================================
st.divider()
st.subheader("🔮 Sales Forecast")

st.write(
    f"Selected start month: "
    f"**{selected_forecast_start.strftime('%B %Y')}**"
)
st.write(f"Forecast duration: **{duration_days} days**")

if st.button(
    "Generate Sales Forecast",
    type="primary",
    use_container_width=True,
    disabled=not valid_forecast_date
):
    try:
        with st.spinner("Requesting forecast from the backend..."):
            payload = {
                "year": int(selected_year),
                "month": int(selected_month),
                "horizon": int(horizon)
            }

            response = requests.post(
                FORECAST_API_URL,
                json=payload,
                timeout=120
            )

        if response.status_code >= 400:
            try:
                error_data = response.json()
                detail = error_data.get("detail", response.text)
            except ValueError:
                detail = response.text

            st.error(
                f"Backend error ({response.status_code}): {detail}"
            )

        else:
            result = response.json()

            if not result.get("forecasts"):
                st.error("The backend returned no forecast results.")

            else:
                forecast_df = pd.DataFrame(result["forecasts"])

                forecast_df["Month"] = pd.to_datetime(
                    forecast_df["month"],
                    errors="coerce"
                )

                forecast_df["Predicted Sales"] = pd.to_numeric(
                    forecast_df["predicted_sales"],
                    errors="coerce"
                ).clip(lower=0)

                forecast_df = forecast_df.dropna(
                    subset=["Month", "Predicted Sales"]
                )

                forecast_df = (
                    forecast_df[["Month", "Predicted Sales"]]
                    .sort_values("Month")
                    .reset_index(drop=True)
                )

                if forecast_df.empty:
                    st.error(
                        "The backend returned invalid forecast values."
                    )

                elif (
                    forecast_df["Month"].iloc[0]
                    != selected_forecast_start
                ):
                    st.error(
                        "The forecast start month returned by the "
                        "backend does not match the selected month."
                    )

                elif len(forecast_df) != horizon:
                    st.error(
                        f"Expected {horizon} monthly predictions, "
                        f"but received {len(forecast_df)}."
                    )

                else:
                    st.success(
                        "Forecast generated successfully through the API!"
                    )

                    # ---------------------------------------------
                    # 8. FORECAST DATE RANGE
                    # ---------------------------------------------
                    range_start = forecast_df["Month"].min()
                    range_end = (
                        forecast_df["Month"].max()
                        + pd.offsets.MonthEnd(0)
                    )
                    calendar_days = (
                        range_end - range_start
                    ).days + 1

                    date_col1, date_col2, date_col3 = st.columns(3)

                    date_col1.metric(
                        "Forecast Start Date",
                        range_start.strftime("%d %b %Y")
                    )
                    date_col2.metric(
                        "Forecast End Date",
                        range_end.strftime("%d %b %Y")
                    )
                    date_col3.metric(
                        "Calendar Days Covered",
                        f"{calendar_days} days"
                    )

                    st.caption(
                        "Calendar days show the forecast date range, "
                        "not daily predictions."
                    )

                    # ---------------------------------------------
                    # 9. FORECAST KPIs
                    # ---------------------------------------------
                    metric1, metric2, metric3 = st.columns(3)

                    metric1.metric(
                        "Total Predicted Sales",
                        f"{forecast_df['Predicted Sales'].sum():,.2f}"
                    )

                    metric2.metric(
                        "Average Monthly Forecast",
                        f"{forecast_df['Predicted Sales'].mean():,.2f}"
                    )

                    metric3.metric(
                        "Forecast Duration",
                        f"{duration_days} days"
                    )

                    # ---------------------------------------------
                    # 10. FORECAST TABLE
                    # ---------------------------------------------
                    st.divider()
                    st.subheader("📋 Monthly Forecast Results")

                    display_forecast = forecast_df.copy()
                    display_forecast["Month"] = (
                        display_forecast["Month"].dt.strftime("%B %Y")
                    )
                    display_forecast["Predicted Sales"] = (
                        display_forecast["Predicted Sales"].round(2)
                    )

                    st.dataframe(
                        display_forecast,
                        use_container_width=True,
                        hide_index=True
                    )

                    # ---------------------------------------------
                    # 11. FORECAST CHART
                    # ---------------------------------------------
                    st.subheader("📈 Predicted Monthly Sales")

                    forecast_fig = create_sales_chart(
                        historical_data=None,
                        forecast_data=forecast_df,
                        title="Predicted Monthly Sales",
                        height=400
                    )

                    st.plotly_chart(
                        forecast_fig,
                        use_container_width=True,
                        config={"displaylogo": False}
                    )

                    # ---------------------------------------------
                    # 12. HISTORICAL VS FORECAST
                    # ---------------------------------------------
                    st.subheader("Historical vs Forecast Sales")

                    recent_history = monthly_sales[
                        monthly_sales.index < selected_forecast_start
                    ].tail(12)

                    comparison_fig = create_sales_chart(
                        historical_data=recent_history,
                        forecast_data=forecast_df,
                        title="Historical vs Forecast Sales",
                        height=500
                    )

                    chart_start = (
                        recent_history.index.min()
                        if not recent_history.empty
                        else forecast_df["Month"].min()
                    )

                    chart_end = (
                        forecast_df["Month"].max()
                        + pd.offsets.MonthEnd(0)
                    )

                    comparison_fig.update_xaxes(
                        range=[chart_start, chart_end]
                    )

                    st.plotly_chart(
                        comparison_fig,
                        use_container_width=True,
                        config={"displaylogo": False}
                    )

                    # ---------------------------------------------
                    # 13. DOWNLOAD FORECAST CSV
                    # ---------------------------------------------
                    st.divider()
                    st.subheader("Download Forecast")

                    download_df = forecast_df.copy()
                    download_df["Month"] = (
                        download_df["Month"].dt.strftime("%Y-%m-%d")
                    )
                    download_df["Predicted Sales"] = (
                        download_df["Predicted Sales"].round(2)
                    )

                    st.download_button(
                        label="⬇️ Download Forecast as CSV",
                        data=download_df.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name=(
                            f"sales_forecast_{duration_days}_days.csv"
                        ),
                        mime="text/csv",
                        use_container_width=True
                    )

                    st.caption(
                        "Forecast values are estimates and may differ "
                        "from actual sales."
                    )

    except requests.exceptions.RequestException as error:
        st.error(
            f"Could not connect to the forecasting backend: {error}"
        )

    except Exception as error:
        st.error(f"Forecast generation failed: {error}")


# =========================================================
# 14. FOOTER
# =========================================================
st.divider()

st.caption(
    "Sales Forecasting Dashboard | Ridge Regression | "
    "Monthly Time-Series Forecasting"
)
