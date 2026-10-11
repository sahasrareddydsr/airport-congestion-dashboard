
import json
from pathlib import Path
from datetime import date

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Airport Intelligence",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Airport Congestion & Flight Delay Intelligence")
st.caption("Predictive analytics | January–March 2025 dataset")

@st.cache_resource
def load_model():
    return joblib.load(
        PROJECT / "models/delay_schedule_only_logistic_regression.joblib"
    )

@st.cache_data
def load_data():
    airport = pd.read_csv(PROJECT / "results/airport_congestion_summary.csv")
    hourly = pd.read_csv(PROJECT / "results/hourly_congestion_summary.csv")
    congestion = pd.read_csv(PROJECT / "results/congestion_distribution.csv")

    with open(PROJECT / "results/metrics/schedule_only_metrics.json") as f:
        metrics = json.load(f)

    with open(
        PROJECT / "results/metrics/schedule_only_march_threshold_evaluation.json"
    ) as f:
        threshold_metrics = json.load(f)

    return airport, hourly, congestion, metrics, threshold_metrics

airport, hourly, congestion, metrics, threshold_metrics = load_data()

overview, prediction_tab, congestion_tab, evaluation_tab = st.tabs([
    "Overview", "Delay Prediction", "Airport Congestion", "Model Evaluation"
])

with overview:
    st.subheader("Project Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Flight records", "1,611,046")
    c2.metric("Airports", str(airport["Origin"].nunique()))
    c3.metric("Test period", "March 2025")

    st.info(
        "Dataset coverage: January 1–March 31, 2025. "
        "These results describe this dataset and do not establish "
        "performance on future years."
    )

    st.subheader("Congestion Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(congestion["CongestionLevel"], congestion["Percentage"])
    ax.set_ylabel("Flight records (%)")
    ax.set_xlabel("Congestion level")
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Hourly Congestion and Delay")
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(
        hourly["DepartureHour"],
        hourly["HighCongestionPercentage"],
        marker="o",
        label="High congestion (%)"
    )
    ax1.set_xlabel("Scheduled departure hour")
    ax1.set_ylabel("High congestion (%)")
    ax1.set_xticks(range(24))
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        hourly["DepartureHour"],
        hourly["MeanDepartureDelayMinutes"],
        marker="s",
        linestyle="--",
        label="Mean departure delay"
    )
    ax2.set_ylabel("Mean departure delay (minutes)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with prediction_tab:
    st.subheader("Scheduled Flight Delay Risk")
    st.warning(
        "This is an experimental model, not an operational airline system. "
        "Its March ROC-AUC is approximately 0.63. Predictions are estimates, "
        "not guarantees."
    )

    with st.form("flight_form"):
        col1, col2 = st.columns(2)

        with col1:
            flight_date = st.date_input(
                "Flight date",
                value=date(2025, 3, 15),
                min_value=date(2025, 1, 1),
                max_value=date(2025, 3, 31)
            )
            airline = st.text_input("Airline code", value="AA").strip().upper()
            origin = st.text_input("Origin airport code", value="JFK").strip().upper()
            destination = st.text_input("Destination airport code", value="LAX").strip().upper()

        with col2:
            flight_number = st.number_input(
                "Flight number", min_value=1, max_value=99999, value=100, step=1
            )
            departure_hour = st.slider("Scheduled departure hour", 0, 23, 9)
            departure_minute = st.selectbox(
                "Scheduled departure minute", [0, 15, 30, 45]
            )
            scheduled_duration = st.number_input(
                "Scheduled duration (minutes)",
                min_value=1, max_value=1500, value=360
            )
            distance = st.number_input(
                "Distance (miles)", min_value=1, max_value=10000, value=2475
            )

        submitted = st.form_submit_button("Estimate delay risk")

    if submitted:
        d = pd.Timestamp(flight_date)
        departure_hhmm = departure_hour * 100 + departure_minute
        arrival_total = departure_hour * 60 + departure_minute + scheduled_duration
        arrival_hour = (arrival_total // 60) % 24
        arrival_minute = arrival_total % 60
        arrival_hhmm = arrival_hour * 100 + arrival_minute

        row = pd.DataFrame([{
            "Year": d.year,
            "Quarter": d.quarter,
            "Month": d.month,
            "DayofMonth": d.day,
            "DayOfWeek": d.dayofweek + 1,
            "Reporting_Airline": airline,
            "Flight_Number_Reporting_Airline": flight_number,
            "Origin": origin,
            "Dest": destination,
            "CRSDepTime": departure_hhmm,
            "CRSArrTime": arrival_hhmm,
            "CRSElapsedTime": scheduled_duration,
            "Distance": distance,
            "DepartureHour": departure_hour,
            "DepartureMinute": departure_minute
        }])

        try:
            model = load_model()
            probability = float(model.predict_proba(row)[0, 1])
            st.metric("Estimated delay probability", f"{probability:.1%}")
            st.caption(
                "Model output for a departure delay of at least 15 minutes. "
                "The estimate is based on the supplied flight details."
            )
            st.progress(min(max(probability, 0.0), 1.0))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with congestion_tab:
    st.subheader("Airport Congestion Explorer")

    min_records = st.slider(
        "Minimum airport flight records",
        min_value=0,
        max_value=int(airport["FlightRecords"].max()),
        value=1000,
        step=500
    )

    filtered = airport[airport["FlightRecords"] >= min_records].copy()
    filtered = filtered.sort_values(
        "HighCongestionPercentage", ascending=False
    )

    if filtered.empty:
        st.warning("No airports meet this record-count filter.")
    else:
        st.dataframe(
            filtered[[
                "Origin", "FlightRecords", "MeanHourlyTraffic",
                "MeanTrafficIntensity", "HighCongestionPercentage",
                "MeanDepartureDelayMinutes"
            ]].round(2),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Hourly Summary")
        st.dataframe(
            hourly.round(2),
            use_container_width=True,
            hide_index=True
        )

with evaluation_tab:
    st.subheader("March Test Evaluation")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f'{metrics["accuracy"]:.1%}')
    c2.metric("F1-score", f'{metrics["f1"]:.4f}')
    c3.metric("ROC-AUC", f'{metrics["roc_auc"]:.4f}')

    st.write("**Schedule-only model at threshold 0.50**")
    st.json(threshold_metrics["threshold_0.50"])

    st.write("**Using threshold selected on February validation data**")
    st.json(threshold_metrics["february_selected_threshold"])

    st.caption(
        "The 0.45 threshold was selected using February F1. "
        "Its performance on March was not better than the 0.50 threshold "
        "on F1. The model has modest discrimination and requires improvement."
    )

st.divider()
st.caption(
    "Research prototype | Data coverage: January–March 2025 | "
    "Descriptive congestion labels may be rule-derived from traffic intensity."
)
