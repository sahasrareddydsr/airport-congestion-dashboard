# Airport Congestion & Flight Delay Intelligence

## Run locally
Install dependencies:
    pip install -r requirements.txt

Start the dashboard:
    streamlit run app.py

This research prototype uses a model evaluated on March 2025 data.
Its measured ROC-AUC is approximately 0.6275. Predictions are estimates,
not operational guarantees.

The supplied congestion labels may be rule-derived from traffic intensity.
The deployment package excludes the original flight-level dataset.
