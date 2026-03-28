
# 🚦 Time Traffic Prediction System (Streamlit + Python)

This project predicts **traffic congestion levels (Low / Medium / High)** in near real time
using the provided historical dataset **G1traffic.csv** (hourly vehicle counts per junction).

## What this app does
- Loads local data from `data/G1traffic.csv` (included)
- Engineers time features (hour, day, lags, rolling means)
- Derives congestion labels per junction from training data only
- Trains a RandomForest classifier with a time-aware split
- Predicts congestion for a simulated live feed
- Analyzes traffic patterns with charts and metrics

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Data
- File included: `data/G1traffic.csv`
- Required columns (auto-detected): DateTime, Junction, Vehicles
