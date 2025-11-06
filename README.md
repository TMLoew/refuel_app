# Refuel Operations Cockpit

Refuel is our data cockpit that blends weather signals, gym attendance, and snack demand to guide merchandising, staffing, and marketing for campus fuel bars. The repo contains both the Streamlit frontend and the (in-progress) backend services used to synthesize telemetry.

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://refuelapp.streamlit.app/)

## Getting Started

```bash
pip install -r requirements.txt
streamlit run frontend/streamlit_app/Home.py
```

Drop your gym telemetry CSV into `data/gym_badges.csv`, then toggle “Use live weather API” inside the app to pull Open-Meteo context automatically.

To refresh backend datasets:

```bash
# Fetch weather aligned with the gym dataset
python backend/app/services/weather_service.py
# Rebuild product mix suggestions
python backend/app/services/pipeline.py
```

## Repository Layout

- `frontend/streamlit_app/` – multipage Streamlit app (dashboard, forecast explorer, what-if simulator, data editor, settings)
- `frontend/streamlit_app/services/` – shared loaders, weather API wrapper, synthetic data utilities
- `backend/` – placeholder for future API services
- `data/` – sample telemetry (`gym_badges.csv`)
- `logo.webp` – brand asset used across the UI

## Dev Tips

- Use the top navigation bar inside Streamlit to hop between modules.
- The Settings page exposes weather API latency, credentials placeholders, and webhook status.
- Forecast Explorer features coefficient breakdowns, residual diagnostics, and correlation heatmaps to understand drivers.

## Links

- **Streamlit Cloud**: https://refuelapp.streamlit.app/
- **Issues/feedback**: please use GitHub Issues in this repo.
