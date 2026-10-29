# House Price Predictor

A Streamlit app that predicts California house prices from area and location details.

## Run locally

```bash
pip install streamlit pandas scikit-learn joblib
streamlit run houseprice_pro.py
```

The app loads `My_Model.joblib` from the repository root.

## Model

- **Dataset:** California Housing (`sklearn.datasets.fetch_california_housing`)
- **Algorithm:** `HistGradientBoostingRegressor`
- **Training:** 80/20 split with `random_state=432`
- **Parameters:** `max_iter=300`, `learning_rate=0.05`
- **Inputs:** median income, house age, average rooms, average bedrooms, population, average occupancy, latitude, and longitude
- **Output:** predicted median house value in US dollars; the dataset target is scaled in units of $100,000

Retrain with `python model.py` if needed. This project is for educational use and is not a professional property valuation.
