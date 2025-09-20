# 🏎️ Hamilton Podium Predictor

Machine Learning model predicting Lewis Hamilton's Formula 1 podium probabilities using historical data and real-time qualifying results.

## 🎯 Features
- **Automated Data Collection**: Fetches historical race data (2021-2025) using FastF1 API
- **Intelligent Feature Engineering**: Circuit history, recent form, qualifying position analysis
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost comparison
- **Real-time Predictions**: Live podium probability based on current qualifying results
- **Visual Analytics**: Qualifying position impact visualization

## 📊 Model Performance
- **Best Model**: Logistic Regression (84.21% accuracy, ROC AUC: 0.869)
- **Features**: Grid position, circuit history, recent form, previous finish
- **Data**: 95 races (2021-2025), 37 podiums analyzed
