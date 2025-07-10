# 📈 Stock Market Price Predictor

This project provides a lightweight, interactive dashboard to predict short-term stock market movements using machine learning models trained on historical OHLCV (Open, High, Low, Close, Volume) data.

By default, the app includes pre-trained models for **AAPL (Apple Inc.)**, but you can train your own models for any supported S&P 500 ticker.

---

## 🚀 Getting Started

> **Python 3.10 is required** 

### 1. Set up the virtual environment

```bash
python3.10 -m venv venv
source venv/bin/activate

```

### 2. Install required dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash 
streamlit run app.py
```

---

## ⚙️ How to Use

* **Default behavior**: The app includes trained models for the `AAPL` ticker. Use `AAPL` during your first run for immediate results.
* **Training other tickers**: If you'd like to train models on a different stock, simply choose another ticker from the dropdown within the app. The system will automatically train and cache models for your selected ticker.

---

## 🧠 Features

* Dual model architecture (e.g. RNN and LSTM) to capture different trading patterns.
* Classification and regression output: Predict both direction (Buy/Sell/Hold) and expected % price change.
* Streamlit-based interface for ease of use and fast visualization.
* Modular training and prediction pipeline with support for expansion.

---

## 📂 Project Structure


.
├── app.py                 # Main dashboard entry point
├── Main.py                # Training and prediction logic
├── models/                # Stores trained models
├── data/                  # Raw and processed data files
├── sp500_tickers.csv      # List of supported stock tickers
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
