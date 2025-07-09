# Main.py
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import MinMaxScaler

from utils import train_test_split, split_sequence
from train import train_rnn_model, train_lstm_model
from projectpro import model_snapshot, checkpoint

N_STEPS = 25

def compute_volatility(df, window=30, scale=252):
    # 1. compute returns
    df['Return']   = df['Close'].pct_change()
    # 2. rolling std
    df['RollStd']  = df['Return'].rolling(window).std()
    # 3. annualize (or horizon-scale)
    df['Volatility'] = df['RollStd'] * np.sqrt(scale)
    return df['Volatility']

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
    labels = ["Sell (-1)", "Hold (0)", "Buy (1)"]
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=0.5, linecolor='gray')
    plt.title(f"{model_name} Confusion Matrix", fontsize=14)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.show()


def make_signals(arr, buy_thr, sell_thr):
    """
    Generate simple buy/sell/hold signals based on return thresholds.
    """
    ret = np.diff(arr) / arr[:-1]
    sig = []
    for r in ret:
        if r > buy_thr:
            sig.append(1)
        elif r < sell_thr:
            sig.append(-1)
        else:
            sig.append(0)
    return np.array(sig)


def regression_metrics(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"=== {name} Regression Metrics ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")


def process_ticker(ticker, period, buy_thr, sell_thr, vol_thr, risk_thr):
    # Determine interval and lookback
    now = datetime.now()
    if period == '1h':
        interval = '1h'
        delta = timedelta(hours=1)
        start = now - timedelta(days=729)
    elif period == '1d':
        interval = '1d'
        delta = timedelta(days=1)
        start = now - timedelta(days=365*10)
    elif period == '1wk':
        interval = '1wk'
        delta = timedelta(days=1)
        start = now - timedelta(days=365*10)
    else:
        raise ValueError("Invalid period. Choose from '1h','1d','1y'.")

    # Download data and compute volatility
    dataset = yf.download(ticker, start=start, end=now, interval=interval)
    dataset.index = dataset.index.tz_localize(None)
    T = 1  # one‐hour horizon in “periods”
    dataset['Volatility'] = compute_volatility(dataset, window=30, scale=T)
    dataset.dropna(inplace=True)

    print(f"Dataset head for {ticker}:\n", dataset.head(10))
    checkpoint(ticker)

    # Split train/test by date
    tstart = start
    tend = start + (now - start) * 0.8
    train_set, test_set = train_test_split(dataset[['High']], tstart, tend)

    # Prepare test volatility labels (unused for decision)
    test_start = tend + delta
    vol_series = dataset.loc[test_start:]['Volatility'].values
    vol_labels = (vol_series[1:] > vol_thr).astype(int)

    # Scale and split
    sc = MinMaxScaler(feature_range=(0, 1))
    train_scaled = sc.fit_transform(train_set.reshape(-1, 1))
    n_steps = N_STEPS
    X_train, y_train = split_sequence(train_scaled, n_steps)
    X_train = X_train.reshape(-1, n_steps, 1)

    # Train models
    model_rnn = train_rnn_model(
        X_train, y_train, n_steps, 1, sc, test_set, dataset,
        save_model_path=f"output/{ticker}_rnn.h5"
    )
    model_snapshot(ticker)
    model_lstm = train_lstm_model(
        X_train, y_train, n_steps, 1, sc, test_set, dataset,
        save_model_path=f"output/{ticker}_lstm.h5"
    )

    # Predict
    scaled_test = sc.transform(test_set.reshape(-1, 1))
    X_test, y_test_raw = split_sequence(scaled_test, n_steps)
    X_test = X_test.reshape(-1, n_steps, 1)
    y_test = sc.inverse_transform(y_test_raw.reshape(-1,1)).flatten()
    y_pred_rnn = sc.inverse_transform(model_rnn.predict(X_test).reshape(-1,1)).flatten()
    y_pred_lstm = sc.inverse_transform(model_lstm.predict(X_test).reshape(-1,1)).flatten()

    # Estimate forecast-error volatility
    res_rnn = y_test - y_pred_rnn
    res_lstm = y_test - y_pred_lstm
    sigma_rnn = np.std(res_rnn)
    sigma_lstm = np.std(res_lstm)

    # Regression metrics
    regression_metrics(y_test, y_pred_rnn, f"{ticker} RNN")
    regression_metrics(y_test, y_pred_lstm, f"{ticker} LSTM")

    # Initial signals
    sig_true = make_signals(y_test, buy_thr, sell_thr)
    sig_rnn  = make_signals(y_pred_rnn, buy_thr, sell_thr)
    sig_lstm = make_signals(y_pred_lstm, buy_thr, sell_thr)

    # Risk-filtered signals (suppress if P(loss)>risk_thr)
    def apply_risk_filter(sig, arr, sigma):
        for i in range(len(sig)):
            if sig[i] != 0 and i+1 < len(arr):
                r_pred = (arr[i+1] - arr[i]) / arr[i]
                z = (0 - r_pred) / sigma
                p_loss = norm.cdf(z)
                if p_loss > risk_thr:
                    sig[i] = 0
        return sig

    sig_rnn = apply_risk_filter(sig_rnn, y_pred_rnn, sigma_rnn)
    sig_lstm = apply_risk_filter(sig_lstm, y_pred_lstm, sigma_lstm)

    plot_confusion_matrix(sig_true, sig_rnn, f"{ticker} RNN")
    plot_confusion_matrix(sig_true, sig_lstm, f"{ticker} LSTM")

    # Classification reports
    print(f"=== {ticker} RNN Signal Classification ===")
    print(classification_report(sig_true, sig_rnn, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(sig_true, sig_rnn))
    print(f"=== {ticker} LSTM Signal Classification ===")
    print(classification_report(sig_true, sig_lstm, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(sig_true, sig_lstm))

    # Volatility label summary
    print(f"Volatility labels (1=high): counts = {np.bincount(vol_labels)}")




def main():
    tickers = ["AAPL"]
    period = "1h"
    buy_threshold = 0.005
    sell_threshold = -0.005
    vol_thr = 1
    risk_thr = 0.4

    for ticker in tickers:
        process_ticker(ticker, period, buy_threshold, sell_threshold, vol_thr, risk_thr)

if __name__ == '__main__':
    main()
