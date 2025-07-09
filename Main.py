# Main.py

import os
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import backend as K
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm

from utils import train_test_split, split_sequence, return_rmse
from train import train_rnn_model, train_lstm_model
from sklearn.metrics import mean_squared_error

# ── CONFIG ───────────────────────────────────────────────────────────────────
INTERVALS = {
    '5m': {'interval': '5m', 'lookback_days': 30},
    '1h':  {'interval': '1h',  'lookback_days': 729},
    '1d':  {'interval': '1d',  'lookback_days': 365*10},
    '1wk': {'interval': '1wk','lookback_days': 365*10},
}
N_STEPS   = 30
FEATURES  = 1
MODEL_DIR = 'models'

os.makedirs(MODEL_DIR, exist_ok=True)


def process_interval(ticker, key, cfg, val_split=0.2, epochs=10, batch_size=64):
    """
    Download data, split train/test, train RNN & LSTM, benchmark via RMSE, and save best model.
    """
    now   = datetime.now()
    start = now - timedelta(days=cfg['lookback_days'])
    dataset = yf.download(ticker, start=start, end=now, interval=cfg['interval'])
    dataset.index = dataset.index.tz_localize(None)

    # Split train/test by date
    tstart = start
    tend = start + (now - start) * 0.8
    train_set, test_set = train_test_split(dataset[['High']], tstart, tend)

    # Scale and split
    sc = MinMaxScaler(feature_range=(0, 1))
    train_scaled = sc.fit_transform(train_set.reshape(-1, 1))

    X_train, y_train = split_sequence(train_scaled, N_STEPS)
    X_train = X_train.reshape(-1, N_STEPS, 1)

    # Train models
    model_rnn = train_rnn_model(
        X_train, y_train, N_STEPS, 1, sc, test_set, dataset, epochs, batch_size,
        save_model_path=f"output/{ticker}_rnn.h5"
    )
    model_lstm = train_lstm_model(
        X_train, y_train, N_STEPS, 1, sc, test_set, dataset, epochs, batch_size,
        save_model_path=f"output/{ticker}_lstm.h5"
    )

    # Predict
    scaled_test = sc.transform(test_set.reshape(-1, 1))
    X_test, y_test_raw = split_sequence(scaled_test, N_STEPS)
    X_test = X_test.reshape(-1, N_STEPS, 1)
    y_test        = sc.inverse_transform(y_test_raw.reshape(-1,1)).flatten()
    y_pred_rnn    = sc.inverse_transform(model_rnn.predict(X_test).reshape(-1,1)).flatten()
    y_pred_lstm  = sc.inverse_transform(model_lstm.predict(X_test).reshape(-1,1)).flatten()

    # Compute RMSEs
    rmse_rnn  = np.sqrt(mean_squared_error(y_test, y_pred_rnn))
    rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))

    # Print them
    print(f"{ticker} @ {key} — RNN  RMSE: {rmse_rnn:.4f}")
    print(f"{ticker} @ {key} — LSTM RMSE: {rmse_lstm:.4f}")

    # Choose best
    if rmse_rnn < rmse_lstm:
        best_model, best_type, best_rmse = model_rnn, 'rnn', rmse_rnn
    else:
        best_model, best_type, best_rmse = model_lstm, 'lstm', rmse_lstm

    # Save only the best
    save_path = f"{MODEL_DIR}/{ticker}_{key}_{best_type}.h5"
    best_model.save(save_path)
    print(f"Saved best {best_type.upper()} model for {key} (RMSE={best_rmse:.4f})")


def train_all_intervals(ticker):
    ticker = ticker.upper()
    for key, cfg in INTERVALS.items():
        process_interval(ticker, key, cfg)

def predict_forward(
    ticker,
    interval,
    steps_future=5,
    buy_thr=0.005,
    sell_thr=-0.005,
    vol_thr=1.0,
    risk_thr=0.5
):
    # 0) Clear TF state
    K.clear_session()

    # 1) Download data
    cfg   = INTERVALS[interval]
    now   = datetime.now()
    start = now - timedelta(days=cfg['lookback_days'])
    df    = yf.download(ticker, start=start, end=now, interval=cfg['interval'])
    df.dropna(inplace=True)

    # ─ Compute returns & volatility ─────────────────────────────────────
    df['Return']  = df['Close'].pct_change()
    df['RollStd'] = df['Return'].rolling(window=30).std()
    df['Volatility'] = df['RollStd'] * (1**0.5)  # scale=1
    last_vol = df['Volatility'].iloc[-1]

    # 2) Scale for model input
    vals   = df['High'].values.reshape(-1,1)
    scaler = MinMaxScaler((0,1))
    scaled = scaler.fit_transform(vals)

    # 3) Load best model
    files = [f for f in os.listdir(MODEL_DIR) if f.startswith(f"{ticker}_{interval}")]
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, files[0]), compile=False)

    # 4) Roll forward predictions
    seed = scaled[-N_STEPS:].reshape(1, N_STEPS, FEATURES)
    preds, curr = [], seed.copy()
    for _ in range(steps_future):
        nxt = model(curr, training=False).numpy()
        preds.append(nxt.flatten()[0])
        curr = np.concatenate([curr[:,1:,:], nxt.reshape(1,1,1)], axis=1)

    future_prices = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

    # 5) Compute return & risk filter
    last_price   = vals[-1,0]
    final_price  = future_prices[-1]
    ret          = (final_price - last_price) / last_price

    # P(loss)
    sigma  = df['Return'].std()
    z      = (0 - ret) / sigma
    p_loss = norm.cdf(z)

    # Determine raw signal
    if   ret > buy_thr:  sig =  1
    elif ret < sell_thr: sig = -1
    else:                sig =  0

    # Apply risk filter
    if sig != 0 and p_loss > risk_thr:
        sig = 0

    # Volatility warning flag
    high_vol = last_vol > vol_thr

    return {
        "future_prices": future_prices,
        "signal":        sig,
        "high_vol":      high_vol,
        "p_loss":        p_loss,
        "last_vol":      last_vol,
        "return":        ret,
    }


if __name__ == '__main__':
    print(predict_forward("AAPL", '1h'))
