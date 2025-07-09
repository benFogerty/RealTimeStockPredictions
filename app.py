# app.py
import os
from datetime import timedelta
import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
from pandas.tseries.offsets import BDay

from Main import train_all_intervals, predict_forward

st.set_page_config(page_title="Stock Dashboard", layout="wide")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

INTERVALS   = ["5m", "1h", "1d", "1wk"]
HIST_PERIOD = {"5m": "1d", "1h": "3d", "1d": "2wk", "1wk": "3mo"}
DELTAS      = {
    "5m": timedelta(minutes=5),
    "1h": timedelta(hours=1),
    "1d": BDay(1),
    "1wk": timedelta(weeks=1),
}

df = pd.read_csv("sp500_tickers.csv")
ALL_TICKERS = df["Ticker"].tolist()

@st.cache_data(show_spinner=False)
def cached_predict(ticker, interval, steps, buy, sell, vol, risk):
    return predict_forward(ticker, interval, steps, buy, sell, vol, risk)


def remove_ticker(tkr: str):
    """Callback to remove ticker from portfolio and clear its state."""
    if tkr in st.session_state.portfolio:
        st.session_state.portfolio.remove(tkr)
    st.session_state.pop(f"out_{tkr}", None)
    st.session_state.pop(f"params_{tkr}", None)


# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("➕ Add Ticker")
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

with st.sidebar.form("add_form", clear_on_submit=True):
    new_ticker = st.selectbox(
        "Ticker symbol",
        [""] + ALL_TICKERS,
        index=0,
        help="Select or search a S&P 500 ticker",
        key="ticker_select"
    )
    add_btn = st.form_submit_button("Add to Portfolio")
    if add_btn and new_ticker:
        if new_ticker in st.session_state.portfolio:
            st.sidebar.warning(f"{new_ticker} already added.")
        else:
            missing = [
                iv for iv in INTERVALS
                if not any(fn.startswith(f"{new_ticker}_{iv}_")
                           for fn in os.listdir(MODEL_DIR))
            ]
            if missing:
                st.sidebar.info(
                    f"Training {new_ticker} for: {', '.join(missing)}"
                )
                train_all_intervals(new_ticker)
                st.sidebar.success("Done training")
            else:
                st.sidebar.success("Models found, no retrain.")
            st.session_state.portfolio.append(new_ticker)


# ── Main Page ───────────────────────────────────────────────────────────────
st.title("Portfolio Dashboard")

for ticker in st.session_state.portfolio:

    # --- compute current params ---
    interval = st.session_state.get(f"int_{ticker}", "1h")
    steps    = st.session_state.get(f"steps_{ticker}", 5)
    buy_pct  = st.session_state.get(f"buy_{ticker}", 0.5) / 100
    sell_pct = st.session_state.get(f"sell_{ticker}", -0.5) / 100
    vol_pct  = st.session_state.get(f"vol_{ticker}", 100.0) / 100
    risk_pct = st.session_state.get(f"risk_{ticker}", 50.0) / 100
    params   = (ticker, interval, steps, buy_pct, sell_pct, vol_pct, risk_pct)

    # --- Row 1: Ticker | Predict btn | Prediction | Signal ---
    c1, remove_btn, c2, _, c3, c4 = st.columns([1.2,1,1,2,2.5,3], gap="small")
    with c1:
        st.markdown(f"### {ticker}")
    with remove_btn:
        st.write("")
        st.button(
            "Remove", key=f"rm_{ticker}",
            on_click=remove_ticker, args=(ticker,)
        )
    with c2:
        st.write("")
        if st.button("Refresh", key=f"pred_{ticker}"):
            out_new = cached_predict(*params)
            st.session_state[f"out_{ticker}"]    = out_new
            st.session_state[f"params_{ticker}"] = params
    with c3:
        st.write("")
        out    = st.session_state.get(f"out_{ticker}")
        stored = st.session_state.get(f"params_{ticker}")
        if out and stored == params:
            unit = {"5m": "minutes", "1h":"hours","1d":"days","1wk":"weeks"}[interval]
            st.write(f"Predicted: ${out['future_prices'][-1]:.2f} in {steps} {unit}")
        else:
            st.write("Predicted: —")
    with c4:
        out    = st.session_state.get(f"out_{ticker}")
        stored = st.session_state.get(f"params_{ticker}")
        if out and stored == params:
            sig_map = {
                1: "Decision: ✅ BUY",
                0: "Decision: ➖ HOLD",
               -1: "Decision: 🔻 SELL"
            }
            st.markdown(f"### {sig_map[out['signal']]}")
            if out['high_vol']:
                st.warning("⚠️ High Volatility")
        else:
            st.markdown("### Decision: –")

    # --- Row 2: Parameters expander ---
    with st.expander("⚙️ Parameters", expanded=False):
        p1, p2, p3, p4, p5 = st.columns([2,2,2,2,2])
        with p1:
            st.selectbox(
                "Interval",
                INTERVALS,
                key=f"int_{ticker}",
                help="Data granularity for model"
            )
        with p2:
            st.number_input(
                "Steps Ahead",
                min_value=1, max_value=100, value=5,
                key=f"steps_{ticker}",
                help="How many future points to forecast"
            )
        with p3:
            st.slider(
                "Buy threshold (%)",
                -10.0, 10.0, 0.5,
                step=0.1, format="%.2f",
                key=f"buy_{ticker}",
                help="Min return (%) to trigger a BUY"
            )
        with p4:
            st.slider(
                "Sell threshold (%)",
                -10.0, 10.0, -0.5,
                step=0.1, format="%.2f",
                key=f"sell_{ticker}",
                help="Max drop (%) to trigger a SELL"
            )
        with p5:
            st.slider(
                "Vol. threshold (%)",
                0.0, 300.0, 100.0,
                step=5.0, format="%.2f",
                key=f"vol_{ticker}",
                help="Volatility (%) above which warn"
            )
        st.slider(
            "Risk threshold (%)",
            0.0, 100.0, 50.0,
            step=5.0, format="%.2f",
            key=f"risk_{ticker}",
            help="Max probability of loss to allow"
        )

    # --- Chart full width ---
    out    = st.session_state.get(f"out_{ticker}")
    stored = st.session_state.get(f"params_{ticker}")
    if out and stored == params:
        raw = yf.download(
            ticker, period=HIST_PERIOD[interval], interval=interval
        ).dropna()

        close_s = (
            raw['Close'][ticker]
            if isinstance(raw['Close'], pd.DataFrame)
            else raw['Close']
        )

        hist = (
            close_s
            .rename("Actual")
            .to_frame()
            .rename_axis("Date")
            .reset_index()
        )

        last_date = hist["Date"].iloc[-1]
        delta     = DELTAS[interval]

        future_idx = [
            last_date + i*delta
            for i in range(1, len(out['future_prices'])+1)
        ]

        forecast_dates  = [last_date] + future_idx
        forecast_values = [hist["Actual"].iloc[-1]] + list(out['future_prices'])

        fc_df = pd.DataFrame({
            "Date":     forecast_dates,
            "Forecast": forecast_values
        })

        all_vals = pd.concat([hist["Actual"], fc_df["Forecast"]])
        y0, y1  = all_vals.min(), all_vals.max()

        actual_line = (
            alt.Chart(hist)
            .mark_line(color="steelblue")
            .encode(
                x='Date:T',
                y=alt.Y('Actual:Q', scale=alt.Scale(domain=[y0, y1]))
            )
        )

        forecast_line = (
            alt.Chart(fc_df)
            .mark_line(strokeDash=[5,5], color="orange")
            .encode(
                x='Date:T',
                y='Forecast:Q'
            )
        )

        st.altair_chart(
            (actual_line + forecast_line).properties(height=300),
            use_container_width=True
        )
