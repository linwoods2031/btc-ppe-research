# BTC PPE Research

Research-only study of **Price Propulsion Efficiency (PPE)** on BTCUSDT 1-minute Binance spot data.

## Question

Does the amount of directional price displacement produced per unit of *normalized trading activity* contain incremental information about BTC's next 5-minute return, and does that information depend on market regime?

No live trading or order execution is included.

## Core design

- Data: Binance public BTCUSDT spot 1m klines.
- Initial window: latest ~90 complete UTC days.
- Inputs include OHLC, quote volume, trade count, taker-buy quote volume.
- PPE windows: 1m, 3m, 5m, 15m.
- Derivatives: PPE slope, acceleration, PPE-DIF/DEA/HIST.
- Order flow proxy: `2 * taker_buy_quote / quote_volume - 1`.
- Regimes: weekend/weekday, New York session buckets, trend, volatility and liquidity.
- Target: forward 5-minute log return.
- Primary inference uses non-overlapping 5-minute observations to reduce label dependence.
- All rolling features are backward-looking only.

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/download_binance.py --days 90
python scripts/run_study.py
pytest -q
```

Outputs are written to `artifacts/`. Raw data is intentionally gitignored.

## Interpretation discipline

PPE is a hypothesis, not a proven alpha factor. High PPE can reflect changing supply/demand, thinner liquidity, or external shocks. Results are segmented by market state and compared with baseline price/volume information before any predictive claim is made.
