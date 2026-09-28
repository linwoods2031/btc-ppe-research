import pandas as pd
from src.ppe import add_ppe_features

def test_future_return_is_forward_only():
    n=300
    x=pd.DataFrame({"open_time":pd.date_range("2026-01-01",periods=n,freq="min",tz="UTC"),
                    "close":[100+i*.01 for i in range(n)],
                    "quote_volume":[1000.0]*n,
                    "taker_buy_quote_volume":[500.0]*n})
    y=add_ppe_features(x)
    assert y.loc[100,"fwd_ret_5m"]>0
    assert pd.isna(y.loc[n-1,"fwd_ret_5m"])

def test_balanced_taker_flow_is_zero():
    n=300
    x=pd.DataFrame({"open_time":pd.date_range("2026-01-01",periods=n,freq="min",tz="UTC"),
                    "close":[100.0]*n,"quote_volume":[1000.0]*n,
                    "taker_buy_quote_volume":[500.0]*n})
    y=add_ppe_features(x)
    assert abs(y.loc[100,"taker_imbalance"])<1e-12
