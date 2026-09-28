from pathlib import Path
import json, sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.ppe import add_ppe_features
from src.regimes import add_regimes

def summarize(g):
    return pd.Series({"n":len(g),"p_up_5m":g.up_5m.mean(),
                      "mean_fwd_ret_5m":g.fwd_ret_5m.mean(),
                      "median_fwd_ret_5m":g.fwd_ret_5m.median()})

def main():
    inp=Path("data/raw/btcusdt_1m.csv.gz")
    out=Path("artifacts"); out.mkdir(exist_ok=True)
    x=pd.read_csv(inp,parse_dates=["open_time"])
    x=add_regimes(add_ppe_features(x))
    # Causal expanding/rolling rank approximation: bin against trailing 7d PPE distribution.
    p=x["zppe_5"]
    lo=p.rolling(10080,min_periods=1440).quantile(.1).shift(1)
    hi=p.rolling(10080,min_periods=1440).quantile(.9).shift(1)
    x["ppe_bucket"]="mid"
    x.loc[p<=lo,"ppe_bucket"]="low10"
    x.loc[p>=hi,"ppe_bucket"]="high10"
    # Primary inference: non-overlapping clock-aligned 5m samples.
    sample=x[pd.to_datetime(x.open_time,utc=True).dt.minute.mod(5)==0].dropna(subset=["fwd_ret_5m","zppe_5"])
    overall=sample.groupby("ppe_bucket",observed=True).apply(summarize,include_groups=False).reset_index()
    regimes=sample.groupby(["ny_session","trend_regime","vol_regime","ppe_bucket"],observed=True).apply(summarize,include_groups=False).reset_index()
    overall.to_csv(out/"ppe_overall.csv",index=False)
    regimes.to_csv(out/"ppe_regimes.csv",index=False)
    meta={"rows":len(x),"sample_rows":len(sample),"start":str(x.open_time.min()),"end":str(x.open_time.max())}
    (out/"run_meta.json").write_text(json.dumps(meta,indent=2))
    print(overall.to_string(index=False))

if __name__=="__main__": main()
