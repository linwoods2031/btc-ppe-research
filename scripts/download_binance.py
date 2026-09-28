import argparse, io, time, zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd, requests

COLS=["open_time","open","high","low","close","volume","close_time","quote_volume",
      "trades","taker_buy_base_volume","taker_buy_quote_volume","ignore"]
BASE="https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m"

def main(days:int):
    out=Path("data/raw"); out.mkdir(parents=True,exist_ok=True)
    end=datetime.now(timezone.utc).date()-timedelta(days=1)
    start=end-timedelta(days=days-1)
    frames=[]
    d=start
    while d<=end:
        ds=d.isoformat(); url=f"{BASE}/BTCUSDT-1m-{ds}.zip"
        r=requests.get(url,timeout=30)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                frames.append(pd.read_csv(f,header=None,names=COLS))
        d+=timedelta(days=1); time.sleep(.03)
    x=pd.concat(frames,ignore_index=True)
    # Binance archive timestamps may be ms/us depending on era. Infer safely.
    unit="us" if x["open_time"].astype("int64").median()>10**14 else "ms"
    x["open_time"]=pd.to_datetime(x["open_time"],unit=unit,utc=True)
    for c in ["open","high","low","close","volume","quote_volume","taker_buy_quote_volume"]:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.sort_values("open_time").drop_duplicates("open_time")
    expected=pd.date_range(x.open_time.min(),x.open_time.max(),freq="1min",tz="UTC")
    missing=len(expected.difference(pd.DatetimeIndex(x.open_time)))
    print(f"rows={len(x):,} missing_minutes={missing}")
    x.to_csv(out/"btcusdt_1m.csv.gz",index=False,compression="gzip")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--days",type=int,default=90)
    main(p.parse_args().days)
