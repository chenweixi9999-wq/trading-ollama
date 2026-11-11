import yfinance as yf
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from pathlib import Path
from datetime import datetime
from fredapi import Fred
import os, sys
from feature_selector import select_features_by_importance

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_gold_mining_core(df):
    """只保留与AEM有直接传导链的特征：GDX、DXY、白银、实际利率"""
    tickers = {"GDX": "gdx", "DX-Y.NYB": "dxy", "SI=F": "silver"}
    for tk, name in tickers.items():
        try:
            d = yf.download(tk, period="3y", interval="1d", progress=False)
            if d.empty: continue
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.droplevel(1)
            s = d["Close"].squeeze().rename(f"{name}_close")
            df = df.join(s, how="left").ffill()
            df[f"{name}_ret_5d"] = df[f"{name}_close"].pct_change(5)
        except Exception as e:
            print(f"[⚠] {tk} 失败：{e}")
    # 实际利率 DFII10
    try:
        f = Fred(api_key=os.environ.get("FRED_API_KEY"))
        real_rate = f.get_series("DFII10").dropna()
        real_rate.index = pd.to_datetime(real_rate.index)
        df = df.join(real_rate.rename("real_rate"), how="left").ffill()
        df["real_rate_chg_5d"] = df["real_rate"].diff(5)
        print("[✓] 实际利率 (DFII10) 已合并")
    except Exception as e:
        print(f"[⚠] 实际利率失败：{e}")
    return df

def add_alpha101_basic(df):
    close = df['Close'].squeeze()
    volume = df['Volume'].squeeze()
    df['Alpha_4'] = -1 * close.rolling(9).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    df['Alpha_12'] = np.sign(volume.diff(1)) * (-1 * close.diff(1))
    return df

def main():
    if len(sys.argv) < 2:
        print("用法: python3 quant_predict.py AEM"); sys.exit(1)
    ticker = sys.argv[1].upper()
    print(f"→ 拉取 {ticker}...")
    df = yf.download(ticker, period="3y", interval="1d", progress=False)
    if df.empty: return
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    close, volume = df['Close'].squeeze(), df['Volume'].squeeze()

    df['SMA_20'] = close.rolling(20).mean()
    df['RSI'] = calculate_rsi(close)
    df['Bias_20'] = (close - df['SMA_20']) / df['SMA_20']
    df['Return_5d'] = close.pct_change(5)
    df = add_gold_mining_core(df)
    df = add_alpha101_basic(df)

    df['Future_Return'] = close.shift(-5) / close - 1
    df['Target'] = (df['Future_Return'] > 0).astype(int)
    df = df.dropna()

    # 候选特征池（已剔除南向、原油、铜）
    candidates = ['SMA_20', 'RSI', 'Bias_20', 'Return_5d', 'Alpha_4', 'Alpha_12']
    for c in ['gdx_ret_5d', 'dxy_ret_5d', 'silver_ret_5d', 'real_rate_chg_5d']:
        if c in df.columns: candidates.append(c)
    print(f"[候选] {candidates}")

    X_all, y = df[candidates], df['Target']
    # 用重要性粗筛，保留 top 7
    selected = select_features_by_importance(X_all, y, top_k=7)
    X = X_all[selected]

    tscv = TimeSeriesSplit(n_splits=5)
    fold_aucs, last_model = [], None
    for fold, (tr, va) in enumerate(tscv.split(X), 1):
        model = LGBMClassifier(
            n_estimators=200, learning_rate=0.02, max_depth=3,
            num_leaves=7, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.7,
            random_state=42, verbosity=-1,
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[va])[:, 1]
        auc = roc_auc_score(y.iloc[va], proba) if len(np.unique(y.iloc[va])) > 1 else np.nan
        fold_aucs.append(auc); last_model = model
        print(f"  Fold {fold}: AUC = {auc:.4f}")

    mean_auc = np.nanmean(fold_aucs)
    prob_up = last_model.predict_proba(X.iloc[[-1]])[0][1]
    imp = pd.Series(last_model.feature_importances_, index=selected).sort_values(ascending=False)

    lines = [
        f"# {ticker} 量化预测（精简特征版）",
        f"- **5日上涨概率：{prob_up*100:.2f}%**",
        f"- 平均 AUC：{mean_auc:.4f}",
        f"- 最终特征：{selected}",
        "\n## 特征重要性",
    ]
    for n, v in imp.items(): lines.append(f"- {n}: {v:.2f}")
    Path(f"{ticker}_quant_prediction.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ 完成，AUC={mean_auc:.3f}，概率={prob_up*100:.1f}%")

if __name__ == "__main__": main()
