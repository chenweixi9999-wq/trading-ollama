import akshare as ak
import pandas as pd
import yfinance as yf

def add_gold_mining_peers(df):
    """黄金矿业相关领先指标：GDX、DXY、白银、原油、铜"""
    tickers = {
        "GDX": "gdx_close",
        "DX-Y.NYB": "dxy_close",
        "SI=F": "silver_close",
        "CL=F": "oil_close",
        "HG=F": "copper_close",
    }
    for tk, col in tickers.items():
        try:
            d = yf.download(tk, period="3y", interval="1d", progress=False)
            if d.empty:
                continue
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.droplevel(1)
            s = d["Close"].squeeze().rename(col)
            df = df.join(s, how="left").ffill()
            df[f"{col}_ret_5d"] = df[col].pct_change(5)
            df[f"{col}_ret_20d"] = df[col].pct_change(20)
            print(f"[✓] {tk} 已合并")
        except Exception as e:
            print(f"[⚠] {tk} 失败：{e}")
    return df


def add_consensus_hk(df, ak_code):
    """港股一致预期：提取预测EPS、预测PE等数值作为特征"""
    try:
        forecast = ak.stock_hk_profit_forecast_et(symbol=ak_code, indicator="盈利预测概览")
        if forecast is None or forecast.empty:
            return df
        # 打印列名以便调试
        print(f"[诊断] 一致预期列名：{forecast.columns.tolist()}")
        # 尝试提取最新一期的预测EPS
        for col in forecast.columns:
            if "每股" in str(col) or "EPS" in str(col):
                try:
                    val = float(forecast[col].iloc[0])
                    df["consensus_eps"] = val
                    print(f"[✓] 一致预期 {col} = {val}")
                except Exception:
                    pass
                break
    except Exception as e:
        print(f"[⚠] 一致预期失败：{e}")
    return df
