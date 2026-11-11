import akshare as ak
import pandas as pd

def merge_south_flow(df):
    """Southbound overall net flow as market feature."""
    try:
        flow = ak.stock_hsgt_fund_flow_summary_em()
        south = flow[flow['资金方向'] == '南向'].copy()
        if south.empty:
            print("[WARN] Southbound data is empty, skip.")
            return df

        date_col = '交易日' if '交易日' in south.columns else south.columns[0]
        south[date_col] = pd.to_datetime(south[date_col])
        south = south.set_index(date_col).sort_index()

        val_col = None
        for c in ['资金净流入', '成交净买额', '净流入']:
            if c in south.columns:
                val_col = c
                break
        if val_col is None:
            print(f"[WARN] No net flow column. Columns: {south.columns.tolist()}")
            return df

        print(f"[INFO] Southbound flow column used: {val_col}")
        flow_series = south[val_col].rename("south_net_flow")

        merged = df.join(flow_series, how="left")
        merged["south_net_flow"] = merged["south_net_flow"].ffill().fillna(0)
        merged["south_flow_5d"] = merged["south_net_flow"].rolling(5).sum()
        merged["south_flow_20d"] = merged["south_net_flow"].rolling(20).sum()
        return merged
    except Exception as e:
        print(f"[WARN] Southbound merge failed: {e}")
        return df
