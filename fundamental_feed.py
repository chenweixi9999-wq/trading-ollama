import akshare as ak
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

def to_ak_hk(ticker):
    code = ticker.upper().replace(".HK", "").strip()
    return code.zfill(5)

def fetch_indicators(ak_code):
    """獲取核心估值與盈利預期指標"""
    try:
        df = ak.stock_hk_financial_indicator_em(symbol=ak_code)
        if df is not None and not df.empty:
            # 提取最近兩期數據，重點看歷史與預期的對比
            return df.head(2).to_markdown(index=False)
        return "（無指標數據）"
    except Exception as e:
        return f"（指標獲取失敗: {e}）"

def fetch_cashflow_trend(ak_code):
    try:
        df = ak.stock_financial_hk_report_em(stock=ak_code, symbol="现金流量表", indicator="报告期")
        if df is None or df.empty: return "（无现金流数据）"
        
        # 东方财富港股接口有时候列为项目，有时候行为项目。全面搜索：
        text_data = df.to_string()
        mask = df.astype(str).apply(lambda x: x.str.contains("经营活动", na=False)).any(axis=1)
        if mask.any():
            row = df[mask].iloc[0]
            cols = df.columns[1:5] # 取最近4期
            trend = {str(c): row[c] for c in cols}
            return trend
        return "（未找到经营活动现金流科目，请人工核查）"
    except Exception as e:
        return f"（现金流获取失败: {e}）"
    except Exception as e:
        return f"（現金流獲取失敗: {e}）"

def main():
    if len(sys.argv) < 2:
        print("用法: python3 fundamental_feed.py 0700.HK")
        sys.exit(1)
    ticker = sys.argv[1].upper()
    ak_code = to_ak_hk(ticker)
    print(f"→ 港股代碼轉換: {ticker} -> {ak_code}")

    print("→ 抓取財務指標 (PE/PB/預期ROE)...")
    indicators = fetch_indicators(ak_code)

    print("→ 抓取多期經營現金流趨勢...")
    cashflow_trend = fetch_cashflow_trend(ak_code)

    lines = [
        f"# {ticker} 港股基本面與預期數據",
        f"> 生成時間：{datetime.now().isoformat()}",
        "\n## 1. 核心財務指標與預期（東方財富）",
        indicators,
        "\n## 2. 經營活動現金流淨額趨勢（核心造血能力）",
    ]
    if isinstance(cashflow_trend, dict):
        for k, v in cashflow_trend.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append(str(cashflow_trend))

    out = Path(f"{ticker}_fundamental.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ 已寫出 {out}")

if __name__ == "__main__":
    main()
