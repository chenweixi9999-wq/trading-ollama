import yfinance as yf
from pathlib import Path
from datetime import datetime
import sys

def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AEM"
    print(f"→ 抓取美股 {ticker} 基本面...")
    t = yf.Ticker(ticker)
    info = t.info

    lines = [
        f"# {ticker} 美股基本面",
        f"> {datetime.now().isoformat()}",
        "",
        f"- 市盈率 (TTM): {info.get('trailingPE', 'N/A')}",
        f"- 市净率: {info.get('priceToBook', 'N/A')}",
        f"- 总市值: {info.get('marketCap', 'N/A')}",
        f"- 股息率: {info.get('dividendYield', 'N/A')}",
        f"- ROE: {info.get('returnOnEquity', 'N/A')}",
        f"- 营收增长率: {info.get('revenueGrowth', 'N/A')}",
        f"- 经营现金流: {info.get('operatingCashflow', 'N/A')}",
        f"- 自由现金流: {info.get('freeCashflow', 'N/A')}",
    ]

    # 现金流趋势
    try:
        cf = t.cashflow
        if cf is not None and not cf.empty:
            lines.append("\n## 现金流趋势（最近3期）")
            row = cf.loc["Operating Cash Flow"] if "Operating Cash Flow" in cf.index else None
            if row is not None:
                for col in row.index[:3]:
                    lines.append(f"- {col}: {row[col]}")
    except Exception as e:
        lines.append(f"\n（现金流获取失败：{e}）")

    out = Path(f"{ticker}_fundamental.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ 已写出 {out}")

if __name__ == "__main__":
    main()
