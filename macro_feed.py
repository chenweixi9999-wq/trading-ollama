import os, sys
from datetime import datetime
from pathlib import Path
from fredapi import Fred

SERIES = {
    "FEDFUNDS": "联邦基金利率",
    "CPIAUCSL": "CPI（季调）",
    "UNRATE": "失业率",
    "DGS10": "10年期国债收益率",
}

def main():
    key = os.environ.get("FRED_API_KEY")
    if not key:
        print("⚠️ 未设置 FRED_API_KEY，跳过宏观数据。", file=sys.stderr)
        Path("macro_context.md").write_text("（FRED 数据不可用：未设置 API key。）", encoding="utf-8")
        return
    f = Fred(api_key=key)
    lines = ["# FRED 宏观上下文", f"> {datetime.now().isoformat()}", ""]
    for sid, name in SERIES.items():
        try:
            s = f.get_series(sid).dropna()
            if s.empty:
                lines.append(f"- **{name}**（{sid}）：无数据"); continue
            last = s.iloc[-1]; prev = s.iloc[-2] if len(s) > 1 else None
            chg = f"（前值 {prev:.2f}）" if prev is not None else ""
            lines.append(f"- **{name}**（{sid}）：最新 {last:.2f} {chg} | 日期 {s.index[-1].date()}")
        except Exception as e:
            lines.append(f"- **{name}**（{sid}）：获取失败（{e}）")
    Path("macro_context.md").write_text("\n".join(lines), encoding="utf-8")
    print("✓ 已写出 macro_context.md")

if __name__ == "__main__":
    main()
