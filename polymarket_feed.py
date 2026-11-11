#!/usr/bin/env python3
import csv, json, sys
from datetime import datetime, timezone
from pathlib import Path
import requests

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
OUT_CSV = Path("polymarket_markets.csv")
OUT_MD = Path("polymarket_context.md")

VOLUME_KEYS = ["volumeNum", "volume24hr", "volume"]
LIQUIDITY_KEYS = ["liquidityNum", "liquidity"]

def _first(d, keys, default=None):
    for k in keys:
        v = d.get(k)
        if v not in (None, ""):
            return v
    return default

def _f(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def fetch_markets(limit=150):
    params = {"active": "true", "closed": "false", "limit": limit,
              "order": "volume24hr", "ascending": "false"}
    try:
        r = requests.get(GAMMA_URL, params=params,
                         headers={"User-Agent": "trading-ollama-local/1.0"},
                         timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"⚠️ Polymarket 请求失败：{e}", file=sys.stderr)
        return []
    if not isinstance(data, list):
        print(f"⚠️ 返回结构异常：{type(data).__name__}", file=sys.stderr)
        return []
    return data

def normalize(raw):
    rows = []
    for m in raw:
        if not isinstance(m, dict): continue
        q = (m.get("question") or "").strip()
        if not q: continue
        vol = _f(_first(m, VOLUME_KEYS))
        liq = _f(_first(m, LIQUIDITY_KEYS))
        yes = None
        try:
            p = m.get("outcomePrices")
            if isinstance(p, str): p = json.loads(p)
            if isinstance(p, list) and p: yes = _f(p[0])
        except Exception: pass
        rows.append({
            "question": q, "yes_price": yes, "volume": vol,
            "liquidity": liq,
            "end_date": (m.get("endDate") or "")[:10],
            "category": (m.get("category") or "").strip(),
            "slug": m.get("slug", ""),
        })
    rows.sort(key=lambda r: (r["volume"] is None, -(r["volume"] or 0)))
    return rows[:80]

def write_outputs(rows):
    fields = ["question","yes_price","volume","liquidity","end_date","category","slug"]
    try:
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    except OSError as e:
        print(f"⚠️ 写 CSV 失败：{e}", file=sys.stderr)

    lines = ["# Polymarket 宏观事件预期上下文",
             f"> 生成时间：{datetime.now(timezone.utc).isoformat()}",
             "> **Polymarket 是市场交易预期，不是事实，不构成股价信号。**",""]
    if not rows:
        lines.append("（未抓到有效市场数据。宏观维度应标注“数据不可用”后跳过。）")
    else:
        lines.append(f"共 {len(rows)} 个活跃市场。")
        lines.append("")
        for i, r in enumerate(rows, 1):
            yp = f"{r['yes_price']:.3f}" if r["yes_price"] is not None else "N/A"
            vt = f"{r['volume']:,.0f}" if r["volume"] is not None else "N/A"
            lt = f"{r['liquidity']:,.0f}" if r["liquidity"] is not None else "N/A"
            lines += [f"{i}. **{r['question']}**",
                      f"   - Yes 隐含概率：{yp} | 成交量：{vt} | 流动性：{lt}",
                      f"   - 截止：{r['end_date'] or 'N/A'} | 分类：{r['category'] or 'N/A'}",""]
    try:
        OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    except OSError as e:
        print(f"⚠️ 写 context.md 失败：{e}", file=sys.stderr)
    return len(rows)

def main():
    print("→ 拉取 Polymarket …")
    raw = fetch_markets(150)
    print(f"  原始返回 {len(raw)} 条")
    n = write_outputs(normalize(raw))
    if n == 0:
        print("⚠️ 无有效数据，已写出降级 context.md。", file=sys.stderr)
    else:
        print(f"✓ 写出 {OUT_CSV.name} 和 {OUT_MD.name}（{n} 个市场）")

if __name__ == "__main__":
    main()
