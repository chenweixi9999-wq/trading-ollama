#!/usr/bin/env python3
import argparse, sys, json
from datetime import datetime
from pathlib import Path
import requests, yfinance as yf
import pandas as pd

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3.5:9b"

SYSTEM_PROMPT = """你是一个资深的量化投资策略分析师。你的分析必须深入，拒绝表面文章。
请严格按照以下五个维度输出（用 Markdown 格式）：

## 1. 商业模式与产业链地位
公司靠什么赚钱？在上下游的议价能力如何？护城河是什么？

## 2. 财务健康与经营造血
结合提供的财务数据，分析资产负债结构。特别关注：经营性现金流是否为正？是否与净利润匹配？

## 3. 资金行为与增量信息判断
结合提供的宏观和量化因子，指出当前是否存在领先指标背离。

## 4. 宏观与行业传导 (去噪)
只提取与该股真正相关的宏观变量。

## 5. 长期底仓 + 短期网格 双轨策略 (宽网格高赔率模式)
本策略不设止盈止损，专为长线持有设计。

### 5A. 长期底仓（50%~60%）
基于基本面判断是否值得建立长期底仓，给出比例和理由。

### 5B. 短期宽网格（40%~50%）
- 当前策略已放弃高频交易，改为捕捉大波段的“宽网格”。
- 网格间距：设定为 8%~15%（根据ATR波动率决定，确保单次交易覆盖成本并有丰厚利润）。
- 赔率 b = 目标净收益率(例如15%) / 网格间距(8%) ≈ 1.8 以上。
- 胜率 p = ML 预测概率。
- 半凯利公式：f = 0.5 × (b·p - (1-p)) / b。
- **重要原则**：在宽网格模式下，只要数学期望为正（f > 0），就应当执行。不要因为 p < 50% 就拒绝交易，因为高赔率(b)完全可以弥补低胜率。

### 5C. 执行参数
- 网格上限（卖出触发价）
- 网格下限（买入触发价）
- 明确的单次下注比例和资金管理方案
"""

def load_file(path, fallback_msg, max_chars=2500):
    if not path.exists(): return fallback_msg
    try:
        t = path.read_text(encoding="utf-8").strip()
        if len(t) > max_chars: t = t[:max_chars] + "\n...(截断)"
        return t
    except Exception as e: return f"（读取失败：{e}）"

def get_quant_data(ticker, period="6mo"):
    try:
        h = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if h is None or h.empty: return "（未获取到行情）"
        close = h["Close"].squeeze()
        high = h["High"].squeeze()
        low = h["Low"].squeeze()

        sma20 = close.rolling(20).mean().iloc[-1]
        std20 = close.rolling(20).std().iloc[-1]
        boll_up, boll_dn = sma20 + 2 * std20, sma20 - 2 * std20

        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        report = f"- 最新收盘: {close.iloc[-1]:.2f}\n- SMA20: {sma20:.2f}\n- 布林带: 上轨 {boll_up:.2f} / 下轨 {boll_dn:.2f}\n- ATR(真实波幅): {atr:.2f}"
        return report
    except Exception as e:
        return f"（指标计算失败: {e}）"

def call_ollama(prompt):
    payload = {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], "stream": True, "options": {"temperature": 0.3}}
    try:
        r = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=600)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Ollama 请求失败：{e}")

    print("\n--- 模型开始生成 ---\n")
    full_content = ""
    for line in r.iter_lines():
        if line:
            try:
                chunk = json.loads(line.decode('utf-8'))
                msg = chunk.get("message", {})
                if "content" in msg and msg["content"]:
                    text = msg["content"]
                    print(text, end="", flush=True)
                    full_content += text
            except json.JSONDecodeError:
                pass
    print("\n\n--- 生成结束 ---")
    return full_content.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    a = ap.parse_args()
    ticker = a.ticker.strip().upper()

    print(f"→ 计算 {ticker} 量化指标...")
    pb = get_quant_data(ticker)

    print("→ 读取数据文件...")
    macro_ctx = load_file(Path("macro_context.md"), "")
    hk_ctx = load_file(Path("hk_local_context.md"), "")
    polymarket_ctx = load_file(Path("polymarket_context.md"), "")
    fund_ctx = load_file(Path(f"{ticker}_fundamental.md"), "")
    thesis_ctx = load_file(Path("investment_thesis.md"), "")
    pred_ctx = load_file(Path(f"{ticker}_quant_prediction.md"), "")
    backtest_ctx = load_file(Path(f"backtest_{ticker}.md"), "（无回测数据）")

    prompt = f"""分析标的：{ticker}

=== 机器学习量化预测 ===
{pred_ctx}

=== 系统回测最优参数 ===
{backtest_ctx}

=== 财务基本面 ===
{fund_ctx}

=== 量化行情 ===
{pb}

=== 宏观数据 ===
{macro_ctx}
{hk_ctx}

=== 情绪与预期 ===
{polymarket_ctx}

请严格按 system 的五节框架输出，特别是第5节！请结合回测出的最优网格间距，直接给出底仓和宽网格的执行参数，不要因为胜率低就拒绝给出网格策略。
"""
    try:
        body = call_ollama(prompt)
    except RuntimeError as e:
        print(f"\n✗ {e}", file=sys.stderr); sys.exit(1)

    out = Path(f"strategy_{ticker}_{datetime.now():%Y%m%d}.md")
    if body:
        header = f"# {ticker} 专属策略分析 (宽网格版)\n> {datetime.now().isoformat()}\n\n"
        out.write_text(header + body + "\n", encoding="utf-8")
        print(f"\n✓ 报告已保存：{out}")

if __name__ == "__main__":
    main()
