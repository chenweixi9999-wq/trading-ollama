import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

def fetch_hkma():
    url = "https://api.hkma.gov.hk/public/market-data-and-statistics/daily-monetary-statistics/daily-figures-interbank-liquidity"
    try:
        r = requests.get(url, params={"offset": 0}, timeout=20)
        r.raise_for_status()
        latest = r.json()["result"]["records"][0]
        return {
            "date": latest.get("end_of_date", "N/A"),
            "hibor_overnight": latest.get("hibor_overnight", "N/A"),
            "hibor_1m": latest.get("hibor_fixing_1m", "N/A"),
            "aggr_balance": latest.get("closing_balance", "N/A")
        }
    except Exception as e:
        print(f"⚠️ HKMA 获取失败: {e}")
        return None

def fetch_rvd():
    urls = {
        "售价指数": "http://www.rvd.gov.hk/datagovhk/1.4M.csv",
        "租金指数": "http://www.rvd.gov.hk/datagovhk/1.3M.csv"
    }
    results = {}
    for name, url in urls.items():
        try:
            # header=None 防止把第一行长文本当列名；提取最后3行
            df = pd.read_csv(url, header=None, on_bad_lines='skip')
            recent = df.tail(3)
            # RVD 的 CSV 第 0 列是日期，第 1 列是整体指数
            results[name] = [{"日期": row[0], "指数": row[1]} for _, row in recent.iterrows()]
        except Exception as e:
            print(f"⚠️ RVD 获取失败 ({name}): {e}")
            results[name] = []
    return results

def main():
    print("→ 正在抓取 HKMA 数据...")
    hkma_data = fetch_hkma()
    
    print("→ 正在抓取 RVD 房地产数据...")
    rvd_data = fetch_rvd()

    lines = [
        "# 香港本地宏观经济与行业上下文",
        f"> 生成时间：{datetime.now().isoformat()}",
        "",
        "## 1. 香港金管局 (HKMA) 金融流动性",
    ]
    if hkma_data:
        lines.append(f"- 数据日期：{hkma_data['date']}")
        lines.append(f"- 隔夜 HIBOR：{hkma_data['hibor_overnight']}%")
        lines.append(f"- 1个月 HIBOR：{hkma_data['hibor_1m']}%")
        lines.append(f"- 银行体系总结余：{hkma_data['aggr_balance']} 百万港元")
    else:
        lines.append("- 数据不可用")

    lines.extend(["", "## 2. 差饷物业估价署 (RVD) 房地产指标"])
    for name, data in rvd_data.items():
        lines.append(f"### {name}（最新3期）")
        if data:
            for item in data:
                lines.append(f"- {item['日期']}: {item['指数']}")
        else:
            lines.append("- 数据不可用")

    out_path = Path("hk_local_context.md")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ 已写出 {out_path}")

if __name__ == "__main__":
    main()
