import sys, numpy as np, pandas as pd, yfinance as yf
from pathlib import Path
from backtesting import Backtest, Strategy
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore') # 屏蔽多余的警告

def ml_prob_series(close, volume):
    sma20 = close.rolling(20).mean()
    delta = close.diff()
    gain = (delta.where(delta>0,0)).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    rsi = 100 - (100/(1+gain/loss))
    X = pd.DataFrame({"sma20":sma20,"rsi":rsi,"ret5":close.pct_change(5)}).dropna()
    y = (close.shift(-5)/close-1 > 0).astype(int).reindex(X.index)
    prob = pd.Series(index=X.index, dtype=float)
    tscv = TimeSeriesSplit(n_splits=5)
    for tr,va in tscv.split(X):
        m = LGBMClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, num_leaves=7, random_state=42, verbosity=-1)
        m.fit(X.iloc[tr], y.iloc[tr])
        prob.iloc[va] = m.predict_proba(X.iloc[va])[:,1]
    return prob

class GridStrat(Strategy):
    grid_pct = 0.08      
    base_pos = 0.3       
    
    def init(self): 
        self.p = self.I(lambda: self.data.df["ml_prob"].values)
        
    def next(self):
        p = self.p[-1]
        if np.isnan(p): return
        price = self.data.Close[-1]
        
        if not self.position:
            if p >= 0.45: 
                self.buy(size=self.base_pos)
        elif p >= 0.45 and self.position.size < 0.8:
            if price <= self.position.entry_price * (1 - self.grid_pct): 
                self.buy(size=0.2)
        elif p <= 0.35: 
            self.position.close(portion=0.2)

def main():
    ticker = sys.argv[1].upper() if len(sys.argv)>1 else "AEM"
    print(f"→ 下载 {ticker} 数据...")
    df = yf.download(ticker, period="3y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    
    close, vol = df["Close"].squeeze(), df["Volume"].squeeze()
    df["ml_prob"] = ml_prob_series(close, vol).reindex(df.index)
    
    bt = Backtest(df[["Open","High","Low","Close","Volume","ml_prob"]], GridStrat,
                  cash=100000, commission=0.0005, exclusive_orders=True)
                  
    print("→ 正在自动寻优 8%~15% 宽网格参数...")
    # 加上 max_tries=1 解决多进程报错问题
    opt = bt.optimize(
        grid_pct=[0.08, 0.10, 0.12, 0.15], 
        base_pos=[0.3, 0.4, 0.5],
        maximize="Sharpe Ratio", 
        method="grid",
        max_tries=1
    )
    
    print(f"\n✓ 最优参数：网格间距={opt._strategy.grid_pct*100:.0f}%, 底仓={opt._strategy.base_pos*100:.0f}%")
    print(f"✓ 夏普比率：{opt['Sharpe Ratio']:.3f}")
    print(f"✓ 累计收益：{opt['Return [%]']:.2f}%")
    print(f"✓ 交易次数：{opt['# Trades']}")
    
    out = f"backtest_{ticker}.md"
    Path(out).write_text(
        f"# {ticker} 回测报告 (宽网格优化版)\n"
        f"- 最优网格间距：{opt._strategy.grid_pct*100:.0f}%\n"
        f"- 最优底仓比例：{opt._strategy.base_pos*100:.0f}%\n"
        f"- 夏普比率：{opt['Sharpe Ratio']:.3f}\n"
        f"- 累计收益：{opt['Return [%]']:.2f}%\n"
        f"- 最大回撤：{opt['Max. Drawdown [%]']:.2f}%\n"
        f"- 交易次数：{opt['# Trades']}\n", 
        encoding="utf-8"
    )
    print(f"✓ 已写出 {out} (现在可以打开啦)")

if __name__ == "__main__": 
    main()
