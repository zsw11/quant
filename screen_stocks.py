"""
股票筛选器 - 按行业选择候选股票，计算2024-2026区间收益率
用于扩展股票池
"""
import sys
import time
sys.path.insert(0, ".")

from data.provider import AKShareProvider

provider = AKShareProvider()

# 候选股票：每个板块选2-4只大市值龙头
# 排除已有的8只：600519, 300750, 600036, 601318, 002594, 603259, 603288, 601012
CANDIDATES = {
    # 科技/半导体
    "002230": "科大讯飞",
    "002415": "海康威视",
    "600588": "用友网络",
    "688981": "中芯国际",
    
    # 新能源/电力
    "600900": "长江电力",
    "601899": "紫金矿业",
    "003816": "中国广核",
    "600941": "中国移动",
    
    # 消费/零售
    "000858": "五粮液",
    "000568": "泸州老窖",
    "600887": "伊利股份",
    "000333": "美的集团",
    
    # 医药/生物
    "300760": "迈瑞医疗",
    "000538": "云南白药",
    "600276": "恒瑞医药",
    
    # 金融
    "600030": "中信证券",
    "601688": "华泰证券",
    "601166": "兴业银行",
    
    # 汽车/制造
    "600104": "上汽集团",
    "601127": "赛力斯",
    "000625": "长安汽车",
    
    # 军工/航天
    "600760": "中航沈飞",
    "601989": "中国重工",
}

START = "20240101"
END   = "20260227"

results = []
for code, name in CANDIDATES.items():
    try:
        df = provider.get_stock_history(code, START, END, adjust="qfq")
        if df is None or len(df) < 50:
            print(f"  {code} {name}: 数据不足，跳过")
            continue
        
        first_close = df["close"].iloc[0]
        last_close  = df["close"].iloc[-1]
        ret = (last_close / first_close - 1) * 100
        max_close = df["close"].max()
        min_close = df["close"].min()
        max_drawdown = (min_close / max_close - 1) * 100
        avg_volume = df["volume"].mean()
        
        results.append({
            "code": code,
            "name": name,
            "return_pct": ret,
            "first": first_close,
            "last": last_close,
            "max_drawdown": max_drawdown,
            "avg_volume": avg_volume,
            "days": len(df),
        })
        print(f"  {code} {name}: {ret:+.1f}% (最大回撤 {max_drawdown:.1f}%)")
        time.sleep(0.3)  # 防止限流
    except Exception as e:
        print(f"  {code} {name}: 失败 - {e}")
        time.sleep(1)

print("\n" + "="*70)
print("候选股票按收益率排序：")
print("="*70)
results.sort(key=lambda x: x["return_pct"], reverse=True)
for i, r in enumerate(results, 1):
    print(f"  {i:2d}. {r['code']} {r['name']:8s}  收益: {r['return_pct']:+7.1f}%  "
          f"回撤: {r['max_drawdown']:6.1f}%  日均量: {r['avg_volume']/10000:.0f}万手")
