"""
MACD底背离筛选模块
===================
识别条件：价格创新低，但DIF没有创新低

标准技术分析定义：
    - 价格形成两个波段低点 Low2 < Low1
    - 对应DIF值 DIF2 > DIF1
    - 即：价格创新低 + DIF没有创新低 = MACD底背离

低点识别（无未来函数）：
    - 过去 lookback 窗口寻找局部低点
    - 只使用当前及历史数据（不依赖未来数据确认）
    - 最近低点必须已经确认（排除最后一个数据点）

有效性限制：
    - 第二低点至少低于第一低点 2%：Low2 < Low1 × 0.98
    - 两个低点之间间隔：5 <= distance <= 60 交易日
"""

import numpy as np


def find_valleys(low, lookback=20, min_gap=5):
    """
    寻找波段低点（无未来函数）

    算法：
        - 在位置 i，观察过去 lookback 天的最低点
        - 如果最低点在昨天（i-1），且今天已反弹（low[i] > low[i-1]）
        - 则昨天是一个已确认的波段低点
        - 遍历整个数据范围，找出所有此类波段低点
        - 两个波谷之间至少间隔 min_gap 个交易日

    未来函数检查：
        ❌ low[i] < low[i-10:i+10]  ← 对称窗口，用了未来数据
        ✅ low[i-1] == min(low[i-20:i]) and low[i] > low[i-1]
                                  ↑ 只向左看，只用过去数据
                                  波谷由今天的反弹确认

    参数：
        low: 最低价数组（一维 numpy 数组）
        lookback: 局部窗口大小（默认20）
        min_gap: 波谷最小间隔（默认5）

    返回：
        list[int]: 波谷位置的索引列表（按时间升序），每个位置都已确认

    确认机制：
        - 波谷在位置 i-1 被识别，由位置 i 的数据确认
        - 最后一个数据点无法被确认，自然排除
        - 持续下跌时不产生波谷（需要看到反弹信号）
    """
    n = len(low)
    valleys = []

    # 从第 lookback 个数据点开始（需要足够的历史数据）
    # 到 n-1 结束（最后一个数据点用于确认前一天的波谷）
    for i in range(lookback, n):
        # 过去 lookback 天的数据，不含今天
        # 窗口 = [i-lookback, i-1] 共 lookback 个数据点
        window = low[i - lookback : i]

        # 条件1：最低点在窗口末尾（昨天是最低点）
        # 条件2：今天已反弹（low[i] > low[i-1]）
        # 同时满足 → 昨天是已确认的波段低点
        if low[i - 1] == np.min(window) and low[i] > low[i - 1]:
            valleys.append(i - 1)

    # 按最小间隔 min_gap 过滤
    if not valleys:
        return []

    filtered = [valleys[0]]
    for v in valleys[1:]:
        if v - filtered[-1] >= min_gap:
            filtered.append(v)

    return filtered


def check_macd_divergence(df, lookback=20, min_gap=5, max_gap=60, price_drop=0.98):
    """
    检测单只股票是否存在MACD底背离

    使用DIF线判断底背离（标准技术分析定义）：
        - 在价格序列中寻找波段低点
        - 检查最近两个波段低点是否满足：
          1. 价格明显创新低（Low2 < Low1 × 0.98，至少低2%）
          2. DIF没有创新低（DIF2 > DIF1）
          3. 两个低点间隔在 [5, 60] 个交易日内

    参数：
        df: DataFrame
            - 必须包含 'low' 列（最低价）
            - 必须包含 'DIF' 列（DIF线值）
        lookback: 波谷识别窗口（默认20）
        min_gap: 波谷最小间隔（默认5）
        max_gap: 波谷最大间隔（默认60）
        price_drop: 价格新低阈值（默认0.98，即至少低2%）

    返回：
        tuple (has_divergence, detail_dict)
        - has_divergence: bool 是否检测到底背离
        - detail: dict 或 None（仅内部调试用，不输出到最终结果）

    可用于每日收盘后运行：
        - 所有数据截至当日收盘
        - 不需要未来数据确认
        - 最近低点已经在历史数据中确认
    """
    low = df["low"].values
    dif = df["DIF"].values

    # 在价格序列上寻找波段低点
    valleys = find_valleys(low, lookback=lookback, min_gap=min_gap)

    # 至少需要2个波谷才能判断背离
    if len(valleys) < 2:
        return False, None

    # 取最近的两个波谷（最新的底背离信号）
    v1 = valleys[-2]  # 第一个波谷（较早）
    v2 = valleys[-1]  # 第二个波谷（较近，已确认）

    # ── 条件1：距离检查 ──
    # 两个低点之间间隔必须在 [5, 60] 个交易日内
    dist = v2 - v1
    if dist < min_gap or dist > max_gap:
        return False, None

    # ── 条件2：价格检查 ──
    # 价格必须明显创新低（至少低2%）
    price1 = low[v1]
    price2 = low[v2]
    if not (price2 < price1 * price_drop):
        return False, None

    # ── 条件3：DIF检查 ──
    # DIF必须没有创新低（DIF值抬高）
    dif1 = dif[v1]
    dif2 = dif[v2]
    if not (dif2 > dif1):
        return False, None

    # ── 全部条件满足 → 底背离成立 ──
    detail = {
        "price_low1": round(float(price1), 2),
        "price_low2": round(float(price2), 2),
        "dif1": round(float(dif1), 4),
        "dif2": round(float(dif2), 4),
        "valley1_index": int(v1),
        "valley2_index": int(v2),
        "distance": int(dist),
    }
    # 如果有日期列，附加日期信息
    if "date" in df.columns:
        try:
            detail["valley1_date"] = str(df["date"].iloc[v1])[:10]
            detail["valley2_date"] = str(df["date"].iloc[v2])[:10]
        except Exception:
            pass

    return True, detail


def scan_divergence(df):
    """
    扫描单只股票是否满足MACD底背离条件

    统一接口，供 main.py 调用。
    输入输出与之前版本一致，无需修改其他模块。

    参数：
        df: 股票的日线数据
            - 必须包含列：date, open, high, low, close
            - 必须包含列：DIF, DEA, MACD（由 indicators.macd.calculate_macd 生成）

    返回：
        bool: True = 存在底背离

    数据流：
        main.py → scan_divergence(df) → check_macd_divergence(df) → find_valleys(low)
    """
    # 数据量检查（至少需要60根K线）
    if df is None or len(df) < 60:
        return False

    has_div, _ = check_macd_divergence(df)
    return has_div