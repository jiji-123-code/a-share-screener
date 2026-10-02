"""
MACD指标计算模块
================
使用标准MACD算法：
- EMA(12)：快线周期
- EMA(26)：慢线周期
- DEA(9)：信号线周期

计算公式：
    DIF = EMA(Close, 12) - EMA(Close, 26)
    DEA = EMA(DIF, 9)
    MACD柱 = 2 × (DIF - DEA)

计算方式与同花顺、东方财富保持一致（ewm adjust=False）。
"""

import pandas as pd
import numpy as np


def ema(series, period):
    """
    计算指数移动平均（EMA）

    使用 pandas ewm 的 adjust=False 模式：
    - 与同花顺、东方财富的EMA计算一致
    - 平滑因子 alpha = 2 / (period + 1)

    参数：
        series: 价格序列（pd.Series）
        period: 周期（如12、26、9）

    返回：
        pd.Series，EMA值
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    计算MACD指标

    参数：
        df: DataFrame，必须包含 'close' 列（收盘价）
        fast: 快线周期（默认12）
        slow: 慢线周期（默认26）
        signal: 信号线周期（默认9）

    返回：
        DataFrame，新增列：
        - DIF: 快线 - 慢线（DIF线）
        - DEA: DIF的信号线（DEA线）
        - MACD: 柱状线 = 2 × (DIF - DEA)

    说明：
        - MACD柱 > 0 表示 DIF 在 DEA 上方（多头动能）
        - MACD柱 < 0 表示 DIF 在 DEA 下方（空头动能）
        - 底背离时：价格新低，但 MACD柱低点抬高
    """
    # 确保收盘价为数值类型
    close = df["close"].astype(float)

    # 计算EMA
    ema_fast = ema(close, fast)   # EMA(12)
    ema_slow = ema(close, slow)   # EMA(26)

    # DIF = 快线 - 慢线
    dif = ema_fast - ema_slow

    # DEA = DIF 的9日EMA
    dea = ema(dif, signal)

    # MACD柱 = 2 × (DIF - DEA)
    macd = 2.0 * (dif - dea)

    # 写入结果
    result = df.copy()
    result["DIF"] = dif
    result["DEA"] = dea
    result["MACD"] = macd

    return result