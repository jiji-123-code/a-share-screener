"""
数据获取模块
===========
负责：
1. 获取A股全市场股票列表（过滤北交所/ST/退市）
2. 获取单只股票日线数据（AKShare + 本地缓存）
3. 增量更新（避免重复请求）

数据源：新浪 stock_zh_a_daily（之前已验证：支持日期范围、含成交量）
"""

import os
import time
import socket
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import akshare as ak

# ── 全局设置 ──────────────────────────────────────────────────────────
# 全局socket超时（防止新浪源对个别股票请求挂起卡死，20秒无响应自动抛异常）
socket.setdefaulttimeout(20)

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── 常量 ──────────────────────────────────────────────────────────────
REQUIRED_BARS = 120       # 需要的K线数量（MACD(26)+DEA(9)+lookback(20)+余量）
MIN_TRADING_DAYS = 60     # 上市不足60日剔除
BJ_PREFIXES = ("43", "83", "87", "92")  # 北交所代码前缀


def get_stock_list():
    """
    获取A股全市场股票列表

    返回：
        DataFrame，列：股票代码, 股票名称
        自动过滤：北交所、ST/*ST、退市

    数据源：ak.stock_info_a_code_name()（5549只，含代码+名称）
    """
    try:
        df = ak.stock_info_a_code_name()
    except Exception as e:
        # akshare 内部拉取北交所数据可能失败（网络问题），重试一次
        print(f"   ⚠️ 首次拉取股票列表失败（{type(e).__name__}），重试中...")
        try:
            df = ak.stock_info_a_code_name()
        except Exception:
            print("   ❌ 拉取股票列表失败，请检查网络后重试")
            return pd.DataFrame(columns=["股票代码", "股票名称"])

    # 适配新版 akshare 列名（code/name）
    if "股票代码" not in df.columns and "code" in df.columns:
        df = df.rename(columns={"code": "股票代码", "name": "股票名称"})

    # 过滤北交所（43/83/87/92开头）
    df = df[~df["股票代码"].str.startswith(BJ_PREFIXES)]

    # 过滤ST/*ST/退市
    df = df[~df["股票名称"].str.contains("ST|退市|退", na=False)]

    # 重置索引
    df = df.reset_index(drop=True)

    return df


def get_daily_data(stock_code, start_date=None, end_date=None, max_retries=3, retry_delay=2):
    """
    获取单只股票日线数据

    数据源：新浪 stock_zh_a_daily
    - 支持日期范围（start_date/end_date）
    - 含成交量（volume）
    - 前复权

    参数：
        stock_code: 6位股票代码（如 "000001"）
        start_date: 开始日期（YYYYMMDD），None=不限
        end_date: 结束日期（YYYYMMDD），None=最新
        max_retries: 最大重试次数（默认3）
        retry_delay: 重试间隔（秒，默认2）

    返回：
        DataFrame（date, open, high, low, close, volume），按日期升序排列
        失败返回 None
    """
    # 构造新浪格式代码
    if stock_code.startswith("6"):
        symbol = f"sh{stock_code}"
    elif stock_code.startswith(("0", "3")):
        symbol = f"sz{stock_code}"
    else:
        # 其他代码（如北交所），尝试直接使用
        symbol = stock_code

    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",  # 前复权
            )

            if df is not None and len(df) > 0:
                # 标准化列名
                df = df.rename(columns={
                    "date": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                return df
            else:
                return None

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            # 最后一次重试也失败，返回 None
            return None

    return None


# ── 最新交易日缓存（进程级，避免每只股票都请求一次指数接口）──
_LATEST_DATE_CACHE = None
_LATEST_DATE_LOCK = None
try:
    import threading
    _LATEST_DATE_LOCK = threading.Lock()
except Exception:
    pass


def get_latest_trading_date():
    """
    获取最新交易日（进程级缓存，只请求一次）

    通过上证指数新浪源判断最新交易日。
    兜底返回今天。

    注意：
        本函数在扫描 5000+ 只股票时会被调用 5000+ 次。
        若每次都发网络请求，会造成两个严重后果：
          1. 极慢（每次调用都要等一个 HTTP 往返）
          2. 高频冲击新浪接口，容易被限流甚至触发异常
        因此改为进程级缓存，整个扫描过程只请求一次。

    返回：
        str, 格式 "YYYYMMDD"
    """
    global _LATEST_DATE_CACHE

    # 已缓存，直接返回
    if _LATEST_DATE_CACHE is not None:
        return _LATEST_DATE_CACHE

    # 加锁，避免多线程首次调用时同时发起请求
    if _LATEST_DATE_LOCK is not None:
        with _LATEST_DATE_LOCK:
            if _LATEST_DATE_CACHE is not None:
                return _LATEST_DATE_CACHE
            _LATEST_DATE_CACHE = _fetch_latest_trading_date()
            return _LATEST_DATE_CACHE

    _LATEST_DATE_CACHE = _fetch_latest_trading_date()
    return _LATEST_DATE_CACHE


def _fetch_latest_trading_date():
    """实际请求最新交易日（仅由 get_latest_trading_date 内部调用一次）"""
    try:
        idx = ak.stock_zh_index_daily(symbol="sh000001")
        latest = pd.to_datetime(idx["date"].iloc[-1])
        return latest.strftime("%Y%m%d")
    except Exception:
        # 兜底：取今天
        return datetime.now().strftime("%Y%m%d")


def get_cache_path(stock_code):
    """获取本地缓存文件路径"""
    return DATA_DIR / f"{stock_code}.csv"


def load_from_cache(stock_code):
    """
    从本地缓存加载数据

    返回：
        DataFrame 或 None（缓存不存在时）
    """
    cache_path = get_cache_path(stock_code)
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None


def save_to_cache(stock_code, df):
    """保存数据到本地缓存（覆盖写入）"""
    df.to_csv(get_cache_path(stock_code), index=False, encoding="utf-8")


def get_stock_data(stock_code, force_refresh=False):
    """
    获取股票日线数据（带本地缓存和增量更新）

    策略：
    1. 优先从本地缓存读取
    2. 如果缓存未包含最新交易日，增量拉取缺失数据
    3. 如果缓存数据不足 REQUIRED_BARS，完整拉取

    参数：
        stock_code: 6位股票代码
        force_refresh: 是否强制重新拉取（默认False）

    返回：
        DataFrame（含至少 REQUIRED_BARS 根K线）或 None
    """
    latest_date = get_latest_trading_date()

    # ── 尝试从缓存读取 ──
    if not force_refresh:
        cached = load_from_cache(stock_code)
        if cached is not None and len(cached) > 0:
            # 检查缓存是否已包含最新交易日
            last_cached_date = cached["date"].max().strftime("%Y%m%d")

            if last_cached_date >= latest_date and len(cached) >= REQUIRED_BARS:
                # 缓存已是最新且数据充足
                return cached

            if last_cached_date < latest_date:
                # 增量更新：拉取缺失日期
                # 从缓存最后一天往前推5天，避免边界遗漏
                start = (pd.to_datetime(last_cached_date) - timedelta(days=5)).strftime("%Y%m%d")
                new_data = get_daily_data(stock_code, start_date=start, end_date=latest_date)

                if new_data is not None and len(new_data) > 0:
                    # 合并数据（去重）
                    combined = pd.concat([cached, new_data], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["date"])
                    combined = combined.sort_values("date").reset_index(drop=True)
                    save_to_cache(stock_code, combined)
                    return combined

            # 缓存数据不足但无法增量更新，返回缓存数据
            return cached

    # ── 完整拉取 ──
    df = get_daily_data(stock_code, end_date=latest_date)
    if df is not None and len(df) > 0:
        save_to_cache(stock_code, df)
    return df