"""
周线动力系统颜色筛选模块
=========================
功能：
    读取前置策略输出的股票CSV文件，
    对每只股票进行周线动力系统分析，
    筛选出动力系统颜色为红色或蓝色的股票。

动力系统颜色定义（基于MACD指标）：
    E13 = EMA(Close, 13)
    DIF = EMA(Close, 12) - EMA(Close, 26)
    DEA = EMA(DIF, 9)
    HIST = DIF - DEA

    - 红色：E13 > 前一周E13 且 HIST > 前一周HIST（多头动能增强）
    - 蓝色：(E13 > 前一周E13 且 HIST < 前一周HIST)
           或 (E13 < 前一周E13 且 HIST > 前一周HIST)（分歧/调整阶段）
    - 绿色：E13 < 前一周E13 且 HIST < 前一周HIST（空头动能增强，过滤）

输入：
    前置策略CSV（至少含"股票代码"列）

输出：
    DataFrame（股票代码, 股票名称, 颜色）
"""

import time
import socket
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import akshare as ak
import concurrent.futures
import json

# ── 全局设置 ──────────────────────────────────────────────────────────
# 防止新浪源个别股票请求挂起卡死
socket.setdefaulttimeout(20)

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
PROGRESS_DIR = ROOT / "progress"
PROGRESS_DIR.mkdir(exist_ok=True)

# ── 常量 ──────────────────────────────────────────────────────────────
MIN_WEEKS = 30          # 至少需要30周数据（MACD的26周EMA+9周DEA，留充足余量）
REQUIRED_DAYS = 400     # 拉取约400个交易日，足够重采样为约80周
TIMEOUT_PER_STOCK = 30  # 每只股票数据获取超时（秒）
MAX_CONSECUTIVE_FAILURES = 5  # 连续失败上限，达到后自动保存退出


def get_progress_file_path(csv_path):
    """根据输入CSV文件名生成进度文件路径"""
    csv_name = Path(csv_path).stem
    return PROGRESS_DIR / f"power_color_{csv_name}.json"


def load_progress(progress_file):
    """加载断点续跑进度"""
    if progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"processed_codes": [], "failed_codes": [], "results": []}
    return {"processed_codes": [], "failed_codes": [], "results": []}


def save_checkpoint(progress_file, processed_codes, failed_codes, results, output_path):
    """保存进度检查点"""
    progress = {
        "processed_codes": list(processed_codes),
        "failed_codes": list(failed_codes),
        "results": results,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        if len(results) > 0:
            result_df = pd.DataFrame(results)
            # 红色优先排序
            color_order = {"红色": 0, "蓝色": 1}
            result_df["排序键"] = result_df["颜色"].map(color_order)
            result_df = result_df.sort_values("排序键").drop(columns="排序键").reset_index(drop=True)
            result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    except Exception:
        pass


def fetch_with_timeout(stock_code, timeout=30):
    """带超时的日线数据获取"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
        future = exe.submit(get_daily_data, stock_code)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None


def get_daily_data(stock_code, max_retries=3, retry_delay=2):
    """
    获取股票日线数据（用于重采样为周线）

    数据源：新浪 stock_zh_a_daily
    - 前复权
    - 含成交量
    - 只拉取最近约150个交易日，避免全量拉取

    参数：
        stock_code: 6位股票代码（如 "000001"）
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）

    返回：
        DataFrame（date, close, volume），按日期升序排列
        失败返回 None
    """
    # 构造新浪格式代码
    if stock_code.startswith("6"):
        symbol = f"sh{stock_code}"
    elif stock_code.startswith(("0", "3")):
        symbol = f"sz{stock_code}"
    else:
        # 其他代码，直接使用
        symbol = stock_code

    # 只拉取最近 REQUIRED_DAYS 个交易日的数据
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=int(REQUIRED_DAYS * 1.5))).strftime("%Y%m%d")

    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )

            if df is not None and len(df) > 0:
                df = df[["date", "close", "volume"]].copy()
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                return df

            return None

        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return None

    return None


def resample_to_weekly(df):
    """
    将日线数据重采样为周线

    规则：
        - 以周五为周的最后一天（W-FRI）
        - 收盘价：取该周最后一个交易日收盘价
        - 成交量：取该周成交量总和

    参数：
        df: 日线 DataFrame（含 date, close, volume）

    返回：
        DataFrame（date, close, volume），按日期升序
    """
    df = df.set_index("date").copy()
    weekly = df.resample("W-FRI").agg({
        "close": "last",
        "volume": "sum",
    })
    # 去除NaN行（不完整的周）
    weekly = weekly.dropna().reset_index()
    return weekly


def calculate_macd_weekly(weekly_df):
    """
    计算周线MACD相关指标

    公式：
        E13 = EMA(Close, 13)
        DIF = EMA(Close, 12) - EMA(Close, 26)
        DEA = EMA(DIF, 9)
        HIST = DIF - DEA

    参数：
        weekly_df: 周线 DataFrame（含 close）

    返回：
        DataFrame（新增 e13, dif, dea, hist 列）
    """
    df = weekly_df.copy()

    # 计算 E13
    df["e13"] = df["close"].ewm(span=13, adjust=False).mean()

    # 计算 MACD
    df["dif"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["hist"] = df["dif"] - df["dea"]

    return df


def get_color(e13_current, e13_prev, hist_current, hist_prev):
    """
    判断周线动力系统颜色（基于MACD指标）

    - 红色：E13上升 且 HIST上升（多头动能增强）
    - 蓝色：(E13上升 且 HIST下降) 或 (E13下降 且 HIST上升)（分歧/调整）
    - 绿色：E13下降 且 HIST下降（空头动能增强，本模块不输出）

    参数：
        e13_current: 当前周 E13 值
        e13_prev: 前一周 E13 值
        hist_current: 当前周 HIST 值
        hist_prev: 前一周 HIST 值

    返回：
        str: "红色" / "蓝色" / None
    """
    e13_up = e13_current > e13_prev
    hist_up = hist_current > hist_prev

    if e13_up and hist_up:
        return "红色"
    elif (e13_up and not hist_up) or (not e13_up and hist_up):
        return "蓝色"
    # 绿色（e13_down and hist_down）→ 不输出
    return None


def filter_power_color(csv_path):
    """
    主函数：读取前置策略CSV → 周线动力系统颜色筛选

    （支持断点续跑：中断后下次运行自动跳过已处理的股票）

    参数：
        csv_path: 前置策略输出的CSV文件路径

    返回：
        DataFrame（股票代码, 股票名称, 颜色），红色在前、蓝色在后
    """
    print(f"📂 读取前置策略结果: {csv_path}")

    # ── 读取输入CSV ──
    candidates = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 适配列名
    if "股票代码" not in candidates.columns:
        if "code" in candidates.columns:
            candidates = candidates.rename(columns={"code": "股票代码"})
        else:
            candidates = candidates.rename(columns={candidates.columns[0]: "股票代码"})
    if "股票名称" not in candidates.columns:
        if "name" in candidates.columns:
            candidates = candidates.rename(columns={"name": "股票名称"})
        else:
            candidates["股票名称"] = ""

    total = len(candidates)
    print(f"📊 待筛选股票: {total} 只")
    print(f"{'='*55}")

    # ── 加载断点续跑进度 ──
    progress_file = get_progress_file_path(csv_path)
    progress = load_progress(progress_file)
    processed_codes = set(progress.get("processed_codes", []))
    failed_codes = set(progress.get("failed_codes", []))
    results = progress.get("results", [])

    # 过滤已处理的股票
    if len(processed_codes) > 0 or len(failed_codes) > 0:
        mask = ~candidates["股票代码"].astype(str).str.strip().str.zfill(6).isin(processed_codes | failed_codes)
        pending = candidates[mask].reset_index(drop=True)
        pending_total = len(pending)
        print(f"📂 发现上次进度：已处理 {len(processed_codes)} 只，超时跳过 {len(failed_codes)} 只")
        print(f"⏩ 跳过已处理的 {len(processed_codes) + len(failed_codes)}/{total} 只")
        print(f"   继续处理剩余 {pending_total} 只")
        print()
    else:
        pending = candidates

    # ── 输出文件路径 ──
    today = datetime.now().strftime("%Y%m%d")
    output_path = OUTPUT_DIR / f"power_color_{today}.csv"

    consecutive_failures = 0
    processed_count = len(processed_codes) + len(failed_codes)

    for idx, row in pending.iterrows():
        code = str(row["股票代码"]).strip().zfill(6)
        name = str(row.get("股票名称", "")).strip()
        processed_count += 1

        # 进度显示（每10只或首只显示一次）
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  📈 进度: {processed_count}/{total} ({(processed_count)/total*100:.1f}%)  |  当前: {code} {name}")

        # ── 步骤1：获取日线数据（带超时） ──
        daily = fetch_with_timeout(code, timeout=TIMEOUT_PER_STOCK)
        if daily is None or len(daily) < REQUIRED_DAYS * 0.5:
            failed_codes.add(code)
            consecutive_failures += 1
            print(f"  ⚠️ {code} {name} 超时/失败 跳过（连续失败: {consecutive_failures}）")
            continue

        consecutive_failures = 0

        # ── 步骤2：重采样为周线 ──
        weekly = resample_to_weekly(daily)
        if len(weekly) < MIN_WEEKS:
            processed_codes.add(code)
            continue

        # ── 步骤3：计算周线MACD ──
        weekly = calculate_macd_weekly(weekly)

        # 获取最新两周数据
        latest = weekly.iloc[-1]
        prev = weekly.iloc[-2]

        # ── 步骤4：判断颜色 ──
        color = get_color(latest["e13"], prev["e13"], latest["hist"], prev["hist"])
        if color:
            results.append({
                "股票代码": code,
                "股票名称": name,
                "颜色": color,
            })
        processed_codes.add(code)

        # 每50只保存一次进度
        if processed_count % 50 == 0:
            save_checkpoint(progress_file, processed_codes, failed_codes, results, output_path)

        # 连续失败过多，自动退出
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"\n⚠️ 连续 {MAX_CONSECUTIVE_FAILURES} 只股票超时/失败，网络可能已断开，自动保存退出")
            print(f"📁 已保存进度（{len(results)} 只），下次运行自动续跑")
            save_checkpoint(progress_file, processed_codes, failed_codes, results, output_path)
            return pd.DataFrame(results)

    # ── 正常完成：清理进度文件 ──
    if progress_file.exists():
        progress_file.unlink()

    # ── 输出结果 ──
    print(f"\n{'='*55}")
    print(f"✅ 完成！共扫描 {total} 只，命中 {len(results)} 只")

    result_df = pd.DataFrame(results)

    if len(result_df) > 0:
        color_order = {"红色": 0, "蓝色": 1}
        result_df["排序键"] = result_df["颜色"].map(color_order)
        result_df = result_df.sort_values("排序键").drop(columns="排序键").reset_index(drop=True)

        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"📁 结果已保存: {output_path}")

        print(f"\n🎯 符合条件的股票（红色=多头增强，蓝色=空头减弱）:")
        print(f"{'─'*45}")
        for _, r in result_df.iterrows():
            print(f"  {r['股票代码']} | {r['股票名称']} | {r['颜色']}")

    return result_df