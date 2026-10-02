"""
A股MACD底背离筛选系统 — 程序入口
===================================
运行方式：python main.py

流程：
    1. 获取A股全市场股票列表（过滤北交所/ST/退市）
    2. 并发拉取每只股票日线数据（本地缓存，增量更新）
    3. 计算MACD指标（EMA12 / EMA26 / DEA9）
    4. 检测MACD底背离（波段低点识别，lookback=20）
    5. 输出结果到控制台 + CSV

输出格式：
    股票代码 | 股票名称
    例如：000001 | 平安银行

当前阶段：
    只实现MACD底背离筛选，不加入其他条件。
    不评分、不生成买卖信号。
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import concurrent.futures
import json

# Windows 终端 GBK 编码兼容（避免 print 中文/emoji 报错）
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

# 将项目根目录加入 Python 路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.stock_data import (
    get_stock_list,
    get_stock_data,
    REQUIRED_BARS,
)
from indicators.macd import calculate_macd
from filters.macd_divergence import scan_divergence


# ── 目录配置 ──
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
PROGRESS_DIR = ROOT / "progress"
PROGRESS_DIR.mkdir(exist_ok=True)
PROGRESS_FILE = PROGRESS_DIR / "macd_progress.json"

# ── 超时与断点续跑配置 ──
TIMEOUT_PER_STOCK = 30           # 每只股票处理超时（秒）
MAX_CONSECUTIVE_FAILURES = 5      # 连续失败上限，达到后自动保存退出


def load_progress():
    """加载断点续跑进度"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"processed_codes": [], "failed_codes": [], "results": []}
    return {"processed_codes": [], "failed_codes": [], "results": []}


def save_checkpoint(processed_codes, failed_codes, results, output_path):
    """保存进度检查点（进度文件 + CSV结果）"""
    progress = {
        "processed_codes": list(processed_codes),
        "failed_codes": list(failed_codes),
        "results": results,
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        if len(results) > 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    except Exception:
        pass  # 保存失败不影响主流程


def process_stock(stock_code, stock_name):
    """
    处理单只股票的全流程

    步骤：
        1. 获取数据（带本地缓存）
        2. 计算MACD指标
        3. 检测MACD底背离

    参数：
        stock_code: 6位股票代码
        stock_name: 股票名称

    返回：
        dict 或 None
        - 若检测到底背离，返回 {"股票代码": code, "股票名称": name}
        - 否则返回 None
    """
    try:
        # ── 步骤1：获取数据 ──
        df = get_stock_data(stock_code)
        if df is None or len(df) < REQUIRED_BARS:
            return None  # 数据不足，跳过

        # ── 步骤2：计算MACD ──
        df = calculate_macd(df)

        # ── 步骤3：检测底背离 ──
        if scan_divergence(df):
            return {"股票代码": stock_code, "股票名称": stock_name}

        return None

    except Exception as e:
        # 任何异常（拉取失败、计算错误等）都跳过
        return None


def main():
    """主函数"""
    print("=" * 55)
    print("  A股MACD底背离筛选系统")
    print("  当前阶段：第一层 — MACD底背离筛选")
    print("=" * 55)
    print()

    # ── 获取股票列表 ──
    print("📥 获取股票列表...")
    stock_list = get_stock_list()
    total = len(stock_list)
    print(f"   共 {total} 只股票（已剔除北交所/ST/退市/停牌）")
    print()

    # ── 加载断点续跑进度 ──
    progress = load_progress()
    processed_codes = set(progress.get("processed_codes", []))
    failed_codes = set(progress.get("failed_codes", []))
    results = progress.get("results", [])

    # 过滤已处理的股票
    if len(processed_codes) > 0 or len(failed_codes) > 0:
        mask = ~stock_list["股票代码"].isin(processed_codes | failed_codes)
        pending = stock_list[mask].reset_index(drop=True)
        pending_total = len(pending)
        print(f"📂 发现上次进度：已处理 {len(processed_codes)} 只，超时跳过 {len(failed_codes)} 只")
        print(f"⏩ 跳过已处理的 {len(processed_codes) + len(failed_codes)}/{total} 只")
        print(f"   继续处理剩余 {pending_total} 只")
        print()
    else:
        pending = stock_list

    # ── 并发扫描 ──
    print("🔍 开始扫描MACD底背离...")
    print(f"   并发线程：6  |  每只最低数据量：{REQUIRED_BARS}根K线")
    print(f"   每只超时：{TIMEOUT_PER_STOCK}秒  |  连续失败上限：{MAX_CONSECUTIVE_FAILURES}")
    print()

    today_str = datetime.now().strftime("%Y%m%d")
    output_path = OUTPUT_DIR / f"macd_divergence_{today_str}.csv"
    consecutive_failures = 0
    processed_count = len(processed_codes) + len(failed_codes)

    # 使用多线程并发加速（最多6线程）
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        # 只提交未处理的股票
        future_to_stock = {
            executor.submit(process_stock, row["股票代码"], row["股票名称"]): row
            for _, row in pending.iterrows()
        }

        # 收集结果（带超时控制）
        for future in concurrent.futures.as_completed(future_to_stock):
            row = future_to_stock[future]
            code = row["股票代码"]
            name = row["股票名称"]
            processed_count += 1

            try:
                result = future.result(timeout=TIMEOUT_PER_STOCK)
                if result is not None:
                    results.append(result)
                processed_codes.add(code)
                consecutive_failures = 0
            except concurrent.futures.TimeoutError:
                failed_codes.add(code)
                consecutive_failures += 1
                print(f"  ⚠️ {code} {name} 超时({TIMEOUT_PER_STOCK}s) 跳过（连续失败: {consecutive_failures}）")
                sys.stdout.flush()

            # 每100只保存一次进度
            if processed_count % 100 == 0:
                pct = processed_count / total * 100
                print(f"   进度：{processed_count}/{total} ({pct:.1f}%)  |  已发现 {len(results)} 只")
                sys.stdout.flush()
                save_checkpoint(processed_codes, failed_codes, results, output_path)

            # 连续失败过多，自动退出
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n⚠️ 连续 {MAX_CONSECUTIVE_FAILURES} 只股票超时，网络可能已断开，自动保存退出")
                print(f"📁 已保存进度（{len(results)} 只），下次运行自动续跑")
                save_checkpoint(processed_codes, failed_codes, results, output_path)
                return

    # ── 正常完成：清理进度文件 ──
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    # ── 输出结果 ──
    print()
    print("=" * 55)
    print(f"  ✅ 扫描完成！")
    print(f"  📊 扫描总数：{total} 只")
    print(f"  🎯 MACD底背离：{len(results)} 只")
    print("=" * 55)
    print()

    if len(results) > 0:
        print("MACD底背离股票列表：")
        print("-" * 55)
        for i, row in enumerate(results, 1):
            code = row["股票代码"]
            name = row["股票名称"]
            print(f"  {code} | {name}")
        print("-" * 55)
        print()

        result_df = pd.DataFrame(results)
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"📁 结果已保存：{output_path}")
        print(f"   共 {len(results)} 只股票")

    else:
        print("未发现MACD底背离股票。")
        print("（可能原因：市场整体偏弱，MACD柱持续走低，尚未形成底背离结构）")

    print()
    print("=" * 55)


if __name__ == "__main__":
    # 记录开始时间
    start_time = datetime.now()
    main()
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"⏱ 运行耗时：{elapsed:.1f} 秒")