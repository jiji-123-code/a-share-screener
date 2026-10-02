"""
价值区间选股策略
《以交易为生》中的价值区间定义：
  13日EMA 和 26日EMA 之间的区域 = 价值区间

选股条件：
  当前收盘价 <= 价值区间上沿（即股价在价值区间内 或 在价值区间之下）

数据源：
  复用 MACD 底背离筛选系统的本地缓存（5017 只股票日线数据）
  无需重新拉取网络数据

输入：
  前置策略输出的 CSV 文件（如周线动力系统结果 power_color_*.csv）
  默认自动查找周线动力系统最新输出，也支持命令行参数指定路径
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 配置
# ============================================================

# MACD 系统的缓存目录（复用已有日线数据）
MACD_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "MACD底背离筛选系统",
    "data",
)

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# 周线动力系统输出目录（默认输入来源）
POWER_COLOR_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "output",
)

# 并发线程数
MAX_WORKERS = 6

# MACD 参数
EMA_SHORT = 13
EMA_LONG = 26


# ============================================================
# 获取股票列表
# ============================================================

def find_latest_power_color():
    """从周线动力系统输出目录查找最新的 power_color_*.csv 文件。"""
    if not os.path.exists(POWER_COLOR_DIR):
        return None
    files = glob.glob(os.path.join(POWER_COLOR_DIR, "power_color_*.csv"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def get_stock_list(csv_path=None):
    """
    从 CSV 文件获取股票列表。

    参数：
      csv_path — 指定 CSV 路径；为 None 时自动查找周线动力系统最新输出

    返回：
      list of (股票代码, 股票名称, 颜色)
    """
    # 1. 优先使用指定的 CSV 路径
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        code_col = "股票代码" if "股票代码" in df.columns else "code"
        name_col = "股票名称" if "股票名称" in df.columns else "name"
        color_col = "颜色" if "颜色" in df.columns else None
        stocks = []
        for _, row in df.iterrows():
            color = row.get(color_col, "") if color_col else ""
            stocks.append((str(row[code_col]).zfill(6), row.get(name_col, ""), color))
        print(f"  📄 从指定文件读取：{os.path.basename(csv_path)}，{len(stocks)} 只股票")
        return stocks

    # 2. 自动查找周线动力系统最新输出
    latest = find_latest_power_color()
    if latest:
        df = pd.read_csv(latest, encoding="utf-8-sig")
        code_col = "股票代码" if "股票代码" in df.columns else "code"
        name_col = "股票名称" if "股票名称" in df.columns else "name"
        color_col = "颜色" if "颜色" in df.columns else None
        stocks = []
        for _, row in df.iterrows():
            color = row.get(color_col, "") if color_col else ""
            stocks.append((str(row[code_col]).zfill(6), row.get(name_col, ""), color))
        print(f"  📄 自动读取周线动力结果：{os.path.basename(latest)}，{len(stocks)} 只股票")
        return stocks

    # 3. 回退：从缓存目录文件名推断
    if not os.path.exists(MACD_DATA_DIR):
        print(f"  ❌ 缓存目录不存在：{MACD_DATA_DIR}")
        return []

    cache_files = [f for f in os.listdir(MACD_DATA_DIR) if f.endswith(".csv")]
    codes = sorted([f.replace(".csv", "") for f in cache_files])
    stocks = [(code, "", "") for code in codes]
    print(f"  📂 从缓存目录读取：{len(stocks)} 只股票")
    return stocks


# ============================================================
# 读取缓存数据
# ============================================================

def load_cache_data(stock_code):
    """
    从 MACD 系统的缓存目录读取单只股票日线数据。
    """
    cache_path = os.path.join(MACD_DATA_DIR, f"{stock_code}.csv")
    if not os.path.exists(cache_path):
        return None

    try:
        df = pd.read_csv(cache_path)
        # 标准化列名
        df.columns = df.columns.str.lower()
        required = ["date", "close"]
        for col in required:
            if col not in df.columns:
                return None

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return None


# ============================================================
# 价值区间判断
# ============================================================

def check_value_zone(df):
    """
    判断股票是否在价值区间内或之下。

    价值区间定义：
      上沿 = max(13EMA, 26EMA)
      下沿 = min(13EMA, 26EMA)

    返回：
      (bool, str)  — (是否入选, 位置说明)
        位置说明: "价值区间内" / "价值区间下"
    """
    if df is None or len(df) < EMA_LONG + 1:
        return False, ""

    close = df["close"].values

    # 计算 EMA13 和 EMA26（与同花顺/东方财富一致）
    # 使用 ewm(adjust=False) 得到标准 EMA
    df_ema = df.copy()
    df_ema["ema13"] = df_ema["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df_ema["ema26"] = df_ema["close"].ewm(span=EMA_LONG, adjust=False).mean()

    last = df_ema.iloc[-1]
    current_close = last["close"]
    ema13 = last["ema13"]
    ema26 = last["ema26"]

    # 价值区间上沿 = 两条均线中较高的那条
    upper = max(ema13, ema26)
    lower = min(ema13, ema26)

    # 选股条件：收盘价 <= 价值区间上沿
    if current_close > upper:
        return False, ""

    # 判断位置
    if current_close >= lower:
        return True, "价值区间内"
    else:
        return True, "价值区间下"


# ============================================================
# 处理单只股票
# ============================================================

def process_stock(code_name_color):
    """
    处理单只股票：读取缓存 → 计算 EMA → 判断价值区间。
    """
    code, name, color = code_name_color
    try:
        df = load_cache_data(code)
        if df is None:
            return None

        passed, position = check_value_zone(df)
        if passed:
            return {"股票代码": code, "股票名称": name, "周线动力颜色": color, "位置": position}
        return None
    except Exception:
        return None


# ============================================================
# 主函数
# ============================================================

def main():
    # 解析命令行参数
    csv_path = None
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    print("=" * 55)
    print("  价值区间选股策略")
    print("  《以交易为生》EMA13 / EMA26 价值区间")
    print("=" * 55)
    print()
    print(f"  用法：python value_zone_scanner.py [输入CSV路径]")
    print(f"  不传参数时，自动读取周线动力系统最新输出")
    print()

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 获取股票列表
    print("📥 获取股票列表...")
    stocks = get_stock_list(csv_path)
    if not stocks:
        print("  ❌ 未获取到股票列表")
        print("  请先运行周线动力系统颜色筛选，或指定 CSV 路径：")
        print("  python value_zone_scanner.py ../output/power_color_20260905.csv")
        return

    total = len(stocks)
    print(f"  共 {total} 只股票")
    print()

    # 2. 并发处理
    print("🔍 开始扫描价值区间...")
    print(f"  并发线程：{MAX_WORKERS}")
    print(f"  价值区间上沿 = max(EMA13, EMA26)")
    print()

    results = []
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_stock, s): s for s in stocks}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                results.append(result)

            # 每 500 只打印进度并追加写入
            if i % 500 == 0 or i == total:
                print(f"  进度：{i}/{total} ({i/total*100:.1f}%)  |  已发现 {len(results)} 只")

                # 追加写入中间结果
                try:
                    temp_df = pd.DataFrame(results)
                    today_str = datetime.now().strftime("%Y%m%d")
                    temp_path = os.path.join(OUTPUT_DIR, f"value_zone_{today_str}.csv")
                    temp_df.to_csv(temp_path, index=False, encoding="utf-8-sig")
                except PermissionError:
                    print(f"  ⚠️ 无法写入 CSV（文件被其他程序占用），数据保留在内存中")

    # 3. 最终结果
    print()
    print(f"{'=' * 55}")
    print(f"  ✅ 扫描完成！")
    print(f"  共发现 {len(results)} 只股票在价值区间内或之下")
    print(f"{'=' * 55}")

    if results:
        # 按位置分组统计
        in_zone = sum(1 for r in results if r["位置"] == "价值区间内")
        below_zone = sum(1 for r in results if r["位置"] == "价值区间下")
        print(f"    价值区间内：{in_zone} 只")
        print(f"    价值区间下：{below_zone} 只")
        print()

        # 排序：蓝色优先 > 红色在后，区间内优先 > 区间下在后
        color_order = {"蓝色": 0, "红色": 1}
        zone_order = {"价值区间内": 0, "价值区间下": 1}
        results.sort(key=lambda x: (color_order.get(x["周线动力颜色"], 9), zone_order.get(x["位置"], 9)))
        result_df = pd.DataFrame(results)

        # 保存最终结果
        today_str = datetime.now().strftime("%Y%m%d")
        output_path = os.path.join(OUTPUT_DIR, f"value_zone_{today_str}.csv")
        try:
            result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"  📄 结果已保存：{output_path}")
        except PermissionError:
            print(f"  ⚠️ 无法写入 CSV（文件被其他程序占用），请关闭打开的 CSV 文件后重试")
            print(f"\n  📋 全部结果（控制台输出）：")
            for r in results:
                print(f"    {r['股票代码']} | {r['股票名称']} | {r['周线动力颜色']} | {r['位置']}")
        print()

        # 打印前 20 只
        print("  🏆 前 20 只（按位置排序）：")
        for r in results[:20]:
            print(f"    {r['股票代码']} | {r['股票名称']} | {r['周线动力颜色']} | {r['位置']}")
        if len(results) > 20:
            print(f"    ... 共 {len(results)} 只，完整列表见 CSV 文件")
    else:
        print("  未发现符合条件的股票。")

    print()


if __name__ == "__main__":
    # Windows 终端 UTF-8 支持
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()