"""
周线动力系统颜色筛选 — 程序入口
=================================

作用：
    读取前置策略输出的股票CSV文件，
    对每只股票进行周线动力系统颜色分析，
    筛选出红色或蓝色的股票。

用法：
    python power_filter_main.py [输入CSV路径]

默认输入：
    自动查找 ../MACD底背离筛选系统/output/ 下最新的 macd_divergence_*.csv

示例：
    # 自动使用MACD底背离最新结果（推荐）
    python power_filter_main.py

    # 指定其他策略结果
    python power_filter_main.py ../其他策略/output/result.csv

输出：
    output/power_color_YYYYMMDD.csv            
    控制台打印筛选结果（股票代码 | 股票名称 | 颜色）
"""

import sys
from pathlib import Path
from datetime import datetime
from filters.power_color_filter import filter_power_color


def find_latest_macd_output():
    """
    自动查找 MACD 底背离系统的最新输出文件

    返回：
        str, 文件路径；找不到则返回 None
    """
    macd_output_dir = Path(__file__).resolve().parent / "MACD底背离筛选系统" / "output"
    if not macd_output_dir.exists():
        return None

    # 查找所有 macd_divergence_*.csv 文件
    csv_files = sorted(macd_output_dir.glob("macd_divergence_*.csv"), reverse=True)
    if csv_files:
        return str(csv_files[0])
    return None


def main():
    # 解析命令行参数
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # 默认自动查找 MACD 最新输出
        csv_path = find_latest_macd_output()
        if csv_path is None:
            print("❌ 未找到 MACD 底背离输出文件")
            print("   用法: python power_filter_main.py [输入CSV路径]")
            print("   示例: python power_filter_main.py candidate_pool.csv")
            sys.exit(1)

    # 检查文件是否存在
    if not Path(csv_path).exists():
        print(f"❌ 文件不存在: {csv_path}")
        print(f"   用法: python power_filter_main.py [输入CSV路径]")
        print(f"   示例: python power_filter_main.py candidate_pool.csv")
        sys.exit(1)

    # ── 标题 ──
    print(f"{'='*55}")
    print(f"  A股股票筛选系统 — 周线动力系统颜色筛选")
    print(f"  输入: {csv_path}")
    print(f"{'='*55}\n")

    # ── 执行筛选 ──
    result = filter_power_color(csv_path)

    # ── 统计汇总 ──
    if result is not None and len(result) > 0:
        red_count = len(result[result["颜色"] == "红色"])
        blue_count = len(result[result["颜色"] == "蓝色"])
        print(f"\n📊 统计汇总: 红色 {red_count} 只 | 蓝色 {blue_count} 只 | 合计 {len(result)} 只")
    else:
        print(f"\n📭 没有符合条件的股票")

    print(f"\n{'='*55}")


if __name__ == "__main__":
    main()