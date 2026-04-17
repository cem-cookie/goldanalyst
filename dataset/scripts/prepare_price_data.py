#!/usr/bin/env python3
"""
从gold_history.csv加载价格数据的辅助脚本
"""

import csv
from datetime import datetime
from typing import Dict, Tuple


def load_gold_prices_from_csv(csv_file: str) -> Tuple[Dict[str, float], str]:
    """
    从CSV文件加载黄金历史价格

    假设CSV格式:
    date,price
    2025-08-26,3000.00
    2025-08-27,3404.60
    ...

    或其他常见格式，自动检测分隔符和日期列

    Returns:
        (prices_dict, format_description)
        prices_dict: {date_str: price_float}
        format_description: 数据格式说明
    """

    prices = {}

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            # 尝试自动检测分隔符
            sample = f.read(1024)
            f.seek(0)

            # 尝试检测分隔符
            delimiter = ','
            if '\t' in sample:
                delimiter = '\t'
            elif ';' in sample:
                delimiter = ';'

            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)

            if not header:
                print("❌ CSV文件为空")
                return {}, "Empty file"

            print(f"📋 检测到列: {header}")

            # 找到日期和价格列
            date_col = None
            price_col = None

            for i, col in enumerate(header):
                col_lower = col.lower().strip()
                if any(x in col_lower for x in ['date', '时间', '日期']):
                    date_col = i
                if any(x in col_lower for x in ['price', 'close', '价格', '收盘']):
                    price_col = i

            # 如果没找到，使用默认索引
            if date_col is None:
                date_col = 0
            if price_col is None:
                price_col = 1

            print(f"📌 使用第 {date_col} 列作为日期，第 {price_col} 列作为价格")

            # 读取数据
            for row_idx, row in enumerate(reader):
                if len(row) <= max(date_col, price_col):
                    continue

                try:
                    date_str = row[date_col].strip()
                    price_str = row[price_col].strip()

                    # 尝试解析价格
                    price = float(price_str)

                    # 标准化日期格式为 YYYY-MM-DD
                    date_formatted = _parse_date(date_str)

                    if date_formatted:
                        prices[date_formatted] = price
                except (ValueError, IndexError) as e:
                    if row_idx < 5:  # 只打印前几行错误
                        print(f"  ⚠️  第 {row_idx + 2} 行解析失败: {row}")

        print(f"\n✅ 成功加载 {len(prices)} 条价格记录")
        print(f"📅 日期范围: {min(prices.keys())} 到 {max(prices.keys())}")
        print(f"💹 价格范围: ${min(prices.values()):.2f} - ${max(prices.values()):.2f}")

        return prices, f"Loaded {len(prices)} records from CSV"

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return {}, str(e)


def _parse_date(date_str: str) -> str:
    """将各种日期格式转换为 YYYY-MM-DD"""

    # 已经是正确格式
    if isinstance(date_str, str) and len(date_str) == 10 and date_str[4] == '-':
        return date_str

    # 尝试多种格式
    formats = [
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%m-%d-%Y',
        '%m/%d/%Y',
        '%d-%m-%Y',
        '%d/%m/%Y',
        '%Y%m%d',
        '%m-%d-%y',
        '%Y.%m.%d',
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime('%Y-%m-%d')
        except:
            continue

    return None


def validate_prices_for_improver(prices_dict: Dict[str, float]) -> bool:
    """验证价格数据是否适合用于改进器"""

    if not prices_dict:
        print("❌ 价格字典为空")
        return False

    # 检查日期格式
    for date in list(prices_dict.keys())[:5]:
        if not isinstance(date, str) or len(date) != 10:
            print(f"❌ 日期格式错误: {date}")
            return False

    # 检查价格值
    for price in list(prices_dict.values())[:5]:
        if not isinstance(price, (int, float)) or price <= 0:
            print(f"❌ 价格值无效: {price}")
            return False

    print("✅ 价格数据验证通过")
    return True


def create_sample_jsonl_with_prices(prices_dict: Dict[str, float],
                                    output_file: str = "training_data_raw.jsonl"):
    """
    创建示例JSONL文件，包含价格数据
    用于测试改进器
    """

    import json

    sample_records = [
        {
            "date": "2025-08-26",
            "prices": {
                "current": prices_dict.get("2025-08-26", 3000),
                "historical": {}
            },
            "strategy": {
                "recommendation": "BUY",
                "confidence": 7,
                "reasoning": ["金价处于低位", "经济不确定性上升"]
            },
            "messages": [
                {
                    "role": "user",
                    "content": "请分析2025-08-26的黄金走势"
                },
                {
                    "role": "assistant",
                    "content": "建议买入，预期未来7天上升"
                }
            ]
        },
        {
            "date": "2025-08-27",
            "prices": {
                "current": prices_dict.get("2025-08-27", 3404),
                "historical": {}
            },
            "strategy": {
                "recommendation": "HOLD",
                "confidence": 5,
                "reasoning": ["走势不明确"]
            },
            "messages": [
                {
                    "role": "user",
                    "content": "2025-08-27的建议"
                }
            ]
        }
    ]

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in sample_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n📝 已创建示例JSONL: {output_file}")


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    import sys

    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/gold_history.csv"

    print(f"📂 正在加载 {csv_file}...")
    prices, description = load_gold_prices_from_csv(csv_file)

    if prices:
        validate_prices_for_improver(prices)

        # 创建示例JSONL
        create_sample_jsonl_with_prices(prices)

        print("\n✨ 现在你可以运行:")
        print("   python jsonl_improver.py")
        print("\n如果需要自己的JSONL文件，请确保每条记录包含:")
        print("  - 'date' 字段 (YYYY-MM-DD格式)")
        print("  - 'prices' 字段包含 'current' 价格")
        print("  - 'strategy' 字段包含推荐和置信度")