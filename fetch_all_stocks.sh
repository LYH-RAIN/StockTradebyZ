#!/bin/bash
# 抓取全量股票数据脚本
# 使用方法：./fetch_all_stocks.sh

# 设置TUSHARE_TOKEN（如果还没设置）
# export TUSHARE_TOKEN="你的token"

# 检查token是否设置
if [ -z "$TUSHARE_TOKEN" ]; then
    echo "错误: 请先设置环境变量 TUSHARE_TOKEN"
    echo "使用方法: export TUSHARE_TOKEN=你的token"
    echo "或者在当前shell中运行: TUSHARE_TOKEN=你的token ./fetch_all_stocks.sh"
    exit 1
fi

echo "=========================================="
echo "开始抓取全量股票数据"
echo "=========================================="
echo "股票池: stocklist.csv"
echo "数据目录: ./data"
echo "日期范围: 2024-01-01 至 今天"
echo "并发数: 6"
echo "=========================================="

# 抓取全量数据（从2024年1月1日至今）
python3 fetch_kline.py \
    --start 20240101 \
    --end today \
    --stocklist stocklist.csv \
    --out ./data \
    --workers 6

echo ""
echo "=========================================="
echo "数据抓取完成！"
echo "=========================================="
echo "数据保存在: ./data/"
echo "文件数量: $(ls -1 data/*.csv 2>/dev/null | wc -l)"

