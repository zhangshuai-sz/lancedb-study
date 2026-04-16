#!/bin/bash
# ES 导入进度监控脚本
# 用法: nohup bash es_monitor.sh > es_monitor.log 2>&1 &

INTERVAL=300  # 5分钟 = 300秒
INDEX="wikipedia-1"
TOTAL=62398110
PREV_COUNT=0
PREV_TIME=0

echo "=========================================="
echo "ES 导入进度监控已启动"
echo "检查间隔: ${INTERVAL}秒 (5分钟)"
echo "=========================================="

while true; do
    echo ""
    echo "--- $(date '+%Y-%m-%d %H:%M:%S') ---"

    # 检查导入进程是否还在运行（只匹配实际的 python3 进程，排除 uv run 包装进程）
    PROC=$(ps aux | grep -E "python3.*index\.py" | grep -v grep | grep -v "uv run" | head -1)
    if [ -z "$PROC" ]; then
        echo "[!] 导入进程已结束"
        DOC_COUNT=$(curl -s "localhost:9200/${INDEX}/_stats/docs" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['indices']['${INDEX}']['primaries']['docs']['count'])" 2>/dev/null)
        echo "最终文档数: ${DOC_COUNT} / ${TOTAL}"
        echo "监控结束"
        break
    fi

    # 获取进程信息
    PID=$(echo "$PROC" | awk '{print $2}')
    CPU=$(echo "$PROC" | awk '{print $3}')
    MEM_MB=$(echo "$PROC" | awk '{printf "%d", $6/1024}')

    # 获取文档数
    DOC_COUNT=$(curl -s "localhost:9200/${INDEX}/_stats/docs" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['indices']['${INDEX}']['primaries']['docs']['count'])" 2>/dev/null)

    # 获取存储大小
    STORE_GB=$(curl -s "localhost:9200/${INDEX}/_stats/store" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); s=d['indices']['${INDEX}']['primaries']['store']['size_in_bytes']; print(f'{s/1024/1024/1024:.2f}')" 2>/dev/null)

    # 计算进度和速度
    NOW=$(date +%s)
    if [ -n "$DOC_COUNT" ] && [ "$DOC_COUNT" -gt 0 ] 2>/dev/null; then
        PCT=$(python3 -c "print(f'${DOC_COUNT}/${TOTAL}*100:.1f}' if False else f'{${DOC_COUNT}/${TOTAL}*100:.1f}')" 2>/dev/null || echo "N/A")
        PCT=$(python3 -c "print(f'{${DOC_COUNT}/${TOTAL}*100:.1f}')" 2>/dev/null)
        if [ "$PREV_COUNT" -gt 0 ] && [ "$PREV_TIME" -gt 0 ]; then
            DELTA_DOCS=$((DOC_COUNT - PREV_COUNT))
            DELTA_TIME=$((NOW - PREV_TIME))
            if [ "$DELTA_TIME" -gt 0 ] && [ "$DELTA_DOCS" -ge 0 ]; then
                SPEED=$(python3 -c "print(f'{${DELTA_DOCS}/${DELTA_TIME}:.0f}')" 2>/dev/null)
                REMAINING=$((TOTAL - DOC_COUNT))
                if [ -n "$SPEED" ] && [ "$SPEED" != "0" ] && [ "$SPEED" -gt 0 ] 2>/dev/null; then
                    ETA_HOURS=$(python3 -c "print(f'{${REMAINING}/${SPEED}/3600:.1f}')" 2>/dev/null)
                    echo "进程: PID=${PID}, CPU=${CPU}%, MEM=${MEM_MB}MB"
                    echo "文档数: ${DOC_COUNT} / ${TOTAL} (${PCT}%)"
                    echo "存储大小: ${STORE_GB} GB"
                    echo "速度: ${SPEED} docs/s, 新增: ${DELTA_DOCS} 条"
                    echo "剩余: ${REMAINING} 条, 预计还需 ${ETA_HOURS} 小时"
                else
                    echo "进程: PID=${PID}, CPU=${CPU}%, MEM=${MEM_MB}MB"
                    echo "文档数: ${DOC_COUNT} / ${TOTAL} (${PCT}%)"
                    echo "存储大小: ${STORE_GB} GB"
                    echo "速度: 暂无新增"
                fi
            fi
        else
            echo "进程: PID=${PID}, CPU=${CPU}%, MEM=${MEM_MB}MB"
            echo "文档数: ${DOC_COUNT} / ${TOTAL} (${PCT}%)"
            echo "存储大小: ${STORE_GB} GB"
            echo "速度: 首次采样，下次计算"
        fi
        PREV_COUNT=$DOC_COUNT
        PREV_TIME=$NOW
    else
        echo "进程: PID=${PID}, CPU=${CPU}%, MEM=${MEM_MB}MB"
        echo "文档数: 获取失败"
    fi

    sleep $INTERVAL
done
