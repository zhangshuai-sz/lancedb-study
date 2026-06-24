#!/bin/bash
rm -f /tmp/bench_run.log /tmp/top_lance.log /tmp/pidstat_lance.log /tmp/mem_lance.log /tmp/iostat_lance.log

# 启动压测（FTS并发64，Vector/Hybrid并发32）
/data/workspace/lancedb-study/.venv/bin/python3 /data/workspace/lancedb-study/lancedb/bench.py --max-concurrency 32 --fts-concurrency 64 > /tmp/bench_run.log 2>&1 &
BENCH_PID=$!
echo "BENCH_PID=$BENCH_PID"
sleep 3

# 使用 BENCH_PID 的子进程来找到真正的 python3 进程
# 或者直接用 BENCH_PID（因为 python3 是直接启动的，BENCH_PID 就是 python3 进程）
REAL_PID=$BENCH_PID
# 验证进程是否存在
if ! kill -0 $REAL_PID 2>/dev/null; then
    echo "ERROR: 进程 $REAL_PID 不存在"
    cat /tmp/bench_run.log
    exit 1
fi
PROC_NAME=$(cat /proc/$REAL_PID/comm 2>/dev/null)
echo "REAL_PID=$REAL_PID (comm=$PROC_NAME)"

# top线程级监控
(while kill -0 $REAL_PID 2>/dev/null; do echo "=== $(date '+%H:%M:%S') ==="; top -b -n1 -H -p $REAL_PID 2>/dev/null | head -60; echo '---'; sleep 2; done) > /tmp/top_lance.log 2>&1 &

# pidstat进程级CPU（-t 显示线程级别）
pidstat -p $REAL_PID -u 2 > /tmp/pidstat_lance.log 2>&1 &

# 内存监控
(echo 'TIME RSS_MB VSZ_MB'; while kill -0 $REAL_PID 2>/dev/null; do RSS_KB=$(grep VmRSS /proc/$REAL_PID/status 2>/dev/null | awk '{print $2}'); VSZ_KB=$(grep VmSize /proc/$REAL_PID/status 2>/dev/null | awk '{print $2}'); echo "$(date '+%H:%M:%S') $((${RSS_KB:-0}/1024))MB $((${VSZ_KB:-0}/1024))MB"; sleep 2; done) > /tmp/mem_lance.log 2>&1 &

# 磁盘IO
iostat -xd 2 > /tmp/iostat_lance.log 2>&1 &

echo "Monitors started, waiting..."
wait $BENCH_PID
echo "Done exit=$?"
sleep 3
pkill -f "pidstat -p $REAL_PID" 2>/dev/null
pkill -f "iostat -xd" 2>/dev/null
echo "Stopped"
