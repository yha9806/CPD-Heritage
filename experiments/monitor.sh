#!/bin/bash
RESULTS_DIR="/mnt/i/JOCCH ICH Survey/jocch_submission/experiments/results"
LOG="/mnt/i/JOCCH ICH Survey/jocch_submission/experiments/train.log"
while true; do
    COUNT=$(ls "$RESULTS_DIR"/*.json 2>/dev/null | wc -l)
    RUNNING=$(ps aux | grep "02_train" | grep python3 | grep -v grep | wc -l)
    if [ "$RUNNING" -eq 0 ]; then
        echo "=== TRAINING FINISHED ==="
        echo "Completed runs: $COUNT/36"
        echo ""
        echo "=== ALL RESULTS ==="
        for f in "$RESULTS_DIR"/*.json; do
            python3 -c "
import json,sys
with open(sys.argv[1]) as f: d=json.load(f)
s=d.get('status','')
if s=='OOM_FAILED':
    print(f\"  {d['run_id']:<43} OOM_FAILED\")
else:
    print(f\"  {d['run_id']:<43} acc={d['test_accuracy']:.3f}  F1={d['macro_f1']:.3f}  {d['train_time_sec']:.0f}s\")
" "$f"
        done
        echo ""
        echo "=== LAST 30 LINES OF LOG ==="
        tail -30 "$LOG"
        exit 0
    fi
    sleep 300
done
