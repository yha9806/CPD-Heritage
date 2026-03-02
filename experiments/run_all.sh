#!/bin/bash
# Run each model×resolution as separate processes to prevent GPU memory accumulation
# Each process exits → GPU memory fully freed

CD="/mnt/i/JOCCH ICH Survey/jocch_submission/experiments"
LOG="$CD/train.log"

MODELS="efficientnet_b0 dinov2_vits14 convnextv2_tiny resnet50 vit_base_16 swinv2_tiny"
RESOLUTIONS="224 512 1024"
TASKS="3class fine"

echo "=== CPD Batch Runner (per-run isolation) ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

for model in $MODELS; do
    for res in $RESOLUTIONS; do
        for task in $TASKS; do
            RUN_ID="${model}_${res}_${task}"
            RESULT="$CD/results/${RUN_ID}.json"

            # Skip completed
            if [ -f "$RESULT" ]; then
                STATUS=$(python3 -c "import json; d=json.load(open('$RESULT')); print(d.get('status','ok'))" 2>/dev/null)
                if [ "$STATUS" != "OOM_FAILED" ]; then
                    continue
                fi
            fi

            echo "" | tee -a "$LOG"
            echo ">>> $RUN_ID ($(date))" | tee -a "$LOG"

            # GPU memory cleanup before each run
            python3 -c "import torch; torch.cuda.empty_cache(); print(f'  GPU free: {torch.cuda.mem_get_info()[0]//1048576}MB')" 2>/dev/null | tee -a "$LOG"

            # Run as isolated process
            cd "$CD" && PYTHONUNBUFFERED=1 python3 02_train.py --model "$model" --res "$res" --task "$task" >> "$LOG" 2>&1
            EXIT=$?

            if [ $EXIT -ne 0 ]; then
                echo ">>> CRASH: $RUN_ID exit=$EXIT ($(date))" | tee -a "$LOG"
                # Test GPU
                python3 -c "import torch; torch.cuda.init(); print('GPU alive')" 2>/dev/null
                if [ $? -ne 0 ]; then
                    echo ">>> GPU DEAD. Restart WSL and rerun." | tee -a "$LOG"
                    exit 1
                fi
                sleep 5
            fi

            # Force GPU cleanup after each run
            python3 -c "import torch,gc; gc.collect(); torch.cuda.empty_cache()" 2>/dev/null
            sleep 2
        done
    done
done

echo "" | tee -a "$LOG"
echo "=== ALL DONE ($(date)) ===" | tee -a "$LOG"

# Summary
echo "" | tee -a "$LOG"
echo "=== SUMMARY ===" | tee -a "$LOG"
for f in "$CD/results/"*.json; do
    python3 -c "
import json,sys
with open(sys.argv[1]) as f: d=json.load(f)
s=d.get('status','')
if s=='OOM_FAILED': print(f\"  {d['run_id']:<43} OOM\")
else: print(f\"  {d['run_id']:<43} acc={d['test_accuracy']:.3f}  F1={d['macro_f1']:.3f}  {d['train_time_sec']:.0f}s\")
" "$f"
done | sort | tee -a "$LOG"
