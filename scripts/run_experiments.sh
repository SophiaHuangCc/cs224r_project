#!/bin/bash
# Sequential experiment runner - one task at a time to manage memory
# Logs to /tmp/cs224r_experiment2.log
#
# IMPORTANT: always run from the project root. Training code uses relative
# paths like save_dir="models", so launching from elsewhere (e.g. scripts/)
# creates a parallel models/ tree in the wrong place and silently splits
# outputs across two directories. The cd below enforces this.

cd ~/Desktop/cs224r/cs224r_project
PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/cs224r_project/bin/python

echo "========================================" 
echo "Experiment pipeline (sequential per-task)"
echo "Time: $(date)"
echo "========================================"

# Eureka training - task by task
for TASK in FetchReach-v4 FetchPickAndPlace-v4 FetchSlide-v4; do
    echo ""
    echo "[$(date)] Eureka: $TASK"
    $PYTHON scripts/train_local.py --reward-type eureka --env $TASK --skip-existing
    if [ $? -ne 0 ]; then
        echo "[$(date)] ERROR: Eureka $TASK failed!"
    fi
done

# Ensemble training - task by task
for TASK in FetchReach-v4 FetchPickAndPlace-v4 FetchSlide-v4; do
    echo ""
    echo "[$(date)] Ensemble: $TASK"
    $PYTHON scripts/train_local.py --reward-type ensemble --env $TASK --skip-existing
    if [ $? -ne 0 ]; then
        echo "[$(date)] ERROR: Ensemble $TASK failed!"
    fi
done

# KL training - task by task
for TASK in FetchReach-v4 FetchPickAndPlace-v4 FetchSlide-v4; do
    echo ""
    echo "[$(date)] KL: $TASK"
    $PYTHON scripts/train_local.py --reward-type kl --env $TASK --skip-existing
    if [ $? -ne 0 ]; then
        echo "[$(date)] ERROR: KL $TASK failed!"
    fi
done

# Evaluation
echo ""
echo "[$(date)] Evaluation..."
$PYTHON scripts/eval_local.py --skip-existing
if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: Evaluation failed!"
fi

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Time: $(date)"
echo "========================================"
