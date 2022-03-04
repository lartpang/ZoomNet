#!/usr/bin/env bash
set -e          # 只要脚本发生错误就停止运行
set -u          # 如果遇到不存在的变量就报错并停止执行
set -x          # 运行指令结果的时候，输出对应的指令
set -o pipefail # 确保只要一个子命令失败，整个管道命令就失败

export CUDA_VISIBLE_DEVICES="$1"
echo 'Excute the script on GPU: ' "$1"

python test.py --model-name ZoomNetV1 --batch-size 4 \
  --load-from ./output/ZoomNet_BS8_LR0.05_E40_H384_W384_OPMsgd_OPGMfinetune_SCf3_AMP_INFOdemo/pth/state_final.pth
