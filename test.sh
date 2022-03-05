#!/usr/bin/env bash
set -e          # 只要脚本发生错误就停止运行
set -u          # 如果遇到不存在的变量就报错并停止执行
set -x          # 运行指令结果的时候，输出对应的指令
set -o pipefail # 确保只要一个子命令失败，整个管道命令就失败

export CUDA_VISIBLE_DEVICES="$1"
echo 'Excute the script on GPU: ' "$1"

echo 'For COD'
python test.py --config ./configs/zoomnet/cod_zoomnet.py \
    --model-name ZoomNet \
    --batch-size 22 \
    --load-from ./output/ForSharing/cod_zoomnet_r50_bs8_e40_2022-03-04.pth \
    --save-path ./output/ForSharing/COD_Results

echo 'For SOD'
python test.py --config ./configs/zoomnet/sod_zoomnet.py \
    --model-name ZoomNet \
    --batch-size 22 \
    --load-from ./output/ForSharing/sod_zoomnet_r50_bs22_e50_2022-03-04_fixed.pth \
    --save-path ./output/ForSharing/SOD_Results
