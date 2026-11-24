#!/bin/bash
# KADC E1/E2 (post-hoc): replay of the Sup-21K seed-1993 K runs (no KD = alpha, KD = beta) with --kadc.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for cfg in k:cub200_224:0:0 k:cars196_224:0:0 kb:cub200_224:1.0:0.1 kb:cars196_224:1.0:0.1; do
  IFS=: read tag ds gkd gnorm <<< "$cfg"
  name=${tag}_$ds
  [ -f my_run_out/kadc/$name/seed1993/task9.json ] && continue
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 12000 ] && \
        [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)" -le 30 ]; do sleep 60; done
  echo "$(date '+%F %T') start $name" >> my_run_out/kadc/queue.log
  python main_sldc.py --dataset=$ds --convnet_type=vit-b-p16-lora \
    --use_auxiliary_data_enhancement "" --use_linear_compensation 1 --use_weak_nonlinear_compensation 1 \
    --gamma_kd=$gkd --gamma_norm=$gnorm --smart_defaults --seed 1993 --kadc --replay_ckpt my_ckpt/$name \
    --log_root my_logs/kadc_$name --out_dir my_run_out/kadc/$name > my_logs/kadc_$name.out 2>&1
  rc=$?
  echo "$(date '+%F %T') end $name rc=$rc" >> my_run_out/kadc/queue.log
done
