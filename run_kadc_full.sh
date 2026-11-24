#!/bin/bash
# SLDC (original branches) + KADC (--kadc) trained from scratch, Sup-21K, sequential.
# Stage 1: CUB / Cars seeds 1996, 1997 (seed 1993 = replay runs). Stage 2: ImageNet-R, CIFAR-100, 3 seeds.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
jobs=()
for s in 1996 1997; do for ds in cub200_224 cars196_224; do jobs+=(k:$ds:0:0:$s kb:$ds:1.0:0.1:$s); done; done
for ds in imagenet-r cifar100_224; do for s in 1993 1996 1997; do jobs+=(k:$ds:0:0:$s kb:$ds:1.0:0.1:$s); done; done
for job in "${jobs[@]}"; do
  IFS=: read tag ds gkd gnorm s <<< "$job"
  name=${tag}_$ds
  [ -f my_run_out/kadc/$name/seed$s/task9.json ] && continue
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 12000 ] && \
        [ "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)" -le 30 ]; do sleep 60; done
  echo "$(date '+%F %T') start $name seed$s" >> my_run_out/kadc/queue_full.log
  python main_sldc.py --dataset=$ds --convnet_type=vit-b-p16-lora \
    --use_auxiliary_data_enhancement "" --use_linear_compensation 1 --use_weak_nonlinear_compensation 1 \
    --gamma_kd=$gkd --gamma_norm=$gnorm --smart_defaults --seed $s --kadc \
    --save_dir my_ckpt/kadc_$name --no_save_stats \
    --log_root my_logs/kadc_$name --out_dir my_run_out/kadc/$name > my_logs/kadc_${name}_s$s.out 2>&1
  rc=$?
  echo "$(date '+%F %T') end $name seed$s rc=$rc" >> my_run_out/kadc/queue_full.log
done
