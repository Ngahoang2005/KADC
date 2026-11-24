#!/bin/bash
# One lane of the SLDC + KADC queue (same jobs as run_kadc_full.sh). Lanes share the list and claim each job with
# a lock directory, so no two lanes run the same job. A failed job releases its lock and is retried once.
cd "$(dirname "$0")"
lane=$1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
L=my_run_out/kadc/locks; mkdir -p $L
jobs=()
for s in 1996 1997; do for ds in cub200_224 cars196_224; do jobs+=(k:$ds:0:0:$s kb:$ds:1.0:0.1:$s); done; done
for ds in imagenet-r cifar100_224; do for s in 1993 1996 1997; do jobs+=(k:$ds:0:0:$s kb:$ds:1.0:0.1:$s); done; done
for pass in 1 2; do
for job in "${jobs[@]}"; do
  IFS=: read tag ds gkd gnorm s <<< "$job"
  name=${tag}_$ds
  [ -f my_run_out/kadc/$name/seed$s/task9.json ] && continue
  mkdir $L/${name}_s$s 2>/dev/null || continue
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 10000 ] && \
        [ "$(free -g | awk '/Mem:/{print $7}')" -ge 11 ]; do sleep 60; done
  echo "$(date '+%F %T') lane$lane start $name seed$s" >> my_run_out/kadc/queue_full.log
  python main_sldc.py --dataset=$ds --convnet_type=vit-b-p16-lora \
    --use_auxiliary_data_enhancement "" --use_linear_compensation 1 --use_weak_nonlinear_compensation 1 \
    --gamma_kd=$gkd --gamma_norm=$gnorm --smart_defaults --seed $s --kadc \
    --save_dir my_ckpt/kadc_$name --no_save_stats \
    --log_root my_logs/kadc_$name --out_dir my_run_out/kadc/$name > my_logs/kadc_${name}_s$s.out 2>&1
  rc=$?
  echo "$(date '+%F %T') lane$lane end $name seed$s rc=$rc" >> my_run_out/kadc/queue_full.log
  [ $rc -ne 0 ] && rmdir $L/${name}_s$s
done
done
