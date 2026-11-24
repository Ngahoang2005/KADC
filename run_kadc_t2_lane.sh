#!/bin/bash
# One lane of the KADC T2 replay (no training): 4 datasets x no KD / KD x 3 seeds, claimed with lock directories.
cd "$(dirname "$0")"
lane=$1
once=$2  # "once": run one job now (smoke); otherwise wait for the T1 lane to finish
[ "$once" = once ] || while systemctl --user is-active --quiet sldc-t1-l1; do sleep 60; done
ulimit -n 65536
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
L=my_run_out/kadc_t2/locks; mkdir -p $L
jobs=()
for ds in cub200_224 cars196_224 imagenet-r cifar100_224; do for s in 1993 1996 1997; do
  jobs+=(k:$ds:0:0:$s kb:$ds:1.0:0.1:$s); done; done
for pass in 1 2; do
for job in "${jobs[@]}"; do
  IFS=: read tag ds gkd gnorm s <<< "$job"
  name=${tag}_$ds
  [ -f my_run_out/kadc_t2/$name/seed$s/task9.json ] && continue
  mkdir $L/${name}_s$s 2>/dev/null || continue
  ck=my_ckpt/kadc_$name; [ -f my_ckpt/$name/seed$s/task9.pt ] && ck=my_ckpt/$name
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 10000 ] && \
        [ "$(free -g | awk '/Mem:/{print $7}')" -ge 11 ]; do sleep 60; done
  echo "$(date '+%F %T') lane$lane start $name seed$s" >> my_run_out/kadc_t2/queue.log
  python main_sldc.py --dataset=$ds --convnet_type=vit-b-p16-lora \
    --use_auxiliary_data_enhancement "" --use_linear_compensation 1 --use_weak_nonlinear_compensation 1 \
    --gamma_kd=$gkd --gamma_norm=$gnorm --smart_defaults --seed $s --kadc --kadc_t2 --replay_ckpt $ck \
    --log_root my_logs/kadc_t2_$name --out_dir my_run_out/kadc_t2/$name > my_logs/kadc_t2_${name}_s$s.out 2>&1
  rc=$?
  echo "$(date '+%F %T') lane$lane end $name seed$s rc=$rc" >> my_run_out/kadc_t2/queue.log
  [ $rc -ne 0 ] && rmdir $L/${name}_s$s
  [ "$once" = once ] && exit $rc
done
done
