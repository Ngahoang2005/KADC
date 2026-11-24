"""Build the official CUB-200-2011 train/test split (5994/5794) as symlinks.

Images come from the PILOT copy (80/20 split across train/ and test/); only the
split membership is taken from CUB_200_2011/{images.txt,train_test_split.txt}.
"""
import os
import sys

meta_dir, pilot_root, out_root = sys.argv[1:4]

images = dict(l.split() for l in open(os.path.join(meta_dir, 'images.txt')))
is_train = dict(l.split() for l in open(os.path.join(meta_dir, 'train_test_split.txt')))

n = {'train': 0, 'val': 0}
for img_id, rel in images.items():
    src = None
    for part in ('train', 'test'):
        p = os.path.join(pilot_root, part, rel)
        if os.path.exists(p):
            src = p
            break
    if src is None:
        sys.exit(f'missing image: {rel}')
    split = 'train' if is_train[img_id] == '1' else 'val'
    dst = os.path.join(out_root, split, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)
    n[split] += 1

print(n)
