"""Last-Block Replay (LBR): split the ViT at z, the CLS state right after the
attention branch of the last block. Everything after z acts on a single token,
so f = h(z) can be recomputed from z alone."""
import torch
from torch import nn


def _vit(model):
    return model.convnet if hasattr(model, 'convnet') else model


def xprime(model, x):
    """z = (X + drop_path(ls1(attn(norm1(X)))))[:, 0], X = output of blocks 1..11."""
    vit = _vit(model)
    x = vit.patch_embed(x)
    cls_token = vit.cls_token.expand(x.shape[0], -1, -1)
    if vit.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
    else:
        x = torch.cat((cls_token, vit.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = vit.pos_drop(x + vit.pos_embed)
    for blk in vit.blocks[:-1]:
        x = blk(x)
    blk = vit.blocks[-1]
    ls1 = getattr(blk, 'ls1', nn.Identity())
    return (x + blk.drop_path(ls1(blk.attn(blk.norm1(x)))))[:, 0]


def head_from_xprime(model, z):
    """f = h(z): the MLP half of the last block, then the final LN and pre_logits."""
    vit = _vit(model)
    blk = vit.blocks[-1]
    ls2 = getattr(blk, 'ls2', nn.Identity())
    x = z[:, None, :]
    x = x + blk.drop_path(ls2(blk.mlp(blk.norm2(x))))
    return vit.pre_logits(vit.norm(x))[:, 0]
