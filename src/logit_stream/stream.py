"""Decompose final logits into per-layer contributions.

Given the logits produced after every layer prefix (a logit-lens
style stream), LOGIT-STREAM reports the per-layer increments for a
target token, the birth layer of the final answer, and how the
positive increments are shared. Pure NumPy.
"""

import numpy as np


def _check(prefix_logits):
    if not isinstance(prefix_logits, (list, tuple)) or len(prefix_logits) == 0:
        raise ValueError("prefix_logits must be a non-empty list")
    arrs = [np.asarray(p, dtype=float) for p in prefix_logits]
    shape = arrs[0].shape
    if arrs[0].ndim != 1:
        raise ValueError("each entry must be a 1-D logit vector")
    for a in arrs:
        if a.shape != shape:
            raise ValueError("all prefix logits must share one vocabulary size")
    return arrs


def layer_contributions(prefix_logits, target):
    """Per-layer logit increments for one target token.

    increment[i] = logits_after_layer_i[target] -
                   logits_after_layer_(i-1)[target],
    with logits after layer -1 defined as zero.
    """
    arrs = _check(prefix_logits)
    target = int(target)
    if not (0 <= target < arrs[0].shape[0]):
        raise ValueError("target index out of range")
    prev = np.zeros_like(arrs[0])
    increments = []
    for a in arrs:
        increments.append(a - prev)
        prev = a
    final_logit = float(arrs[-1][target])
    return {
        "num_layers": len(arrs),
        "target": target,
        "final_logit": final_logit,
        "layer_increments": [float(c[target]) for c in increments],
        "sum_check": float(sum(c[target] for c in increments)),
    }


def birth_certificate(prefix_logits, target=None):
    """Find the layer where the final answer first becomes the top token.

    ``birth_layer`` is the smallest layer prefix whose argmax equals
    the final argmax. ``positive_share`` distributes the positive
    increments of the target across layers.
    """
    arrs = _check(prefix_logits)
    final = arrs[-1]
    answer = int(np.argmax(final))
    tgt = int(target) if target is not None else answer
    if not (0 <= tgt < arrs[0].shape[0]):
        raise ValueError("target index out of range")
    birth = None
    for i, a in enumerate(arrs):
        if int(np.argmax(a)) == answer:
            birth = i
            break
    contrib = layer_contributions(arrs, tgt)
    pos = [max(0.0, c) for c in contrib["layer_increments"]]
    total_pos = sum(pos)
    share = [p / total_pos if total_pos > 0 else 0.0 for p in pos]
    return {
        "answer": answer,
        "target": tgt,
        "final_logit": contrib["final_logit"],
        "layer_increments": contrib["layer_increments"],
        "positive_share": share,
        "birth_layer": birth,
    }