from __future__ import annotations

from typing import Any, Sequence

from tensors import (
    INDEX_BYTES,
    Tensor,
    TensorError,
    dtype_bytes,
    total_parameters,
)

# The four meanings of "removed", Lecture 05 slide 8, in staircase order.
REMOVALS = ("masked", "patterned", "stored sparse", "structurally absent")

# How a tensor is written to disk. Three forms, and `classify_removal` reads
# this alongside the shapes to decide which of the four it got.
STORAGE_FORMS = ("dense", "masked", "sparse")

# What each granularity produces if you implement it the obvious way. Fine
# grained pruning in a framework leaves a mask behind; channel pruning leaves a
# smaller dense tensor. `sweep_model` uses this unless told otherwise, and
# `sweep.py --storage` is how you tell it otherwise.
GRANULARITY_STORAGE = {"fine": "masked", "channel": "dense"}

GRANULARITIES = tuple(GRANULARITY_STORAGE)

# A layer with no output channels is not a smaller layer, it is a broken one.
MIN_CHANNELS = 1

# N:M, for `classify_removal`. 2:4 is the case Ampere implements and the only
# one Lecture 05 slide 18 draws.
NM_N, NM_M = 2, 4


def _drop_count(n: int, ratio: float) -> int:
    if not 0.0 <= ratio <= 1.0:
        raise TensorError(f"ratio {ratio} is not in [0, 1]")
    return max(0, min(int(n), int(int(n) * float(ratio) + 0.5)))


def _smallest_indices(values: Sequence[float], k: int) -> tuple[int, ...]:
    if k <= 0:
        return ()
    order = sorted(range(len(values)), key=lambda i: (abs(values[i]), i))
    return tuple(sorted(order[:k]))


def _group_scores(t: Tensor, p: float = 2.0) -> tuple[float, ...]:
    if p <= 0:
        raise TensorError(f"p must be positive, got {p}")
    scores = []
    for c in range(t.channels):
        lo, hi = _channel_slice(t, c)
        acc = 0.0
        for v in t.data[lo:hi]:
            acc += abs(v) ** p
        scores.append(round(acc ** (1.0 / p), 10))
    return tuple(scores)


def _channel_slice(t: Tensor, c: int) -> tuple[int, int]:
    if not 0 <= c < t.channels:
        raise TensorError(f"{t.name}: no channel {c} in {t.channels}")
    stride = t.channel_stride
    return c * stride, (c + 1) * stride


# ---------------------------------------------------------------------------
# 1. the fine-grained criterion
# ---------------------------------------------------------------------------

def magnitude_mask(t: Tensor, ratio: float) -> tuple[int, ...]:
    n = len(t.data)
    k = _drop_count(n, ratio)
    prun_pos = set(_smallest_indices(t.data, k))
    mask = []

    for i in range(n):
        if i in prun_pos:
            mask.append(0)
        else:
            mask.append(1)

    return tuple(mask)


# ---------------------------------------------------------------------------
# 2. the structured criterion
# ---------------------------------------------------------------------------

def channel_keep(t: Tensor, ratio: float, p: float = 2.0) -> tuple[int, ...]:
    norm = _group_scores(t, p)
    drop = min(_drop_count(t.channels, ratio), t.channels - MIN_CHANNELS)
    drop_set = set(_smallest_indices(norm, drop))
    keep = []

    for c in range(t.channels):
        if c not in drop_set:
            keep.append(c)

    return tuple(keep)


# ---------------------------------------------------------------------------
# 3. masking — the removal that changes no shape
# ---------------------------------------------------------------------------

def apply_mask(t: Tensor, mask: Sequence[int]) -> Tensor:
    if len(mask) != len(t.data):
        raise TensorError()

    data = []
    for v, m in zip(t.data, mask):
        if m == 1:
            data.append(v)
        else:
            data.append(0.0)

    return Tensor(t.name, t.shape, tuple(data), t.dtype)


# ---------------------------------------------------------------------------
# 4. channel removal — the removal that changes the shape
# ---------------------------------------------------------------------------

def drop_channels(t: Tensor, keep: Sequence[int]) -> Tensor:
    if not keep:
        raise TensorError()
    if len(set(keep)) != len(keep):
        raise TensorError()
    for c in keep:
        if not 0 <= c < t.channels:
            raise TensorError()

    data = []
    for c in sorted(keep):
        lo = c * t.channel_stride
        hi = (c + 1) * t.channel_stride
        data.extend(t.data[lo:hi])

    shape_new = (len(keep),) + t.shape[1:]
    return Tensor(t.name, shape_new, tuple(data), t.dtype)


# ---------------------------------------------------------------------------
# 5. the file
# ---------------------------------------------------------------------------

def bytes_stored(tensors: Sequence[Tensor], storage: str = "dense",
                 mask_encoding: str = "framework") -> int:
    if storage not in ("dense", "masked", "sparse"):
        raise TensorError()
    if mask_encoding not in ("framework", "bitmap"):
        raise TensorError()

    total = 0
    for t in tensors:
        n = t.parameters
        db = dtype_bytes(t.dtype)

        if storage == "dense":
            total += n * db
        elif storage == "masked":
            total += n * db
            if mask_encoding == "framework":
                total += n * db
            else:
                total += (n + 7) // 8
        else:
            nnz = 0
            for v in t.data:
                if v != 0.0:
                    nnz += 1
            total += nnz * db + nnz * INDEX_BYTES

    return total


# ---------------------------------------------------------------------------
# 6. one row of the accounting table
# ---------------------------------------------------------------------------

def sparsity_row(model: str, ratio: float, granularity: str,
                 before: Sequence[Tensor], after: Sequence[Tensor],
                 storage: str = "dense",
                 mask_encoding: str = "framework") -> dict[str, Any]:
    if granularity not in ("fine", "channel"):
        raise TensorError()

    p_before = total_parameters(before)
    p_after = total_parameters(after)
    nonzero = 0

    for t in after:
        for v in t.data:
            if v != 0.0:
                nonzero += 1

    zeroed = p_after - nonzero
    zeroed_fraction = zeroed / p_before
    achieved_reduction = 1.0 - p_after / p_before

    # 121965
    out = { 
        "model": model,
        "granularity": granularity,
        "storage": storage,
        "mask_encoding": None,
        "nominal_ratio": ratio,
        "values_zeroed": zeroed,
        "zeroed_fraction": round(zeroed_fraction, 6),
        "parameters_before": p_before,
        "parameters_after": p_after,
        "achieved_reduction": round(achieved_reduction, 6),
        "bytes_dense": bytes_stored(before, "dense"),
        "bytes_stored": bytes_stored(after, storage, mask_encoding),
        "removal": classify_removal(before, after, storage),
    }

    if storage == "masked":
        out["mask_encoding"] = mask_encoding

    return out


# ---------------------------------------------------------------------------
# 7. which of the four you got
# ---------------------------------------------------------------------------

def classify_removal(before: Sequence[Tensor], after: Sequence[Tensor],
                     storage: str = "dense") -> str:
    for old, new in zip(before, after):
        if old.shape != new.shape:
            return "structurally absent"

    if storage == "sparse":
        return "stored sparse"

    zeroed = 0
    for t in after:
        for v in t.data:
            if v == 0.0:
                zeroed += 1

    if zeroed == 0:
        return "dense"

    patterned = True
    for t in after:
        if len(t.data) % NM_M != 0:
            patterned = False
            break

        for i in range(0, len(t.data), NM_M):
            nonzero = 0
            for v in t.data[i:i + NM_M]:
                if v != 0.0:
                    nonzero += 1
            if nonzero > NM_N:
                return "masked"

    if patterned:
        return "patterned"

    if zeroed > 0:
        return "masked"

    return "dense"


# ---------------------------------------------------------------------------
# 8. the whole table
# ---------------------------------------------------------------------------

def sweep_model(model: dict[str, Any], ratios: Sequence[float],
                granularities: Sequence[str] = GRANULARITIES,
                storage: str | None = None,
                mask_encoding: str = "framework",
                p: float = 2.0) -> list[dict[str, Any]]:
    before = model["tensors"]
    rows = []

    for g in granularities:
        form = storage or GRANULARITY_STORAGE[g]

        for r in sorted(ratios):
            after = []
            for t in before:
                if g == "fine":
                    mask = magnitude_mask(t, r)
                    after.append(apply_mask(t, mask))
                elif g == "channel":
                    keep = channel_keep(t, r, p)
                    after.append(drop_channels(t, keep))

            rows.append(sparsity_row(model["name"], r, g, before, after, form, mask_encoding))

    return rows
