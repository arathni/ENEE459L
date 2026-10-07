from __future__ import annotations

from typing import Any

from graph import (
    Graph,
    Layer,
    computed,
    dtype_bytes,
    is_answered,
    unknown,
)


FLOPS_PER_MAC = 2

# The conventions `to_flops` will honour by name. Anything else is unknown
# rather than an assumption, because the whole point of the parameter is that
# the caller has to say which one they mean.
FLOP_CONVENTIONS = {
    "mac_is_two_flops": 2,
    "mac_is_one_flop": 1,
}

# Batch normalisation holds two learnable vectors per channel (scale and shift)
# and two non-learnable ones (running mean and variance). The first pair are
# parameters; the second pair are buffers. Both are in the file.
BN_PARAMS_PER_CHANNEL = 2
BN_BUFFERS_PER_CHANNEL = 2

# Buffers are kept in FP32 even when the weights are not. Halving them saves
# nothing worth having and a denormal running variance is a real failure mode.
BUFFER_DTYPE = "fp32"

# Below this many models there is no line to fit and no residual to report.
MIN_MODELS_FOR_FIT = 3

# Two floats are the same MAC count when they are the same integer. There is no
# tolerance here on purpose: MAC counts are integers, and a tolerance would let
# two genuinely different architectures be reported as tied.
TIE_EXACT = True


# ===========================================================================
# 1. How many numbers are stored
# ===========================================================================

def _layer_parameters(ly: Layer) -> int:
    if ly.kind == "conv":
        C_out = ly.out_shape[0]
        C_in = ly.in_shape[0]
        if ly.kernel is None:
            k_h, k_w = (1, 1)
        else:
            k_h, k_w = ly.kernel

        n_weights = C_out * (C_in // ly.groups) * k_h * k_w
        if ly.bias:
            n_weights += C_out
        return n_weights

    elif ly.kind == "linear":
        f_out = ly.out_shape[0]
        f_in = ly.in_shape[0]
        weights = f_out * f_in
        if ly.bias:
            weights += f_out
        return weights

    elif ly.kind == "bn":
        return BN_PARAMS_PER_CHANNEL * ly.out_shape[0]

    return 0


def count_parameters(graph: Graph) -> dict[str, Any]:
    per_layer = {}
    total = 0

    for ly in graph:
        parameters = _layer_parameters(ly)
        per_layer[ly.name] = parameters
        total += parameters

    return computed(
        total,
        f"{graph.name}: {len(graph)} layers, shapes from the description",
        per_layer=per_layer,
        includes_bias=True,
        excludes_bn_buffers=True,
        bn_params_per_channel=BN_PARAMS_PER_CHANNEL,
    )


# ===========================================================================
# 2. What those numbers weigh, which is not the size of the file
# ===========================================================================


def model_size_bytes(graph: Graph) -> dict[str, Any]:
    per_dtype = {} # datatype to bytes
    per_layer = {} # layer name to bytes
    buffer_bytes = 0.0

    for ly in graph:
        n = _layer_parameters(ly)
        layer_bytes = n * dtype_bytes(ly.weight_dtype)

        if ly.weight_dtype not in per_dtype:
            per_dtype[ly.weight_dtype] = 0.0
        per_dtype[ly.weight_dtype] += layer_bytes

        if ly.kind == "bn":
            buffers = BN_BUFFERS_PER_CHANNEL * ly.out_shape[0]
            bn_bytes = buffers * dtype_bytes(BUFFER_DTYPE)
            buffer_bytes += bn_bytes

            if BUFFER_DTYPE not in per_dtype:
                per_dtype[BUFFER_DTYPE] = 0.0
            per_dtype[BUFFER_DTYPE] += bn_bytes
            layer_bytes += bn_bytes

        per_layer[ly.name] = layer_bytes

    total = sum(per_layer.values())
    return computed(
        total,
        f"{graph.name}: per-layer dtypes, buffers at {BUFFER_DTYPE}",
        per_layer=per_layer,
        per_dtype=per_dtype,
        buffer_bytes=buffer_bytes,
        container_overhead_excluded=True,
        note="not the size of the file on disk; see the handout, Stage A step 3",
    )

# ===========================================================================
# 3. The memory nobody puts in the table
# ===========================================================================

def _elements(shape: tuple[int, ...]) -> int:
    total = 1
    for dimension in shape:
        total *= dimension
    return total


def _last_use(graph: Graph) -> dict[str, int]:
    last = {}
    names = []
    for ly in graph:
        names.append(ly.name)

    for i in range(len(graph)):
        ly = graph.layers[i]
        if ly.reads:
            for t in ly.reads:
                last[t] = i
        elif i == 0:
            last["__input__"] = 0
        else:
            last[names[i - 1]] = i

        last.setdefault(ly.name, i)

    last[graph.layers[-1].name] = len(graph) - 1
    return last


def _peak_elements(graph: Graph, last_use: dict[str, int]) -> int:
    live = {"__input__": _elements(graph.input_shape)}
    peak = 0

    for i in range(len(graph)):
        ly = graph.layers[i]
        live[ly.name] = ly.out_elements
        peak = max(peak, sum(live.values()))

        for name in last_use:
            if last_use[name] == i:
                del live[name]

    return peak


def count_activations(graph: Graph) -> dict[str, Any]:
    last_use = _last_use(graph)
    live = {"__input__": _elements(graph.input_shape) * dtype_bytes(graph.precision)}
    peak_bytes = 0.0
    peak_at = None
    total_elements = 0
    total_bytes = 0.0

    for i in range(len(graph)):
        ly = graph.layers[i]
        out_b = ly.out_elements * dtype_bytes(ly.act_dtype)
        live[ly.name] = out_b
        total_elements += ly.out_elements
        total_bytes += out_b

        resident = sum(live.values())
        if resident > peak_bytes:
            peak_bytes = resident
            peak_at = ly.name

        for name in last_use:
            if last_use[name] == i:
                del live[name]

    return computed(
        peak_bytes,
        f"{graph.name}: liveness over {len(graph)} layers, input included",
        peak_at=peak_at,
        peak_elements=_peak_elements(graph, last_use),
        total_elements=total_elements,
        total_bytes=total_bytes,
        includes_network_input=True,
        note="peak is the resident set, not the largest single tensor",
    )

# ===========================================================================
# 4. The factor of two that halves everybody's numbers
# ===========================================================================

def to_flops(macs: dict[str, Any], convention: str = "mac_is_two_flops") -> dict[str, Any]:
    if not is_answered(macs):
        return unknown("MAC count", "no valid MAC count provided")

    if convention not in FLOP_CONVENTIONS:
        return unknown(macs["source"], f"unknown convention {convention!r}; known: {sorted(FLOP_CONVENTIONS)}")

    factor = FLOP_CONVENTIONS[convention]
    total = macs["value"] * factor
    per_layer = {}

    if "per_layer" in macs:
        for name in macs["per_layer"]:
            per_layer[name] = macs["per_layer"][name] * factor

    return computed(
        total,
        macs["source"],
        convention=convention,
        flops_per_mac=factor,
        per_layer=per_layer,
        note="a count of operations contains no unit of time",
    )
