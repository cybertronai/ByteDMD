"""Generic measurement helpers for heuristic-grid experiments."""

from __future__ import annotations

import gc
from typing import Any

from experiments.memory_management.tracer import (
    Context,
    Tracked,
    trace_to_cost_continuous,
    trace_to_cost_discrete,
    usqrt,
)


def wrap_value(ctx: Context, value: Any):
    """Recursively wrap nested Python structures in tracked scalars."""

    if isinstance(value, Tracked):
        return value
    if isinstance(value, list):
        return [wrap_value(ctx, item) for item in value]
    if isinstance(value, tuple):
        return tuple(wrap_value(ctx, item) for item in value)
    return Tracked(ctx, ctx.allocate(), value)


def measure_function(func, args: tuple[Any, ...], *, strategy: str) -> dict[str, object]:
    """Run one function under one memory-management strategy."""

    ctx = Context(strategy=strategy)
    wrapped_args = tuple(wrap_value(ctx, arg) for arg in args)
    peak_stack = [len(ctx.stack)]

    original_allocate = ctx.allocate

    def tracking_allocate():
        key = original_allocate()
        peak_stack[0] = max(peak_stack[0], len(ctx.stack))
        return key

    ctx.allocate = tracking_allocate  # type: ignore[assignment]
    result = func(*wrapped_args)

    del result
    del wrapped_args
    gc.collect()

    trace = list(ctx.trace)
    return {
        "strategy": strategy,
        "trace": trace,
        "cost_discrete": trace_to_cost_discrete(trace),
        "cost_continuous": trace_to_cost_continuous(trace),
        "n_reads": len(trace),
        "peak_stack": peak_stack[0],
    }


def working_set_proxy(n_reads: int, peak_live: int) -> int:
    """Simple bandwidth-times-footprint proxy."""

    return n_reads * usqrt(peak_live)
