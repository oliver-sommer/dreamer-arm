"""nn.Module helpers: weight init, parameter tree, optimizer state collection."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import init as nn_init


def weight_init_(m: nn.Module, fan_type: str = "in") -> None:
    """In-place truncated-normal weight init scaled by fan, bias zero, RMSNorm scale = 1.

    Replicates the initialiser used in the reference R2-Dreamer implementation
    (factor 1.1368 ~= sqrt(2 / pi) compensating for the truncation at +/-2 sigma).
    """
    if isinstance(m, nn.RMSNorm):
        with torch.no_grad():
            m.weight.fill_(1.0)
        return

    weight = getattr(m, "weight", None)
    if weight is None or weight.numel() == 0:
        return

    # `_calculate_fan_in_and_fan_out` is a private but stable torch helper.
    in_num, out_num = nn_init._calculate_fan_in_and_fan_out(weight)
    fan = {"avg": (in_num + out_num) / 2, "in": in_num, "out": out_num}[fan_type]

    with torch.no_grad():
        std = 1.1368 * float(np.sqrt(1.0 / fan))
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        bias = getattr(m, "bias", None)
        if bias is not None:
            bias.fill_(0.0)


class Every:
    """Schedules a callback every ``every`` steps (returns count of triggers)."""

    def __init__(self, every: int) -> None:
        self._every = every
        self._last: int | None = None

    def __call__(self, step: int) -> int:
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    """Returns True only on the first call, False thereafter."""

    def __init__(self) -> None:
        self._once = True

    def __call__(self) -> bool:
        if self._once:
            self._once = False
            return True
        return False


def recursively_collect_optim_state_dict(
    obj: Any,
    path: str = "",
    state_dicts: dict[str, dict[str, Any]] | None = None,
    visited: set[int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Walk an object graph and collect ``state_dict()`` from every Optimizer found."""
    if state_dicts is None:
        state_dicts = {}
    if visited is None:
        visited = set()
    if id(obj) in visited:
        return state_dicts
    visited.add(id(obj))

    attrs: dict[str, Any] = dict(obj.__dict__) if hasattr(obj, "__dict__") else {}
    if isinstance(obj, nn.Module):
        attrs.update({k: m for k, m in obj.named_modules() if "." not in k and obj is not m})

    for name, attr in attrs.items():
        new_path = f"{path}.{name}" if path else name
        if isinstance(attr, torch.optim.Optimizer):
            state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            recursively_collect_optim_state_dict(attr, new_path, state_dicts, visited)
    return state_dicts


def recursively_load_optim_state_dict(obj: Any, state_dicts: dict[str, dict[str, Any]]) -> None:
    for path, state in state_dicts.items():
        target = obj
        for key in path.split("."):
            target = getattr(target, key)
        target.load_state_dict(state)


def build_module_tree(module: nn.Module, module_name: str = "") -> dict[str, Any]:
    """Recursively summarise a module into a {name, params, children, total} tree."""
    direct = 0
    params: dict[str, int] = {}
    for pname, p in module.named_parameters(recurse=False):
        params[pname] = p.numel()
        direct += p.numel()
    children: dict[str, dict[str, Any]] = {
        cname: build_module_tree(child, cname) for cname, child in module.named_children()
    }
    total = direct + sum(c["total"] for c in children.values())
    return {"name": module_name, "params": params, "children": children, "total": total}


def print_module_tree(info: dict[str, Any], parent_path: str = "", indent: int = 0) -> None:
    """Pretty-print the tree from :func:`build_module_tree` sorted by param count."""
    name = info["name"]
    if not parent_path:
        full_path = name
    elif name:
        full_path = f"{parent_path}/{name}"
    else:
        full_path = parent_path

    print(" " * indent + f"{info['total']:11,d} {full_path}")
    param_nodes = [
        {"name": pn, "params": {}, "children": {}, "total": n} for pn, n in info["params"].items()
    ]
    combined = param_nodes + list(info["children"].values())
    combined.sort(key=lambda x: x["total"], reverse=True)
    for child in combined:
        print_module_tree(child, full_path, indent + 2)
