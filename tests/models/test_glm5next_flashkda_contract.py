# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path


def _source(relative_path: str) -> ast.Module:
    root = Path(__file__).parents[2]
    return ast.parse((root / relative_path).read_text())


def test_glm_flashkda_call_supplies_required_buffers():
    helper_tree = _source("vllm/models/kimi_k3/nvidia/kda.py")
    helper = next(
        node
        for node in helper_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_flashkda_prefill"
    )
    positional = [*helper.args.posonlyargs, *helper.args.args]
    required = {
        arg.arg
        for arg, default in zip(
            positional,
            [None] * (len(positional) - len(helper.args.defaults))
            + list(helper.args.defaults),
        )
        if default is None
    }
    required.update(
        arg.arg
        for arg, default in zip(helper.args.kwonlyargs, helper.args.kw_defaults)
        if default is None
    )

    caller_tree = _source("vllm/models/glm5next/nvidia/kda.py")
    calls = [
        node
        for node in ast.walk(caller_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_flashkda_prefill"
    ]
    assert len(calls) == 1
    supplied = {keyword.arg for keyword in calls[0].keywords}
    assert required <= supplied
    assert {"out", "final_state", "workspace"} <= supplied
