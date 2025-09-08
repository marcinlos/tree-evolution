import copy

import torch
from torch import nn

from tree_evolution.op import operator_map
from tree_evolution.tree import from_dict


class ExpressionModule(torch.nn.Module):
    def __init__(self, expr, ops):
        super().__init__()
        self.expr = copy.deepcopy(expr)
        self.param_count = self._annotate_expr_tree(self.expr, ops)
        self.params = torch.nn.Parameter(torch.randn(self.param_count))

    def forward(self, x):
        return self._eval(x, self.expr)

    def _eval(self, x, node):
        inputs = [self._eval(x, c) for c in node.children]
        return node.eval(x, inputs)

    def _annotate_expr_tree(self, root, ops):
        param_start = 0

        def process(node):
            nonlocal param_start

            if not node.label:
                node.eval = lambda x, inputs: x
            else:
                op = ops[node.label]

                def eval(x, inputs, start=param_start):
                    params = self.params[start : start + op.param_count]
                    return op.fun(*inputs, *params)

                node.eval = eval
                param_start += op.param_count

                for c in node.children:
                    process(c)

        process(root)
        return param_start


_fixed_activations = {
    "tanh": nn.Tanh(),
}


def decode_activations(data):
    ops = operator_map()

    def parse(label, content):
        match label:
            case "expr":
                tree = from_dict(content)
                return ExpressionModule(tree, ops)
            case "fixed":
                return _fixed_activations[content]
            case _:
                raise ValueError(f"Unknown activation type: {label}")

    return [parse(label, content) for label, content in data]
