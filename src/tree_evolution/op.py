import inspect

import torch


class Op:
    def __init__(self, label, input_count, param_count, fun):
        self.label = label
        self.input_count = input_count
        self.param_count = param_count
        self.fun = fun

    def __repr__(self):
        return (
            f"Op({self.label}, input_count={self.input_count}, "
            f"param_count={self.param_count})"
        )


_operators = {}


def operator_map():
    return _operators.copy()


def unary_operators():
    return tuple(label for label, op in _operators.items() if op.input_count == 1)


def binary_operators():
    return tuple(label for label, op in _operators.items() if op.input_count == 2)


def _register_operator(input_count, fun, label=None):
    if label is None:
        label = fun.__name__

    sig = inspect.signature(fun)
    arg_count = len(sig.parameters)
    param_count = arg_count - input_count

    _operators[label] = Op(label, input_count, param_count, fun)


def unary_op(fun):
    _register_operator(1, fun)
    return fun


def binary_op(fun):
    _register_operator(2, fun)
    return fun


@unary_op
def zero(x):
    return torch.zeros_like(x)


@unary_op
def const(x, c):
    return c * torch.ones_like(x)


@unary_op
def scaled(x, a):
    return a * x


@unary_op
def abs(x, a):
    return a * torch.abs(x)


@unary_op
def square(x, a):
    return a * x**2


@unary_op
def exp(x, a, b):
    return a * torch.exp(b * x)


@binary_op
def sum(x, y, a, b):
    return a * x + b * y


@binary_op
def product(x, y, a):
    return a * x * y


@binary_op
def quot(x, y, a, b):
    return x / (a * y + b)
