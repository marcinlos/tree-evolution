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


class OperatorRegister:
    def __init__(self):
        self._ops = {}

    def _register_operator(self, fun, arity, label=None):
        if label is None:
            label = fun.__name__

        sig = inspect.signature(fun)
        arg_count = len(sig.parameters)
        param_count = arg_count - arity

        self._ops[label] = Op(label, arity, param_count, fun)

    def map(self):
        return self._ops.copy()

    @property
    def unary_operators(self):
        return self._ops_with_arity(1)

    @property
    def binary_operators(self):
        return self._ops_with_arity(2)

    def _ops_with_arity(self, n):
        return tuple(label for label, op in self._ops.items() if op.input_count == n)

    def unary(self, fun):
        self._register_operator(fun, arity=1)
        return fun

    def binary(self, fun):
        self._register_operator(fun, arity=2)
        return fun

    def copy(self):
        reg = OperatorRegister()
        reg._ops = self._ops.copy()


_operators = OperatorRegister()


def default():
    return _operators.copy()


@_operators.unary
def zero(x):
    return torch.zeros_like(x)


@_operators.unary
def const(x, c):
    return c * torch.ones_like(x)


@_operators.unary
def scaled(x, a):
    return a * x


@_operators.unary
def abs(x, a):
    return a * torch.abs(x)


@_operators.unary
def square(x, a):
    return a * x**2


@_operators.unary
def cube(x, a):
    return a * x**3


@_operators.unary
def sqrt(x, a):
    return a * torch.sqrt(x)


@_operators.unary
def exp(x, a, b):
    return a * torch.exp(b * x)


@_operators.unary
def exp2(x, a, b):
    return a * torch.exp(b * x**2)


@_operators.unary
def logabs(x, a, b, c):
    return a * torch.log(torch.abs(b + c * x))


@_operators.unary
def logexp(x, a, b, c):
    return a * torch.log(torch.abs(b + torch.exp(c * x)))


@_operators.unary
def sin(x, a, b):
    return a * torch.sin(b * x)


@_operators.unary
def sinh(x, a, b):
    return a * torch.sinh(b * x)


@_operators.unary
def asinh(x, a, b):
    return a * torch.asinh(b * x)


@_operators.unary
def cos(x, a, b):
    return a * torch.cos(b * x)


@_operators.unary
def cosh(x, a, b):
    return a * torch.cosh(b * x)


@_operators.unary
def tanh(x, a, b):
    return a * torch.tanh(b * x)


@_operators.unary
def atanh(x, a, b):
    return a * torch.atanh(torch.clamp(b * x, -1, 1))


@_operators.unary
def max0(x, a, b):
    return a * torch.maximum(b * x, torch.tensor(0.0))


@_operators.unary
def min0(x, a, b):
    return a * torch.minimum(b * x, torch.tensor(0.0))


@_operators.unary
def erf(x, a, b):
    return a * torch.erf(b * x)


@_operators.unary
def sinc(x, a, b):
    return a * torch.sinc(b * x)


@_operators.binary
def sum(x, y, a, b):
    return a * x + b * y


@_operators.binary
def product(x, y, a):
    return a * x * y


@_operators.binary
def quot(x, y, a, b):
    return x / (a * y + b)


@_operators.binary
def min(x, y, a, b):
    return torch.minimum(a * x, b * y)


@_operators.binary
def max(x, y, a, b):
    return torch.maximum(a * x, b * y)
