# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import copy
import inspect
import random

import leap_ec
import leap_ec.probe
import leap_ec.problem
import leap_ec.util
import matplotlib.pyplot as plt
import numpy as np
import torch
from toolz import pipe
from torch import nn

# %% [markdown]
# # Tree infrastructure


# %%
class Node:
    def __init__(self, label, children=()):
        self.label = label
        self.children = list(children)

    def __deepcopy__(self, memo):
        return Node(
            self.label, children=[copy.deepcopy(c, memo) for c in self.children]
        )

    def __repr__(self):
        return f"Node({self.label})"


# %%
def pretty_print(root):
    queue = [(root, 0)]
    while queue:
        node, level = queue.pop()
        indent = "  " * level
        print(f"{indent}{node}")
        queue.extend([(c, level + 1) for c in node.children])


# %%
def node_list(root):
    nodes = []
    queue = [root]

    while queue:
        node = queue.pop()
        nodes.append(node)
        queue.extend(node.children)

    return nodes


# %%
def random_node(root):
    nodes = node_list(root)
    return random.choice(nodes)


# %%
def random_path(root, length):
    path = []
    node = root
    for _ in range(length):
        i = random.randrange(len(node.children))
        node = node.children[i]
        path.append(i)

    return tuple(path)


# %%
def select_node(root, path):
    node = root

    for i in path:
        node = node.children[i]

    return node


# %%
def swap_subtrees(parent1, pos1, parent2, pos2):
    a = parent1.children[pos1]
    b = parent2.children[pos2]
    parent1.children[pos1] = b
    parent2.children[pos2] = a


# %%
def random_expr_tree(depth, unary, binary):
    if depth == 0:
        return Node("")

    def child():
        op = random.choice(unary)
        subtree = random_expr_tree(depth - 1, unary, binary)
        return Node(op, [subtree])

    binary_op = random.choice(binary)
    return Node(binary_op, [child(), child()])


# %% [markdown]
# # Tree evolution


# %%
class CrossoverFunction(leap_ec.ops.Crossover):
    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = func

    def recombine(self, parent_a, parent_b):
        self.func(parent_a.genome, parent_b.genome)
        return parent_a, parent_b


# %%
class ForestRepresentation:
    def __init__(self, count, depth, unary, binary):
        self.count = count
        self.depth = depth
        self.unary = unary
        self.binary = binary

    def initialize(self):
        return [
            random_expr_tree(depth=self.depth, unary=self.unary, binary=self.binary)
            for _ in range(self.count)
        ]

    def _mutate(self, individual):
        for tree in individual.genome:
            nodes = node_list(tree)
            node = random.choice(nodes)

            if node.label in self.unary:
                node.label = random.choice(self.unary)
            elif node.label in self.binary:
                node.label = random.choice(self.binary)

    def _crossover(self, parent_a, parent_b):
        for tree_a, tree_b in zip(parent_a, parent_b, strict=True):
            depth = random.randrange(1, self.depth * 2)
            path_a = random_path(tree_a, depth)
            node_a = select_node(tree_a, path_a[:-1])

            path_b = random_path(tree_b, depth)
            node_b = select_node(tree_b, path_b[:-1])

            swap_subtrees(node_a, path_a[-1], node_b, path_b[-1])

    @property
    def mutation(self):
        @leap_ec.ops.iteriter_op
        def func(next_individual):
            while True:
                individual = next(next_individual)
                self._mutate(individual)
                individual.fitness = None
                yield individual

        return func

    @property
    def crossover(self):
        return CrossoverFunction(self._crossover, persist_children=False, p_xover=1.0)


# %% [markdown]
# # Example evolutionary computation


# %%
def fitness(trees):
    def eval_tree(root):
        nodes = node_list(root)
        return sum(1 if n.label.lower() == "c" else 0 for n in nodes)

    return sum(eval_tree(tree) for tree in trees)


problem = leap_ec.problem.FunctionProblem(fitness, maximize=True)

# %%
decoder = leap_ec.decoder.IdentityDecoder()

# %%
UNARY = ("A", "B", "C")
BINARY = ("a", "b", "c", "d")

# %%
forest = ForestRepresentation(count=3, depth=3, unary=UNARY, binary=BINARY)

# %%
parents = leap_ec.individual.Individual.create_population(
    10, initialize=forest.initialize, decoder=decoder, problem=problem
)

# %%
# Evaluate initial population
parents = leap_ec.individual.Individual.evaluate_population(parents)

# %%
# print initial, random population
leap_ec.probe.print_population(parents, generation=0)

# %%
# generation_counter is an optional convenience for generation tracking
generation_counter = leap_ec.util.inc_generation(context=leap_ec.context)

# %%
while generation_counter.generation() < 300:
    offspring = pipe(
        parents,
        leap_ec.ops.tournament_selection,
        leap_ec.ops.clone,
        forest.mutation,
        forest.crossover,
        leap_ec.ops.evaluate,
        leap_ec.ops.pool(size=len(parents)),
    )
    parents = offspring

    generation_counter()  # increment to the next generation

    if generation_counter.generation() % 50 == 0:
        leap_ec.probe.print_population(parents, leap_ec.context["leap"]["generation"])
        print()


# %% [markdown]
# # Neural network infrastructure


# %%
class ExpressionModule(nn.Module):
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


# %%
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


# %%
_operators = {}


def _unary():
    return [label for label, op in _operators.items() if op.input_count == 1]


def _binary():
    return [label for label, op in _operators.items() if op.input_count == 2]


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


# %%
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


# %%
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")


# %%
class PINN(nn.Module):
    def __init__(self, trees, size=512):
        super().__init__()

        modules = [nn.Linear(2, size)]

        for expr in trees:
            activation = ExpressionModule(expr, _operators)
            modules.append(activation)
            modules.append(nn.Linear(size, size))

        modules[-1] = nn.Linear(size, 1)

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        x.requires_grad_()
        return self.layers(x)

    def grad(self, x):
        vals = self(x)
        return torch.autograd.grad(
            vals,
            x,
            grad_outputs=torch.ones_like(vals),
            create_graph=True,
        )[0]


# %% [markdown]
# # Neural network example

# %%
expr1 = random_expr_tree(depth=1, unary=_unary(), binary=_binary())
expr2 = random_expr_tree(depth=1, unary=_unary(), binary=_binary())

# %%
pinn = PINN([expr1, expr2], size=512).to(DEVICE)
print(pinn)


# %%
class LossFunction:
    def __init__(self, fun, n):
        self.fun = fun

        x_raw = torch.linspace(0, 1, steps=n)
        y_raw = torch.linspace(0, 1, steps=n)
        x, y = torch.meshgrid(x_raw, y_raw, indexing="ij")

        self.x = x.reshape(-1, 1).requires_grad_(True).to(DEVICE)
        self.y = y.reshape(-1, 1).requires_grad_(True).to(DEVICE)
        self.points = torch.cat([self.x, self.y], dim=1)

    def __call__(self, pinn):
        expected = fun(self.x, self.y)
        actual = pinn(self.points)
        difference = expected - actual
        return difference.pow(2).mean()


# %%
def fun(x, y):
    return torch.exp(x + y)


# %%
loss_fn = LossFunction(fun, 100)


# %%
def train(pinn, steps, verbose=False):
    optimizer = torch.optim.Adamax(pinn.parameters())
    log = []

    for i in range(steps):
        loss = loss_fn(pinn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        log.append(loss_val)

        if verbose and i % 10 == 0:
            print(f"Iter {i}: loss = {loss_val}")

    return log


# %%
_ = train(pinn, 200, verbose=True)

# %% [markdown]
# # Evolutionary algorithm + neural network example

# %%
COUNT = 2
DEPTH = 2
EPOCHS = 100
POPULATION_SIZE = 5


# %%
def fun_to_approximate(x, y):
    return torch.exp(x + y)


loss_fn = LossFunction(fun, n=20)


# %%
def fitness(trees):
    pinn = PINN(trees).to(DEVICE)
    losses = train(pinn, EPOCHS)
    err = np.array(losses)
    return np.min(err)


problem = leap_ec.problem.FunctionProblem(fitness, maximize=False)

# %%
decoder = leap_ec.decoder.IdentityDecoder()

# %%
forest = ForestRepresentation(
    count=COUNT, depth=DEPTH, unary=_unary(), binary=_binary()
)

# %%
parents = leap_ec.individual.Individual.create_population(
    POPULATION_SIZE, initialize=forest.initialize, decoder=decoder, problem=problem
)

# %%
# Evaluate initial population
parents = leap_ec.individual.Individual.evaluate_population(parents)

# %%
# print initial, random population
leap_ec.probe.print_population(parents, generation=0)

# %%
generation_counter = leap_ec.util.inc_generation(context=leap_ec.context)

# %%
while generation_counter.generation() < 30:
    offspring = pipe(
        parents,
        leap_ec.ops.tournament_selection,
        leap_ec.ops.clone,
        forest.mutation,
        forest.crossover,
        leap_ec.ops.evaluate,
        leap_ec.ops.pool(size=len(parents)),
    )
    parents = offspring

    generation_counter()  # increment to the next generation

    if generation_counter.generation() % 1 == 0:
        leap_ec.probe.print_population(parents, leap_ec.context["leap"]["generation"])
        print()

# %%
trees = parents[0].genome
tree = trees[0]

mod = ExpressionModule(tree, _operators)
x = torch.linspace(-1, 1, 100, requires_grad=False)
y = mod(x)
plt.plot(x.numpy(), y.detach().numpy())
