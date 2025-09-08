import random

import leap_ec
import leap_ec.ops
from toolz import pipe

from tree_evolution.tree import Node, node_list, random_path, select_node, swap_subtrees


def random_expr_tree(depth, unary, binary):
    if depth == 0:
        return Node("")

    def child():
        op = random.choice(unary)
        subtree = random_expr_tree(depth - 1, unary, binary)
        return Node(op, [subtree])

    binary_op = random.choice(binary)
    return Node(binary_op, [child(), child()])


class CrossoverFunction(leap_ec.ops.Crossover):
    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = func

    def recombine(self, parent_a, parent_b):
        self.func(parent_a.genome, parent_b.genome)
        return parent_a, parent_b


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


class Evolution:
    def __init__(self, problem, size, representation):
        self.representation = representation
        self.problem = problem
        self.decoder = leap_ec.decoder.IdentityDecoder()
        self.size = size
        self.counter = leap_ec.util.inc_generation(context=leap_ec.context)

        parents = leap_ec.individual.Individual.create_population(
            self.size,
            problem=problem,
            initialize=representation.initialize,
            decoder=self.decoder,
        )
        self.population = leap_ec.individual.Individual.evaluate_population(parents)

    def step(self):
        offspring = pipe(
            self.population,
            leap_ec.ops.tournament_selection,
            leap_ec.ops.clone,
            self.representation.mutation,
            self.representation.crossover,
            leap_ec.ops.evaluate,
            leap_ec.ops.pool(size=self.size),
        )
        self.population = offspring
        self.counter()

    def print_population(self):
        leap_ec.probe.print_population(
            self.population, leap_ec.context["leap"]["generation"]
        )

    @property
    def generation(self):
        return self.counter.generation()
