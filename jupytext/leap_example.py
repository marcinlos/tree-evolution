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

# %% editable=true slideshow={"slide_type": ""}
import leap_ec
import leap_ec.probe
import leap_ec.problem
import leap_ec.util
from toolz import pipe

from tree_evolution.evolution import ForestRepresentation
from tree_evolution.tree import node_list

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

# %% editable=true slideshow={"slide_type": ""}
forest = ForestRepresentation(count=3, depth=3, unary=UNARY, binary=BINARY)

# %%
parents = leap_ec.individual.Individual.create_population(
    10,
    initialize=forest.initialize,
    decoder=decoder,
    problem=problem,
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

# %% editable=true slideshow={"slide_type": ""}
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
