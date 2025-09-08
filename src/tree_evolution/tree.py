import copy
import random


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


def pretty_print(root):
    queue = [(root, 0)]
    while queue:
        node, level = queue.pop()
        indent = "  " * level
        print(f"{indent}{node}")
        queue.extend([(c, level + 1) for c in node.children])


def node_list(root):
    nodes = []
    queue = [root]

    while queue:
        node = queue.pop()
        nodes.append(node)
        queue.extend(node.children)

    return nodes


def random_node(root):
    nodes = node_list(root)
    return random.choice(nodes)


def random_path(root, length):
    path = []
    node = root
    for _ in range(length):
        i = random.randrange(len(node.children))
        node = node.children[i]
        path.append(i)

    return tuple(path)


def select_node(root, path):
    node = root

    for i in path:
        node = node.children[i]

    return node


def swap_subtrees(parent1, pos1, parent2, pos2):
    a = parent1.children[pos1]
    b = parent2.children[pos2]
    parent1.children[pos1] = b
    parent2.children[pos2] = a
