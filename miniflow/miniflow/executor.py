from op import *
from typing import Dict


class Executor:
    def __init__(self, eval_node_list):
        self.eval_node_list = eval_node_list
        self.graph = find_topo_sort(self.eval_node_list)

    def run(self, feed_dict: Dict):

        # eval_node_list 是需要求值的节点
        # calculate the value of each node in the graph
        node_to_val_map = {}
        for varaible, varaible_val in feed_dict.items():
            node_to_val_map[varaible] = varaible_val

        # start: copy from [GPT]
        # Now you have the value of the input nodes(feed_dict) and computation graph(self.graph)
        # TODO:Traverse graph in topological order and compute values for all nodes,write you code below
        # 根据计算图计算
        for node in self.graph:
            # 计算图是后序遍历的，因此，node的input都会被先计算
            if node not in node_to_val_map:
                # 节点不应该是输入值节点
                # 获取当前节点的输入值，然后计算
                now_node_input_val = [node_to_val_map[now_node_input_node] for now_node_input_node in node.inputs]
                # 将值存入，作为之后节点的输入值
                node_to_val_map[node] = node.op.compute(node, now_node_input_val)

        # hint: use node.op.compute(node, input_vals) to get value of node

        # return the val of each node
        return [node_to_val_map[node] for node in self.eval_node_list]
        # end: copy from [GPT]


def gradient(output_node: Node, node_list: List[Node]) -> List[Node]:
    # 创建反向传播计算图
    # node_list：原前向传播的非输出节点
    # output_node：原前向传播的输出节点
    # node_to_output_grads_list: 节点到对应梯度节点的映射
    # print(f"output_node: {output_node.name},  \n"
    #       f"node_list: {[node.name for node in node_list]}")

    node_to_output_grads_list = {}
    # 输出节点对应的梯度节点就是
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    node_to_output_grad = {}
    # Traverse forward graph in reverse topological order
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    # Traverse the graph in reverse topological order and calculate the gradient for each variable
    for node in reverse_topo_order:
        # pass
        # 把节点对应的梯度节点相加
        # start: copy from [GPT]
        #TODO: Sum the adjoints from all output nodes(hint: use sum_node_list)
        output_grad = node_to_output_grads_list.get(node, [])
        if len(output_grad) >= 2:
            node_to_output_grad[node] = sum_node_list(output_grad)
        elif len(output_grad) == 1:
            node_to_output_grad[node] = output_grad[0]
        else:
            node_to_output_grad[node] = None

        #TODO: Calculate the gradient of the node with respect to its inputs
        # 一个节点可能会出现多个输入的梯度
        input_grads_nodes = node.op.gradient(node, node_to_output_grad[node])

        #TODO: Traverse the inputs of node and add the partial adjoints to the list
        for i, input_node in enumerate(node.inputs):
            if input_node not in node_to_output_grads_list:
                node_to_output_grads_list[input_node] = []
            # 添加梯度
            node_to_output_grads_list[input_node].append(input_grads_nodes[i])
        # end: copy from [GPT]

    # return the gradient of each node in node_list
    return [node_to_output_grad[node] for node in node_list]


# ========================
# NOTION: Helper functions
# ========================
def find_topo_sort(node_list) -> List[Node]:
    """Given a list of nodes, return a topo ordering of nodes ending in them.
    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a
    topological sort.
    """
    visited = set()
    topo_order = []

    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node: Node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from functools import reduce
    return reduce(add_op, node_list)
