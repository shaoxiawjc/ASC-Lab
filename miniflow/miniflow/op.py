from typing import List
import numpy as np
from node import Node


class Op:  # abstract class, you should inherit this class to implement your own operator
    def __call__(self):  # Differ it from __init__()
        node = Node()
        node.op = self
        return node

    def compute(self, node, input_vals: List):
        raise NotImplementedError

    def gradient(self, node, output_grad):
        raise NotImplementedError


def Variable(name):
    node = PlaceHolderOp()()
    node.name = name
    return node


class PlaceHolderOp(Op):
    def __call__(self):
        node = Op.__call__(self)
        node.op = self
        return node

    def compute(self, node, input_vals: List):
        return None

    def gradient(self, node, output_grad):
        return None


class AddOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.name = f"{node_A.name} + {node_B.name}"
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals: List):
        # y = a + b
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        # y = a+b => da = dy, db = dy
        # according to the formula above, you should cal grad_a and grad_b
        # and return [grad_a,grad_b]
        return [output_grad, output_grad]


class AddConstOp(Op):
    def __call__(self, node_A, const_attr):
        new_node = Op.__call__(self)
        new_node.name = f"{node_A.name} + {const_attr}"
        new_node.inputs = [node_A]
        new_node.const_attr = const_attr
        return new_node

    def compute(self, node, input_vals):
        # y = a + const
        return input_vals[0] + node.const_attr

    def gradient(self, node, output_grad):
        # da = dy
        return [output_grad]


class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.name = f"{node_A.name} * {node_B.name}"
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        # y = a * b
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad) -> List[Node]:
        # y = a*b => da = dy*b, db = dy*a
        # according to the formula above, you should cal grad_a and grad_b
        # and return [grad_a,grad_b]

        # TODO: Write your code below
        # start: copy from [GPT]
        # pass
        return [mul_op(output_grad, node.inputs[1]), mul_op(output_grad, node.inputs[0])]
        # end: copy from [GPT]


class MulConstOp(Op):
    def __call__(self, node, const_attr):
        new_node = Op.__call__(self)
        new_node.name = f"{node.name} * {const_attr}"
        new_node.inputs = [node]
        new_node.const_attr = const_attr
        return new_node

    def compute(self, node, input_vals):
        return input_vals[0] * node.const_attr

    def gradient(self, node, output_grad) -> List[Node]:
        # y = const*a => da = dy*const
        # according to the formula above, you should cal grad_a and grad_b
        # and return [grad_a]

        # TODO: Write your code below
        # pass
        return [mul_const_op(output_grad, node.const_attr)]
        


class OnesLikeOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        return new_node

    def compute(self, node, input_vals):
        # y = np.full_like(y,1)
        # according to the formula above, you should cal grad_y
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad) -> List[Node]:
        # y = np.full_like(y,1) => dy = 0
        return [zerolike_op(node.inputs[0])]


class ZeroLikeOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        return new_node

    def compute(self, node, input_vals):
        # y = np.zeros_like(y)
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad) -> List[Node]:
        # y = np.zeros_like(y) => dy = 0
        return [zerolike_op(node.inputs[0])]


class MatMulOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.name = f"{node_A.name} MatMal {node_B.name}"
        new_node.trans_A = trans_A  # A or A^T
        new_node.trans_B = trans_B  # B or B^T
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        # y = a * b
        val_a, val_b = input_vals[0], input_vals[1]
        # according to the attr to decide the value of a and b
        # and return the result
        # TODO: Write your code below
        # pass
        # start: copy from [GPT]
        if node.trans_A:
            val_a = val_a.T
        if node.trans_B:
            val_b = val_b.T
            # end: copy from [GPT]

        return np.dot(val_a, val_b)


        

    def gradient(self, node, output_grad) -> List[Node]:
        #  Y = A B => dA = dY B^T, dB = A^T dY
        #  According to the formula above, you should cal grad_a and grad_b
        #  and reuturn [grad_a, grad_b]

        # TODO: Write your code below
        # pass
        # start: copy from [GPT]
        grad_a = matmul_op(output_grad, node.inputs[1], trans_A=False, trans_B=True)
        grad_b = matmul_op(node.inputs[0], output_grad , trans_A=True, trans_B=False)
        return [grad_a, grad_b]
        # end: copy from [GPT]


# NOTION: Here, Instantiate the your operators
add_op = AddOp()
add_const_op = AddConstOp()
mul_op = MulOp()
mul_const_op = MulConstOp()
matmul_op = MatMulOp()
oneslike_op = OnesLikeOp()
zerolike_op = ZeroLikeOp()
