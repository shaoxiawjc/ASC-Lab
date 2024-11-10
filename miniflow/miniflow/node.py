

class Node(object):
    def __init__(self):
        """
            Paraemeter:
            1. inputs : list of input nodes(Node type)
            2. op : operation of the node(Op type)
            3. const_attr : constant attribute of the node for add_const_op and mul_const_op
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""
    
    def __str__(self):
        return self.name
    
 
