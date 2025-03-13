class TreeNode:
    def __init__(self, parent=None):
        self.parent = parent
        self.left_child = None
        self.right_child = None
        self.data = None
        self.max_capacity = 0xFFFFFFFF
        self.current_usage = 0
        self.flags = 0
        self.reserved = 0
        self.importance = 0  # New attribute to store importance of the data

    def is_leaf(self):
        return not self.left_child and not self.right_child

    def has_capacity_for(self, data_length):
        return self.current_usage + data_length <= self.max_capacity

class TreeHierarchy:
    def __init__(self):
        self.root = self.allocate_tree_node()
        self.current_node = self.root

    def allocate_tree_node(self, parent=None):
        return TreeNode(parent)

    def navigate_to_child(self, parent, left=True):
        if left:
            if not parent.left_child:
                parent.left_child = self.allocate_tree_node(parent)
            return parent.left_child
        else:
            if not parent.right_child:
                parent.right_child = self.allocate_tree_node(parent)
            return parent.right_child

    def navigate_to_parent(self, node):
        return node.parent

    def insert_memory_block(self, node, data, importance):
        if not node.has_capacity_for(len(data)):
            return False

        node.data = data
        node.current_usage += len(data)
        node.importance = importance
        return True

    def retrieve_memory_block(self, node):
        return node.data if node else None

    def calculate_memory_gradient(self, node):
        if not node:
            return 0
        usage_percentage = (node.current_usage * 100) // node.max_capacity
        gradient = usage_percentage + node.importance  # Include importance in gradient calculation
        # Apply other factors like time-decay and node depth here if needed
        return gradient

    def transfer_memory_between_layers(self, src_node, dst_node):
        if not src_node or not dst_node:
            return False

        dst_node.data = src_node.data
        dst_node.current_usage = src_node.current_usage
        dst_node.importance = src_node.importance  # Transfer importance as well
        src_node.data = None
        src_node.current_usage = 0
        src_node.importance = 0
        return True

# Example usage:
if __name__ == "__main__":
    tree = TreeHierarchy()
    node = tree.allocate_tree_node(tree.root)
    tree.insert_memory_block(node, b"example_data", 100)
    data = tree.retrieve_memory_block(node)
    print(f"Retrieved data: {data}")
