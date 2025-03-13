import numpy as np
from TreehierarchyLTM import TreeHierarchy, TreeNode

class STMBatchAPI:
    def __init__(self):
        self.stm_cache = []
        self.ltm = TreeHierarchy()

    def add_to_stm(self, data):
        """Add data to the short-term memory cache."""
        self.stm_cache.append(data)

    def vectorize_stm(self):
        """Vectorize the short-term memory data."""
        return [np.array(list(data)) for data in self.stm_cache]

    def rollout_to_ltm(self):
        """Roll out vectorized short-term memory to long-term memory."""
        vectorized_stm = self.vectorize_stm()
        for vector in vectorized_stm:
            node = self.ltm.allocate_tree_node(self.ltm.root)
            success = self.ltm.insert_memory_block(node, vector.tobytes(), importance=100)  # Importance can be customized
            if success:
                self.stm_cache.remove(vector)
            else:
                print("Failed to insert memory block into LTM hierarchy")

    def destroy_stm_batch(self):
        """Clear the short-term memory cache."""
        self.stm_cache.clear()

    def confirm_with_ltm(self):
        """Confirm memory blocks in long-term memory based on gradient threshold."""
        for node in self.ltm.root:
            gradient = self.ltm.calculate_memory_gradient(node)
            if gradient > 50:  # Custom threshold
                print(f"Confirmed memory block with gradient {gradient} in LTM")
            else:
                print(f"Memory block with gradient {gradient} not confirmed in LTM")

    def process_stm_batch(self):
        """Process the short-term memory batch."""
        self.rollout_to_ltm()
        self.confirm_with_ltm()
        self.destroy_stm_batch()

# Example usage:
if __name__ == "__main__":
    stm_api = STMBatchAPI()
    stm_api.add_to_stm(b"short_term_memory_data_1")
    stm_api.add_to_stm(b"short_term_memory_data_2")
    stm_api.process_stm_batch()
