#!/usr/bin/env python3
"""
tree_hierarchy.py – Christmas Tree Memory Hierarchy

A vibrant, multi-child tree for long-term NPC memory, sparkling with importance and context.
Nodes grow dynamically, tagged with metadata, and glow with gradient priority.
Designed for integration with npc_system.py’s KnowledgeGraph and MemorySilo.

Run example: python tree_hierarchy.py
"""

import time
from typing import List, Optional
from dataclasses import dataclass, field
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class TreeNode:
    """A memory node decorated like a Christmas ornament."""
    data: bytes = None
    importance: float = 0.0  # 0-100, how critical this memory is
    timestamp: float = field(default_factory=time.time)  # When it was added
    tags: List[str] = field(default_factory=list)  # Contextual tags (e.g., "flirt", "fight")
    children: List['TreeNode'] = field(default_factory=list)  # Multi-child branching
    parent: Optional['TreeNode'] = None
    max_capacity: int = 0xFFFFFFFF  # Max bytes per node
    current_usage: int = 0  # Current bytes used

    def is_leaf(self) -> bool:
        return not self.children

    def has_capacity_for(self, data_length: int) -> bool:
        return self.current_usage + data_length <= self.max_capacity

    def add_child(self, child: 'TreeNode'):
        child.parent = self
        self.children.append(child)

class TreeHierarchy:
    """A festive Christmas tree for long-term memory storage."""
    def __init__(self):
        self.root = TreeNode(tags=["root"], importance=100.0)
        self.current_node = self.root
        self.node_count = 1  # Track total ornaments

    def allocate_tree_node(self, parent: TreeNode = None, tags: List[str] = None) -> TreeNode:
        """Create a new ornament with a festive glow."""
        node = TreeNode(parent=parent, tags=tags or [])
        if parent:
            parent.add_child(node)
        self.node_count += 1
        logger.debug(f"Allocated node {self.node_count} under {parent.tags if parent else 'root'}")
        return node

    def insert_memory_block(self, data: bytes, importance: float, tags: List[str] = None) -> bool:
        """Hang a memory ornament on the tree, finding the best branch."""
        node = self.find_best_node(data, importance)
        if not node or not node.has_capacity_for(len(data)):
            logger.warning(f"No capacity for {len(data)} bytes with importance {importance}")
            return False

        node.data = data
        node.current_usage = len(data)
        node.importance = importance
        node.tags.extend(tags or [])
        node.timestamp = time.time()
        logger.info(f"Hung memory '{data.decode(errors='ignore')}' on tree (imp: {importance}, tags: {node.tags})")
        return True

    def find_best_node(self, data: bytes, importance: float) -> Optional[TreeNode]:
        """Search for the perfect branch based on capacity and importance."""
        def traverse(node: TreeNode, depth: int = 0) -> Optional[TreeNode]:
            if node.has_capacity_for(len(data)):
                if not node.data:  # Empty node is ideal
                    return node
                if node.importance < importance:  # Replace less important
                    return node
            # Explore children based on importance gradient
            sorted_children = sorted(node.children, key=lambda x: x.importance, reverse=True)
            for child in sorted_children:
                result = traverse(child, depth + 1)
                if result:
                    return result
            # If no fit, grow a new branch
            if depth < 5 and len(node.children) < 4:  # Max depth 5, max 4 kids
                return self.allocate_tree_node(node)
            return None

        return traverse(self.root)

    def retrieve_memory_block(self, tags: List[str] = None) -> Optional[bytes]:
        """Fetch a shiny memory by tags, prioritizing recency and importance."""
        def traverse(node: TreeNode) -> Optional[TreeNode]:
            if node.data and (not tags or any(tag in node.tags for tag in tags)):
                return node
            sorted_children = sorted(node.children, key=lambda x: (x.timestamp, x.importance), reverse=True)
            for child in sorted_children:
                result = traverse(child)
                if result:
                    return result
            return None

        node = traverse(self.root)
        if node:
            logger.info(f"Retrieved memory '{node.data.decode(errors='ignore')}' (tags: {node.tags})")
            return node.data
        logger.debug(f"No memory found for tags {tags}")
        return None

    def calculate_memory_gradient(self, node: TreeNode) -> float:
        """Make the node glow with a festive gradient—usage, importance, recency, depth."""
        if not node or not node.data:
            return 0.0
        usage_percent = (node.current_usage * 100) / node.max_capacity
        recency = (time.time() - node.timestamp) / 86400  # Decay over days
        depth = self.get_depth(node)
        gradient = (usage_percent * 0.3) + (node.importance * 0.5) - (recency * 10) - (depth * 2)
        return max(0.0, min(100.0, gradient))

    def get_depth(self, node: TreeNode) -> int:
        """Calculate how deep this ornament hangs."""
        depth = 0
        current = node
        while current.parent:
            depth += 1
            current = current.parent
        return depth

    def transfer_memory_between_layers(self, src_node: TreeNode, dst_node: TreeNode) -> bool:
        """Shift an ornament to a new branch, keeping the tree balanced."""
        if not src_node or not dst_node or not src_node.data or not dst_node.has_capacity_for(src_node.current_usage):
            return False
        dst_node.data = src_node.data
        dst_node.current_usage = src_node.current_usage
        dst_node.importance = src_node.importance
        dst_node.tags = src_node.tags[:]
        dst_node.timestamp = src_node.timestamp
        src_node.data = None
        src_node.current_usage = 0
        src_node.importance = 0
        src_node.tags = []
        logger.info(f"Transferred memory from depth {self.get_depth(src_node)} to {self.get_depth(dst_node)}")
        return True

    def prune_tree(self, min_gradient: float = 10.0):
        """Trim faded ornaments to keep the tree lush."""
        def traverse(node: TreeNode, parent: TreeNode):
            if node.data and self.calculate_memory_gradient(node) < min_gradient:
                logger.debug(f"Pruning faded memory '{node.data.decode(errors='ignore')}' (gradient: {self.calculate_memory_gradient(node):.2f})")
                node.data = None
                node.current_usage = 0
                node.importance = 0
                node.tags = []
            node.children = [child for child in node.children if traverse(child, node)]
            return node.children or node.data

        self.root.children = [child for child in self.root.children if traverse(child, self.root)]
        logger.info("Tree pruned—looking festive!")

# Example usage
if __name__ == "__main__":
    tree = TreeHierarchy()
    # Hang some ornaments
    tree.insert_memory_block(b"Yo, what’s good?", 80.0, ["greeting"])
    tree.insert_memory_block(b"You’re hot!", 90.0, ["flirt"])
    tree.insert_memory_block(b"Fight broke out!", 70.0, ["event"])
    # Fetch a memory
    data = tree.retrieve_memory_block(["flirt"])
    print(f"Retrieved: {data.decode() if data else 'Nothing'}")
    # Check gradients
    for node in tree.root.children:
        print(f"Node: {node.data.decode()}, Gradient: {tree.calculate_memory_gradient(node):.2f}")
    # Prune the tree
    tree.prune_tree(50.0)
    print(f"Tree has {tree.node_count} nodes after pruning")
