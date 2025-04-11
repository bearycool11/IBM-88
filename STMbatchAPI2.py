# -*- coding: utf-8 -*-
"""
Memory Batch API: Manages short-term memory (STM) batching and rollout to long-term memory (LTM).
License: Apache-2.0
"""

import logging
import uuid
from typing import Any, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from dataclasses import dataclass
from collections import deque

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("memory_batch.log")]
)
logger = logging.getLogger(__name__)

@dataclass
class TreeNode:
    """Node in the LTM tree hierarchy."""
    id: str
    memory: Optional[bytes] = None
    embedding: Optional[np.ndarray] = None
    importance: float = 0.0
    access_count: int = 0
    children: List['TreeNode'] = None
    parent: Optional['TreeNode'] = None

    def __post_init__(self):
        self.children = self.children or []
        self.id = str(uuid.uuid4()) if not self.id else self.id

class TreeHierarchy:
    """Long-term memory as a tree hierarchy."""
    def __init__(self, max_depth: int = 5, max_children: int = 10):
        self.root = TreeNode(id="root")
        self.max_depth = max_depth
        self.max_children = max_children
        self.node_map = {"root": self.root}

    def allocate_tree_node(self, parent: TreeNode) -> Optional[TreeNode]:
        """Allocate a new node under the parent."""
        try:
            if self._get_depth(parent) >= self.max_depth:
                logger.warning("Max tree depth reached")
                return None
            if len(parent.children) >= self.max_children:
                logger.warning(f"Max children reached for node {parent.id}")
                return None
            node = TreeNode(id=str(uuid.uuid4()), parent=parent)
            parent.children.append(node)
            self.node_map[node.id] = node
            logger.debug(f"Allocated node {node.id} under {parent.id}")
            return node
        except Exception as e:
            logger.error(f"Failed to allocate node: {e}")
            return None

    def insert_memory_block(self, node: TreeNode, memory: bytes, embedding: np.ndarray, importance: float) -> bool:
        """Insert a memory block into the node."""
        try:
            node.memory = memory
            node.embedding = embedding
            node.importance = max(0.0, min(importance, 100.0))
            logger.info(f"Inserted memory into node {node.id}, importance={node.importance:.2f}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert memory into node {node.id}: {e}")
            return False

    def calculate_memory_gradient(self, node: TreeNode) -> float:
        """Calculate a gradient based on importance, access count, and depth."""
        try:
            depth = self._get_depth(node)
            gradient = node.importance * 0.5 + node.access_count * 10.0 - depth * 5.0
            return max(0.0, gradient)
        except Exception as e:
            logger.error(f"Failed to calculate gradient for node {node.id}: {e}")
            return 0.0

    def _get_depth(self, node: TreeNode) -> int:
        """Calculate depth of a node from root."""
        depth = 0
        current = node
        while current.parent:
            depth += 1
            current = current.parent
        return depth

    def traverse(self):
        """Yield all nodes in the tree."""
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

class STMConfig(BaseModel):
    """Configuration for STMBatchAPI."""
    max_cache_size: int = Field(100, ge=1)
    gradient_threshold: float = Field(50.0, ge=0.0)
    embedding_model: str = "all-MiniLM-L6-v2"

class STMBatchAPI:
    """Manages STM batching and rollout to LTM with embeddings."""
    def __init__(self, config: Optional[STMConfig] = None):
        self.config = config or STMConfig()
        self.stm_cache: List[Any] = []
        self.ltm = TreeHierarchy()
        self.embedder = SentenceTransformer(self.config.embedding_model)
        logger.info("Initialized STMBatchAPI")

    def add_to_stm(self, data: Any):
        """Add data to STM cache."""
        try:
            if len(self.stm_cache) >= self.config.max_cache_size:
                logger.warning("STM cache full, ignoring new data")
                return
            self.stm_cache.append(data)
            logger.debug(f"Added to STM: {str(data)[:50]}...")
        except Exception as e:
            logger.error(f"Failed to add to STM: {e}")

    def vectorize_stm(self) -> List[Tuple[Any, np.ndarray]]:
        """Vectorize STM data into embeddings."""
        try:
            results = []
            for data in self.stm_cache:
                # Convert data to string for embedding
                data_str = data.decode() if isinstance(data, bytes) else str(data)
                embedding = self.embedder.encode([data_str])[0]
                results.append((data, embedding))
            logger.debug(f"Vectorized {len(results)} STM items")
            return results
        except Exception as e:
            logger.error(f"Failed to vectorize STM: {e}")
            return []

    def calculate_importance(self, data: Any, embedding: np.ndarray) -> float:
        """Estimate importance based on novelty (distance from existing LTM embeddings)."""
        try:
            ltm_embeddings = [n.embedding for n in self.ltm.traverse() if n.embedding is not None]
            if not ltm_embeddings:
                return 100.0  # Max importance for first entries
            distances = [np.linalg.norm(embedding - e) for e in ltm_embeddings]
            avg_distance = np.mean(distances) if distances else 0.0
            importance = min(100.0, avg_distance * 10.0)  # Scale to 0-100
            logger.debug(f"Calculated importance for {str(data)[:20]}...: {importance:.2f}")
            return importance
        except Exception as e:
            logger.error(f"Failed to calculate importance: {e}")
            return 50.0

    def rollout_to_ltm(self):
        """Roll out STM to LTM with importance scoring."""
        try:
            vectorized = self.vectorize_stm()
            for data, embedding in vectorized:
                node = self.ltm.allocate_tree_node(self.ltm.root)
                if not node:
                    logger.warning(f"Skipping rollout for {str(data)[:20]}...: No node allocated")
                    continue
                importance = self.calculate_importance(data, embedding)
                memory = data if isinstance(data, bytes) else str(data).encode()
                if self.ltm.insert_memory_block(node, memory, embedding, importance):
                    self.stm_cache.remove(data)
                    logger.info(f"Rolled out to LTM: {str(data)[:50]}...")
                else:
                    logger.warning(f"Failed to roll out: {str(data)[:50]}...")
        except Exception as e:
            logger.error(f"Rollout failed: {e}")

    def confirm_with_ltm(self):
        """Confirm LTM memories based on gradient threshold."""
        try:
            for node in self.ltm.traverse():
                if node.memory is None:
                    continue
                gradient = self.ltm.calculate_memory_gradient(node)
                if gradient > self.config.gradient_threshold:
                    node.access_count += 1
                    logger.info(f"Confirmed LTM node {node.id}, gradient={gradient:.2f}")
                else:
                    logger.debug(f"Unconfirmed LTM node {node.id}, gradient={gradient:.2f}")
        except Exception as e:
            logger.error(f"Confirmation failed: {e}")

    def process_stm_batch(self):
        """Process the STM batch: rollout and confirm."""
        logger.info("Processing STM batch")
        self.rollout_to_ltm()
        self.confirm_with_ltm()
        if not self.stm_cache:
            logger.info("STM cache cleared")
        else:
            logger.warning(f"STM cache not fully cleared: {len(self.stm_cache)} items remain")

# Demo
def demonstrate_memory_api():
    """Showcase the STM-to-LTM pipeline."""
    api = STMBatchAPI()
    inputs = [
        "AI ethics are critical",
        "Machine learning is advancing",
        "Ethical AI needs transparency",
        b"raw_data_sample"
    ]
    for inp in inputs:
        api.add_to_stm(inp)
    
    print("\nProcessing batch...")
    api.process_stm_batch()
    
    print("\nAdding and processing another batch...")
    api.add_to_stm("AI should prioritize safety")
    api.process_stm_batch()

if __name__ == "__main__":
    demonstrate_memory_api()
