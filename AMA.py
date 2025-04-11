# -*- coding: utf-8 -*-
"""
yacineMTB and bearycool11 are locked in
Adaptive Memory Architecture (AMA)
A lean, biomimetic memory system for AI with semantic embeddings and efficient retrieval.
"""


import uuid
import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    """Represents a single memory with content and metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.5
    access_count: int = 0
    tags: set = field(default_factory=set)

    def update_access(self):
        """Increment access count for importance tracking."""
        self.access_count += 1

class MemoryConfig(BaseModel):
    """Configuration for MemoryManager."""
    max_capacity: int = 1000
    similarity_threshold: float = 0.7
    embedding_model: str = "all-MiniLM-L6-v2"

class MemoryManager:
    """Core memory management system with semantic search and compression."""
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.memories: Dict[str, MemoryItem] = {}
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        self.index = faiss.IndexFlatL2(384)  # Dimension matches MiniLM output
        self.id_to_index: Dict[str, int] = {}
        self.index_count = 0

    def insert(self, content: str, confidence: float = 0.5, tags: Optional[set] = None) -> MemoryItem:
        """Insert a new memory item with embedding."""
        if len(self.memories) >= self.config.max_capacity:
            self._prune()

        memory = MemoryItem(content=content, confidence=confidence, tags=tags or set())
        memory.embedding = self.embedding_model.encode([content])[0]
        
        self.memories[memory.id] = memory
        self.id_to_index[memory.id] = self.index_count
        self.index.add(memory.embedding.reshape(1, -1))
        self.index_count += 1
        
        logger.info(f"Inserted memory: {content[:50]}...")
        return memory

    def _prune(self):
        """Remove least accessed memory to maintain capacity."""
        if not self.memories:
            return
        lru_id = min(self.memories.items(), key=lambda x: x[1].access_count)[0]
        self._remove_memory(lru_id)

    def _remove_memory(self, memory_id: str):
        """Remove a memory from storage and index."""
        index = self.id_to_index.pop(memory_id, None)
        if index is not None:
            self.index.remove_ids(np.array([index]))
        del self.memories[memory_id]
        logger.info(f"Pruned memory ID: {memory_id}")

    def retrieve(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """Retrieve memories similar to the query."""
        query_embedding = self.embedding_model.encode([query])[0]
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            memory_id = next(k for k, v in self.id_to_index.items() if v == idx)
            memory = self.memories[memory_id]
            similarity = 1 - (distance / 2)  # Normalize to [0,1]
            if similarity >= self.config.similarity_threshold:
                memory.update_access()
                results.append(memory)
        
        logger.info(f"Retrieved {len(results)} memories for query: {query[:50]}...")
        return results

    def compress(self) -> Dict:
        """Deduplicate similar memories."""
        to_remove = []
        to_add = []
        
        for mem1_id, mem1 in list(self.memories.items()):
            for mem2_id, mem2 in list(self.memories.items()):
                if mem1_id >= mem2_id or mem1_id in to_remove:
                    continue
                similarity = 1 - faiss.pairwise_distances(
                    mem1.embedding.reshape(1, -1), mem2.embedding.reshape(1, -1)
                )[0][0] / 2
                if similarity >= 0.9:
                    combined_content = f"{mem1.content} | {mem2.content}"
                    to_remove.append(mem2_id)
                    to_add.append(MemoryItem(
                        content=combined_content[:200],  # Limit length
                        confidence=max(mem1.confidence, mem2.confidence),
                        tags=mem1.tags | mem2.tags
                    ))
        
        for mem_id in to_remove:
            self._remove_memory(mem_id)
        for mem in to_add:
            self.insert(mem.content, mem.confidence, mem.tags)
        
        stats = {
            "removed": len(to_remove),
            "added": len(to_add),
            "total": len(self.memories)
        }
        logger.info(f"Compression stats: {stats}")
        return stats

# Demo
def demonstrate_memory_system():
    """Show off the memory system."""
    manager = MemoryManager()
    
    # Insert some memories
    manager.insert("AI ethics are crucial for development", confidence=0.9, tags={"ethics", "AI"})
    manager.insert("Machine learning is evolving fast", confidence=0.95, tags={"ML", "tech"})
    manager.insert("Ethical AI needs transparency", confidence=0.85, tags={"ethics", "AI"})
    
    # Retrieve
    results = manager.retrieve("AI ethics")
    print("\nRetrieved Memories:")
    for mem in results:
        print(f"- {mem.content} (Confidence: {mem.confidence}, Tags: {mem.tags})")
    
    # Compress
    stats = manager.compress()
    print(f"\nCompression Stats: {stats}")

if __name__ == "__main__":
    demonstrate_memory_system()
