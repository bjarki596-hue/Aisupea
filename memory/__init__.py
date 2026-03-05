"""
Aisupea Memory Systems

Multiple memory modules for autonomous AI agents.
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from ..core import Tensor
from ..numpy_compat import ndarray, dot, normalize


class VectorSimilarityMemory:
    """
    Vector similarity memory for storing and retrieving embeddings.

    Uses cosine similarity for retrieval.
    """

    def __init__(self, embed_dim: int, max_size: int = 10000):
        self.embed_dim = embed_dim
        self.max_size = max_size
        self.vectors: List[ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self.index: Dict[str, List[int]] = defaultdict(list)

    def add(self, vector: ndarray, metadata: Dict[str, Any], key: Optional[str] = None):
        """
        Add vector to memory.

        Args:
            vector: Embedding vector
            metadata: Associated metadata
            key: Optional key for indexing
        """
        if len(self.vectors) >= self.max_size:
            # Remove oldest entry
            self.vectors.pop(0)
            self.metadata.pop(0)

        # Normalize vector for cosine similarity
        normalized_vector = normalize(vector)
        self.vectors.append(normalized_vector)
        self.metadata.append(metadata)

        if key:
            self.index[key].append(len(self.vectors) - 1)

    def search(self, query_vector: ndarray, top_k: int = 5,
              key_filter: Optional[str] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            key_filter: Optional key to filter results

        Returns:
            List of (metadata, similarity_score) tuples
        """
        if not self.vectors:
            return []

        # Normalize query vector
        query_normalized = normalize(query_vector)

        # Calculate similarities
        similarities = []
        candidates = self.index[key_filter] if key_filter else range(len(self.vectors))

        for idx in candidates:
            similarity = dot(query_normalized, self.vectors[idx]).item()
            similarities.append((idx, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        results = []
        for idx, score in similarities[:top_k]:
            results.append((self.metadata[idx], score))

        return results

    def get_by_key(self, key: str) -> List[Dict[str, Any]]:
        """Get all memories associated with a key."""
        indices = self.index[key]
        return [self.metadata[idx] for idx in indices]

    def clear(self):
        """Clear all memories."""
        self.vectors.clear()
        self.metadata.clear()
        self.index.clear()

    def size(self) -> int:
        """Get current memory size."""
        return len(self.vectors)


class ContextMemory:
    """
    Context memory for conversation history.

    Maintains sliding window of recent conversation turns.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add_turn(self, role: str, content: str):
        """
        Add a conversation turn.

        Args:
            role: Role of the speaker ('user' or 'assistant')
            content: Content of the message
        """
        turn = {
            'role': role,
            'content': content,
            'timestamp': self._get_timestamp()
        }

        self.history.append(turn)

        # Maintain sliding window
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """
        Get formatted conversation context.

        Args:
            max_tokens: Maximum number of tokens (approximate)

        Returns:
            Formatted conversation string
        """
        if not self.history:
            return ""

        context_parts = []
        current_tokens = 0

        # Build context from most recent to oldest
        for turn in reversed(self.history):
            turn_text = f"{turn['role']}: {turn['content']}\n"

            if max_tokens and current_tokens + len(turn_text.split()) > max_tokens:
                break

            context_parts.insert(0, turn_text)
            current_tokens += len(turn_text.split())

        return "".join(context_parts).strip()

    def get_recent_turns(self, n: int = 5) -> List[Dict[str, str]]:
        """Get last n conversation turns."""
        return self.history[-n:] if n > 0 else []

    def clear(self):
        """Clear conversation history."""
        self.history.clear()

    def _get_timestamp(self) -> float:
        """Get current timestamp (simplified)."""
        import time
        return time.time()

    def size(self) -> int:
        """Get number of turns in memory."""
        return len(self.history)


class TaskMemory:
    """
    Task memory for storing completed tasks and outcomes.

    Helps with task planning and learning from experience.
    """

    def __init__(self, max_tasks: int = 1000):
        self.max_tasks = max_tasks
        self.tasks: List[Dict[str, Any]] = []
        self.task_index: Dict[str, List[int]] = defaultdict(list)

    def add_task(self, task_description: str, outcome: str, success: bool,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Add completed task to memory.

        Args:
            task_description: Description of the task
            outcome: Result or outcome of the task
            success: Whether the task was successful
            metadata: Additional metadata
        """
        task = {
            'description': task_description,
            'outcome': outcome,
            'success': success,
            'metadata': metadata or {},
            'timestamp': self._get_timestamp()
        }

        self.tasks.append(task)

        # Maintain size limit
        if len(self.tasks) > self.max_tasks:
            removed_task = self.tasks.pop(0)
            # Clean up index (simplified)
            pass

        # Index by keywords (simplified)
        keywords = self._extract_keywords(task_description)
        for keyword in keywords:
            self.task_index[keyword].append(len(self.tasks) - 1)

    def search_similar_tasks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar tasks.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of similar tasks
        """
        keywords = self._extract_keywords(query)
        candidate_indices = set()

        for keyword in keywords:
            candidate_indices.update(self.task_index[keyword])

        candidates = [self.tasks[idx] for idx in candidate_indices]

        # Simple relevance scoring (count matching keywords)
        scored_candidates = []
        for task in candidates:
            score = sum(1 for kw in keywords if kw in task['description'].lower())
            scored_candidates.append((task, score))

        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        return [task for task, score in scored_candidates[:top_k]]

    def get_success_rate(self, task_type: Optional[str] = None) -> float:
        """Get success rate for tasks."""
        relevant_tasks = self.tasks
        if task_type:
            relevant_tasks = [t for t in self.tasks if task_type in t['description'].lower()]

        if not relevant_tasks:
            return 0.0

        successful = sum(1 for t in relevant_tasks if t['success'])
        return successful / len(relevant_tasks)

    def clear(self):
        """Clear task memory."""
        self.tasks.clear()
        self.task_index.clear()

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simplified)."""
        # Simple keyword extraction - split and filter
        words = text.lower().split()
        return [word for word in words if len(word) > 3]

    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()

    def size(self) -> int:
        """Get number of tasks in memory."""
        return len(self.tasks)


class SessionMemory:
    """
    Session memory for temporary storage during agent sessions.

    Stores session-specific information that doesn't need to persist.
    """

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def set(self, key: str, value: Any):
        """Set a session variable."""
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a session variable."""
        return self.data.get(key, default)

    def add_to_history(self, event: str, data: Optional[Dict[str, Any]] = None):
        """Add event to session history."""
        history_entry = {
            'event': event,
            'data': data or {},
            'timestamp': self._get_timestamp()
        }
        self.history.append(history_entry)

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get session history."""
        if limit:
            return self.history[-limit:]
        return self.history

    def clear(self):
        """Clear session memory."""
        self.data.clear()
        self.history.clear()

    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()