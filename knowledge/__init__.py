"""
Aisupea Knowledge Module

Knowledge representation, storage, and retrieval system.
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import json
import time

class KnowledgeGraph:
    """Graph-based knowledge representation."""

    def __init__(self):
        self.nodes = {}  # node_id -> node_data
        self.edges = defaultdict(list)  # node_id -> [(target_id, relation, weight), ...]
        self.node_counter = 0

    def add_node(self, node_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a node to the knowledge graph."""
        node_id = f"{node_type}_{self.node_counter}"
        self.node_counter += 1

        self.nodes[node_id] = {
            'type': node_type,
            'content': content,
            'metadata': metadata or {},
            'created': time.time(),
            'last_accessed': time.time()
        }

        return node_id

    def add_edge(self, source_id: str, target_id: str, relation: str, weight: float = 1.0):
        """Add a relationship between nodes."""
        if source_id in self.nodes and target_id in self.nodes:
            self.edges[source_id].append((target_id, relation, weight))
            # Add reverse edge for bidirectional traversal
            self.edges[target_id].append((source_id, f"reverse_{relation}", weight))

    def query_relevant(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find nodes relevant to a query."""
        relevant = []

        for node_id, node_data in self.nodes.items():
            if isinstance(node_data['content'], str):
                if query.lower() in node_data['content'].lower():
                    relevance_score = self._calculate_relevance(query, node_data['content'])
                    relevant.append({
                        'node_id': node_id,
                        'content': node_data['content'],
                        'type': node_data['type'],
                        'relevance': relevance_score
                    })

        # Sort by relevance and return top results
        relevant.sort(key=lambda x: x['relevance'], reverse=True)
        return relevant[:limit]

    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        overlap = len(query_words & content_words)
        union = len(query_words | content_words)

        return overlap / union if union > 0 else 0.0

    def get_related_nodes(self, node_id: str, relation: Optional[str] = None) -> List[Tuple[str, str, float]]:
        """Get nodes related to a given node."""
        if node_id not in self.edges:
            return []

        related = []
        for target_id, rel, weight in self.edges[node_id]:
            if relation is None or rel == relation:
                related.append((target_id, rel, weight))

        return related

    def serialize(self) -> str:
        """Serialize the knowledge graph to JSON."""
        return json.dumps({
            'nodes': self.nodes,
            'edges': dict(self.edges),
            'node_counter': self.node_counter
        })

    def deserialize(self, data: str):
        """Deserialize the knowledge graph from JSON."""
        graph_data = json.loads(data)
        self.nodes = graph_data['nodes']
        self.edges = defaultdict(list, graph_data['edges'])
        self.node_counter = graph_data['node_counter']

class KnowledgeBase:
    """Main knowledge management system."""

    def __init__(self):
        self.graph = KnowledgeGraph()
        self.categories = defaultdict(list)  # category -> [node_ids]
        self.index = {}  # term -> [node_ids]

    def store_fact(self, fact: str, category: str = "general", metadata: Optional[Dict[str, Any]] = None):
        """Store a factual statement."""
        node_id = self.graph.add_node('fact', fact, metadata)
        self.categories[category].append(node_id)
        self._index_content(fact, node_id)

    def store_concept(self, concept: str, definition: str, category: str = "concepts"):
        """Store a concept with its definition."""
        node_id = self.graph.add_node('concept', {
            'name': concept,
            'definition': definition
        }, {'category': category})

        self.categories[category].append(node_id)
        self._index_content(f"{concept} {definition}", node_id)

    def store_relationship(self, entity1: str, relation: str, entity2: str):
        """Store a relationship between entities."""
        # Find or create nodes for entities
        node1_id = self._find_or_create_entity(entity1)
        node2_id = self._find_or_create_entity(entity2)

        # Add relationship edge
        self.graph.add_edge(node1_id, node2_id, relation)

    def _find_or_create_entity(self, entity: str) -> str:
        """Find existing entity node or create new one."""
        # Simple search - in practice would use more sophisticated matching
        for node_id, node_data in self.graph.nodes.items():
            if node_data['type'] == 'entity' and node_data['content'] == entity:
                return node_id

        # Create new entity node
        return self.graph.add_node('entity', entity)

    def _index_content(self, content: str, node_id: str):
        """Index content for faster retrieval."""
        words = content.lower().split()
        for word in words:
            if word not in self.index:
                self.index[word] = []
            if node_id not in self.index[word]:
                self.index[word].append(node_id)

    def query(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query the knowledge base."""
        results = self.graph.query_relevant(query)

        # Filter by category if specified
        if category:
            results = [r for r in results if r['node_id'] in self.categories.get(category, [])]

        return results

    def get_concept(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """Get a concept by name."""
        for node_id, node_data in self.graph.nodes.items():
            if (node_data['type'] == 'concept' and
                isinstance(node_data['content'], dict) and
                node_data['content'].get('name') == concept_name):
                return node_data['content']
        return None

    def get_related_concepts(self, concept_name: str) -> List[str]:
        """Get concepts related to a given concept."""
        related = []

        # Find the concept node
        concept_node = None
        for node_id, node_data in self.graph.nodes.items():
            if (node_data['type'] == 'concept' and
                isinstance(node_data['content'], dict) and
                node_data['content'].get('name') == concept_name):
                concept_node = node_id
                break

        if concept_node:
            related_nodes = self.graph.get_related_nodes(concept_node)
            for target_id, relation, weight in related_nodes:
                target_data = self.graph.nodes.get(target_id)
                if target_data and target_data['type'] == 'concept':
                    related.append(target_data['content'].get('name', ''))

        return related

    def learn_from_text(self, text: str, source: str = "unknown"):
        """Extract and learn knowledge from text."""
        # Simple extraction - in practice would use NLP
        sentences = text.split('.')

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                self.store_fact(sentence, "learned", {
                    'source': source,
                    'learned_at': time.time()
                })

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            'total_nodes': len(self.graph.nodes),
            'total_edges': sum(len(edges) for edges in self.graph.edges.values()),
            'categories': dict(self.categories),
            'index_size': len(self.index)
        }

    def save(self, filename: str):
        """Save knowledge base to file."""
        data = {
            'graph': self.graph.serialize(),
            'categories': dict(self.categories),
            'index': self.index
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filename: str):
        """Load knowledge base from file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.graph.deserialize(data['graph'])
            self.categories = defaultdict(list, data['categories'])
            self.index = data['index']
        except FileNotFoundError:
            print(f"Knowledge file {filename} not found")

__all__ = ['KnowledgeGraph', 'KnowledgeBase']