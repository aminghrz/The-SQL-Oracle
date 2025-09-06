from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import networkx as nx
import logging
from datetime import datetime
from models import RelationshipEdge # Assuming RelationshipEdge is defined in a parent module

# Initialize logger for the module
logger = logging.getLogger(__name__)

class RelationshipGraph:
    """
    Manages and represents relationships between database tables as a directed graph.
    It uses NetworkX for graph operations and maintains additional metadata for each edge.
    """

    def __init__(self, persistence_manager=None, target_db: str = None):
        self.graph = nx.DiGraph()
        self.edges: Dict[Tuple[str, str], RelationshipEdge] = {}
        self.persistence_manager = persistence_manager
        self.target_db = target_db

        # Load existing edges if persistence is available
        if self.persistence_manager and self.target_db:
            self._load_from_persistence()

    def _load_from_persistence(self):
        """Load edges from persistent storage."""
        edges = self.persistence_manager.get_relationship_edges(self.target_db)
        for edge in edges:
            self.edges[(edge.from_table, edge.to_table)] = edge
            self.graph.add_edge(edge.from_table, edge.to_table, weight=edge.weight)
        
    def initialize_from_metadata(self, foreign_keys: List[Tuple[str, str]]):
        """
        Initializes the graph with explicit foreign key relationships from metadata.

        Args:
            foreign_keys (List[Tuple[str, str]]): A list of tuples, where each tuple
                                                  represents a foreign key relationship (from_table, to_table).
        """
        for from_table, to_table in foreign_keys:
            # Add each foreign key as an edge with a high initial weight and 'explicit_fk' type
            self.add_edge(from_table, to_table, weight=1.0, edge_type='explicit_fk')
        logger.info(f"Initialized relationship graph with {len(foreign_keys)} foreign keys")
        
    def add_edge(self, from_table: str, to_table: str, weight: float, edge_type: str):
        """Add or update an edge in the graph"""
        edge_key = (from_table, to_table)
        
        if edge_key in self.edges:
            # Update existing edge
            self.edges[edge_key].weight = weight
            edge = self.edges[edge_key]
        else:
            # Create new edge - ensure both edge_type and relation_type are set
            edge = RelationshipEdge(
                from_table=from_table,
                to_table=to_table,
                weight=weight,
                edge_type=edge_type,
                relation_type=edge_type  # Set relation_type same as edge_type
            )
            self.edges[edge_key] = edge
        
        # Update NetworkX graph
        self.graph.add_edge(from_table, to_table, weight=weight)
        
        # Save to persistence
        if self.persistence_manager and self.target_db:
            self.persistence_manager.save_relationship_edge(self.target_db, edge)
    
    def add_typed_edge(self, from_table: str, to_table: str, relation_type: str, 
                    weight: float, features: Dict[str, Any] = None):
        """Add a typed edge with features."""
        edge_key = (from_table, to_table, relation_type)
        
        # Create edge with both edge_type and relation_type
        edge = RelationshipEdge(
            from_table=from_table,
            to_table=to_table,
            weight=weight,
            edge_type=relation_type,
            relation_type=relation_type,
            features=features or {}
        )
        
        # Store with composite key including type
        self.edges[edge_key] = edge
        
        # Update NetworkX graph (may have multiple edges between same nodes)
        if not self.graph.has_edge(from_table, to_table):
            self.graph.add_edge(from_table, to_table, weight=weight)
        else:
            # Update weight if higher
            current_weight = self.graph[from_table][to_table]['weight']
            if weight > current_weight:
                self.graph[from_table][to_table]['weight'] = weight
        
        # Save to persistence
        if self.persistence_manager and self.target_db:
            self.persistence_manager.save_relationship_edge(self.target_db, edge)
        
    def add_potential_relationship(self, table1: str, table2: str, evidence: Dict[str, any]):
        """
        Adds a potential relationship between two tables with a very low initial confidence.
        Also stores the evidence for this potential relationship.

        Args:
            table1 (str): The first table in the potential relationship.
            table2 (str): The second table in the potential relationship.
            evidence (Dict[str, any]): A dictionary containing evidence supporting this potential relationship.
        """
        initial_weight = 0.05 # Start with a low confidence for potential relationships
        self.add_edge(table1, table2, weight=initial_weight, edge_type='potential')
        
        # Store the evidence within the RelationshipEdge object
        edge_key = (table1, table2)
        if edge_key in self.edges:
            self.edges[edge_key].usage_contexts.append({
                'evidence': evidence,
                'timestamp': datetime.now()
            })
        
    def get_neighbors(self, table: str) -> Dict[str, List[RelationshipEdge]]:
        """Get all typed edges for a table."""
        neighbors = defaultdict(list)
        
        # Check all edges
        for edge_key, edge in self.edges.items():
            # Handle both 2-tuple and 3-tuple keys
            if isinstance(edge_key, tuple):
                if len(edge_key) == 2:
                    from_table, to_table = edge_key
                elif len(edge_key) == 3:
                    from_table, to_table, rel_type = edge_key
                else:
                    continue
                    
                if from_table == table:
                    neighbors[to_table].append(edge)
                elif to_table == table:
                    neighbors[from_table].append(edge)
        
        return dict(neighbors)
        
    def get_edge_weight(self, table1: str, table2: str) -> float:
        """
        Gets the weight of an edge between two tables, checking both directions.

        Args:
            table1 (str): The name of the first table.
            table2 (str): The name of the second table.

        Returns:
            float: The weight of the edge, or 0.0 if no direct edge exists in either direction.
        """
        edge_key = (table1, table2)
        if edge_key in self.edges:
            return self.edges[edge_key].weight
        
        # Check the reverse direction as well (table2 -> table1)
        edge_key = (table2, table1)
        if edge_key in self.edges:
            return self.edges[edge_key].weight
        
        return 0.0 # Return 0 if no edge found
        
    def update_edge(self, table1: str, table2: str, new_weight: float):
        """Update the weight of an existing edge"""
        edge_key = (table1, table2)
        if edge_key in self.edges:
            self.edges[edge_key].weight = new_weight
            self.graph[table1][table2]['weight'] = new_weight
            # Save to persistence
            if self.persistence_manager and self.target_db:
                self.persistence_manager.save_relationship_edge(self.target_db, self.edges[edge_key])
        
    def increment_usage_count(self, table1: str, table2: str):
        """
        Increments the usage count for a specific relationship edge.
        This can be used to track how often a relationship is leveraged.

        Args:
            table1 (str): The source table of the edge.
            table2 (str): The destination table of the edge.
        """
        edge_key = (table1, table2)
        if edge_key in self.edges:
            self.edges[edge_key].usage_count += 1
        
    def add_usage_context(self, table1: str, table2: str, context: Dict[str, any]):
        """
        Adds additional context to the `usage_contexts` list of a specific edge.
        This can store details about how and when a relationship was used.

        Args:
            table1 (str): The source table of the edge.
            table2 (str): The destination table of the edge.
            context (Dict[str, any]): A dictionary containing details about the usage.
        """
        edge_key = (table1, table2)
        if edge_key in self.edges:
            self.edges[edge_key].usage_contexts.append(context)
        
    def find_path(self, from_table: str, to_table: str, max_hops: int = 3) -> Optional[List[str]]:
        """
        Finds the shortest path between two tables in the graph up to a maximum number of hops.

        Args:
            from_table (str): The starting table for the path search.
            to_table (str): The target table for the path search.
            max_hops (int): The maximum number of edges (hops) allowed in the path. Defaults to 3.

        Returns:
            Optional[List[str]]: A list of table names representing the shortest path, or None if no path is found
                                 within the max_hops limit.
        """
        try:
            path = nx.shortest_path(self.graph, from_table, to_table)
            # Return path only if it's within the max_hops limit (+1 for nodes)
            if len(path) <= max_hops + 1:
                return path
        except nx.NetworkXNoPath:
            pass # No path exists
        return None
        
    def get_table_importance(self, table: str) -> float:
        """
        Calculates the importance of a table within the graph using the PageRank algorithm.
        PageRank assigns a score to each node based on the number and quality of its incoming links.

        Args:
            table (str): The name of the table to assess importance for.

        Returns:
            float: The PageRank score of the table, or 0.0 if the table is not in the graph or an error occurs.
        """
        try:
            pagerank = nx.pagerank(self.graph) # Compute PageRank for all nodes
            return pagerank.get(table, 0.0) # Return the score for the specific table
        except Exception:
            # Handle cases where PageRank computation might fail (e.g., empty graph)
            return 0.0
        
    def prune_low_confidence_edges(self, min_weight: float = 0.1):
        """
        Removes 'potential' type edges from the graph that have a weight below a specified minimum confidence.

        Args:
            min_weight (float): The minimum weight threshold below which 'potential' edges will be removed.
                                Defaults to 0.1.
        """
        edges_to_remove = []
        
        for edge_key, edge in self.edges.items():
            # Identify potential edges with low confidence to remove
            if edge.edge_type == 'potential' and edge.weight < min_weight:
                edges_to_remove.append(edge_key)
        
        for edge_key in edges_to_remove:
            from_table, to_table = edge_key
            self.graph.remove_edge(from_table, to_table) # Remove from NetworkX graph
            del self.edges[edge_key] # Remove from custom edges dictionary
        
        logger.info(f"Pruned {len(edges_to_remove)} low-confidence edges")
        
    def get_statistics(self) -> Dict[str, any]:
        """
        Retrieves various statistics about the current state of the relationship graph.

        Returns:
            Dict[str, any]: A dictionary containing graph statistics such as node count, edge count,
                            breakdown of edge types, average weight, and graph density.
        """
        return {
            'nodes': self.graph.number_of_nodes(),       # Total number of tables (nodes)
            'edges': self.graph.number_of_edges(),       # Total number of relationships (edges)
            'explicit_fks': sum(1 for e in self.edges.values() if e.edge_type == 'explicit_fk'), # Count of explicit FKs
            'potential_relationships': sum(1 for e in self.edges.values() if e.edge_type == 'potential'), # Count of potential relationships
            'avg_weight': sum(e.weight for e in self.edges.values()) / len(self.edges) if self.edges else 0, # Average edge weight
            'density': nx.density(self.graph) # Graph density (ratio of actual edges to possible edges)
        }
    

    def find_shortest_reliable_path(self, from_table: str, to_table: str,
                                    min_weight: float = 0.6, max_hops: int = 3) -> Optional[List[Tuple[str, str, float]]]:
        """Find shortest path using only reliable edges"""
        import heapq
        
        # Dijkstra's algorithm with weight threshold
        distances = {from_table: 0}
        previous = {}
        visited = set()
        
        # Priority queue: (distance, table)
        pq = [(0, from_table)]
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == to_table:
                # Reconstruct path
                path = []
                while current in previous:
                    prev, edge_weight = previous[current]
                    path.append((prev, current, edge_weight))  # Always return 3-tuple
                    current = prev
                path.reverse()
                
                # Check path length
                if len(path) <= max_hops:
                    return path
                else:
                    return None
            
            # Check neighbors
            for neighbor, edges in self.get_neighbors(current).items():
                # Use highest weight edge
                best_edge = max(edges, key=lambda e: e.weight)
                
                # Skip low-weight edges
                if best_edge.weight < min_weight:
                    continue
                    
                # Calculate distance (inverse of weight for shortest path)
                distance = current_dist + (1 - best_edge.weight)
                
                if neighbor not in distances or distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = (current, best_edge.weight)
                    heapq.heappush(pq, (distance, neighbor))
        
        return None
    
    def get_join_template(self, table1: str, table2: str) -> Optional[Dict[str, Any]]:
        """Get join template for two tables"""
        # Check direct edge
        for (from_table, to_table), edge in self.edges.items():
            if (from_table == table1 and to_table == table2) or \
               (from_table == table2 and to_table == table1):
                features = edge.features
                return {
                    'join_type': 'INNER JOIN',
                    'on_clause': f"{from_table}.{features.get('from_column', 'id')} = {to_table}.{features.get('to_column', 'id')}",
                    'confidence': edge.weight,
                    'edge_type': edge.edge_type
                }
        
        # Check junction path
        for (from_table, to_table), edge in self.edges.items():
            if edge.edge_type == 'via_junction':
                if (from_table == table1 and to_table == table2) or \
                   (from_table == table2 and to_table == table1):
                    junction = edge.features.get('junction_table')
                    return {
                        'join_type': 'INNER JOIN',
                        'via_junction': junction,
                        'confidence': edge.weight,
                        'edge_type': edge.edge_type
                    }
        
        return None
    
    def get_production_edges(self, min_weight: float = 0.85) -> List[RelationshipEdge]:
        """Get only production-grade edges"""
        production_edges = []
        
        for edge in self.edges.values():
            if edge.edge_type == 'explicit_fk' or \
               (edge.edge_type == 'fk_like_inclusion' and edge.weight >= min_weight):
                production_edges.append(edge)
        
        return production_edges
    
    def get_edge_explanation(self, from_table: str, to_table: str) -> str:
        """Get human-readable explanation for why tables are joined"""
        edge = self.edges.get((from_table, to_table))
        if not edge:
            edge = self.edges.get((to_table, from_table))
        
        if not edge:
            return "No direct relationship found"
        
        explanations = {
            'explicit_fk': f"Foreign key constraint: {edge.features.get('from_column', 'column')} references {edge.features.get('to_column', 'column')}",
            'fk_like_inclusion': f"Data analysis shows {edge.features.get('inclusion_ratio', 0):.0%} of values in {edge.features.get('from_column')} exist in {edge.features.get('to_column')}",
            'key_equivalent_or_semantic_match': f"Columns {edge.features.get('from_column')} and {edge.features.get('to_column')} are semantically similar (score: {edge.features.get('similarity_score', 0):.2f})",
            'fk_candidate_from_summary': f"Column analysis suggests {edge.features.get('from_column')} references {edge.features.get('to_column')}",
            'fk_from_table_summary': f"Table analysis identified foreign key relationship to {edge.features.get('to_column')}",
            'fk_name_pattern': f"Column naming pattern suggests {edge.features.get('from_column')} references {edge.features.get('to_column')}",
            'via_junction': f"Tables connected through junction table {edge.features.get('junction_table')}",
            'co_usage': f"Tables frequently used together in successful queries (usage count: {edge.features.get('usage_count', 0)})"
            }
         
        return explanations.get(edge.edge_type, f"Relationship type: {edge.edge_type}")