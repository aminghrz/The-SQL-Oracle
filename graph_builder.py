import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from datetime import datetime
import asyncio
from collections import defaultdict
from models import RelationshipEdge, GraphNode, TableInfo
from relationship_graph import RelationshipGraph

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Builds relationship graph using embeddings and summaries"""
    
    def __init__(self,
                 relationship_graph: RelationshipGraph,
                 metadata: Dict[str, TableInfo],
                 table_summaries: Dict[str, Dict[str, Any]],
                 column_summaries: Dict[str, Dict[str, Any]],
                 embedding_service: Any,
                 schema_analyzer: Any,
                 connection: Any):
        self.graph = relationship_graph
        self.metadata = metadata
        self.table_summaries = table_summaries
        self.column_summaries = column_summaries
        self.embedding_service = embedding_service
        self.schema_analyzer = schema_analyzer
        self.connection = connection
        self.nodes: Dict[str, GraphNode] = {}
        self.persistence_manager = relationship_graph.persistence_manager
        self.target_db = relationship_graph.target_db
    
    async def build_graph(self, stages: List[int] = [0, 1]) -> Dict[str, Any]:
        """Build graph in stages"""
        stats = {'start_time': datetime.now()}
        
        # Load existing nodes if available
        if self.persistence_manager and self.target_db:
            existing_nodes = self.persistence_manager.get_graph_nodes(self.target_db)
            if existing_nodes:
                self.nodes = existing_nodes
                logger.info(f"Loaded {len(existing_nodes)} existing graph nodes")
        
        # Always build Stage 0 (minimal viable graph)
        logger.info("Building Stage 0: Minimal viable graph...")
        stage0_stats = await self._build_stage_0()
        stats['stage_0'] = stage0_stats
        
        # Build additional stages if requested
        if 1 in stages:
            logger.info("Building Stage 1: Embedding-based inference...")
            stage1_stats = await self._build_stage_1()
            stats['stage_1'] = stage1_stats
        
        if 2 in stages:
            logger.info("Scheduling Stage 2: Content validation (background)...")
            asyncio.create_task(self._build_stage_2_background())
            stats['stage_2'] = {'status': 'scheduled'}
        
        stats['end_time'] = datetime.now()
        stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()
        return stats
    
    async def _build_stage_0(self) -> Dict[str, Any]:
        """Stage 0: Nodes + Explicit FKs only"""
        stats = {'nodes_created': 0, 'edges_created': 0}

        # Create enriched nodes for ALL tables in metadata
        for table_name, table_info in self.metadata.items():
            # Always create node, even if it exists
            node = self._create_node(table_name, table_info)
            self.nodes[table_name] = node
            # Add node to NetworkX graph
            self.graph.graph.add_node(table_name)
            stats['nodes_created'] += 1

        # Save nodes to persistence
        if self.persistence_manager and self.target_db:
            for table_name, node in self.nodes.items():
                self.persistence_manager.save_graph_node(self.target_db, node)

        logger.info(f"Created {stats['nodes_created']} nodes")

        # Count existing explicit FKs
        for edge in self.graph.edges.values():
            if edge.edge_type == 'explicit_fk':
                edge.weight = 1.0
                edge.features['stage'] = 0
                edge.features['confidence'] = 'explicit'
                stats['edges_created'] += 1

        return stats
    
    def _create_node(self, table_name: str, table_info: TableInfo) -> GraphNode:
        """Create enriched node from table info and summaries"""
        node = GraphNode(table_name=table_name)
        
        # Get table summary if available
        if table_name in self.table_summaries:
            summary = self.table_summaries[table_name]['summary']
            node.purpose = summary.get('purpose')
            node.table_type = summary.get('table_type', 'other')
            node.grain = summary.get('grain')
            node.key_columns = summary.get('key_columns', [])
            node.foreign_keys = summary.get('foreign_keys', [])
            node.measures = summary.get('measures', [])
            node.dimensions = summary.get('dimensions', [])
        
        # Analyze columns for additional metadata
        for col in table_info.columns:
            col_name = col['name']
            col_key = f"{table_name}.{col_name}"
            
            # Check if it's a time column from summary
            if col_key in self.column_summaries:
                col_summary = self.column_summaries[col_key]['summary']
                if col_summary.get('semantic_role') == 'time':
                    node.time_columns.append(col_name)
                    node.has_time = True
        
        # Get row count if available
        try:
            stats = self.schema_analyzer.get_basic_metadata(table_name)
            node.row_count = stats.get('row_count', 0)
        except:
            pass
        
        # Detect junction tables from summary
        if node.table_type == 'junction' or (
            len(node.foreign_keys) >= 2 and
            len(node.measures) == 0 and
            len(table_info.columns) <= 5):
            node.table_type = 'junction'
        
        # Save node to persistence
        if self.persistence_manager and self.target_db:
            self.persistence_manager.save_graph_node(self.target_db, node)
        
        return node
    
    async def _build_stage_1(self) -> Dict[str, Any]:
        """Stage 1: Embedding-based inference"""
        stats = defaultdict(int)
        
        # 1. Summary-based edges (from LLM analysis)
        summary_edges = self._find_summary_based_edges()
        for edge in summary_edges:
            self._add_edge(edge)
            stats['summary_based_edges'] += 1
        
        # 2. Table embedding similarity
        table_similarity_edges = await self._find_table_similarity_edges()
        for edge in table_similarity_edges:
            self._add_edge(edge)
            stats['table_similarity_edges'] += 1
        
        # 3. Column embedding similarity
        column_similarity_edges = await self._find_column_similarity_edges()
        for edge in column_similarity_edges:
            self._add_edge(edge)
            stats['column_similarity_edges'] += 1
        
        # 4. Junction table paths
        junction_edges = self._find_junction_paths()
        for edge in junction_edges:
            self._add_edge(edge)
            stats['junction_edges'] += 1
        
        # Build joinability index
        self._build_joinability_index()
        stats['joinability_index_built'] = True
        
        # Log summary
        logger.info(f"Stage 1 complete: {dict(stats)}")
        logger.info(f"Total edges in graph: {len(self.graph.edges)}")
        
        return dict(stats)
    
    def _find_summary_based_edges(self) -> List[RelationshipEdge]:
        """Extract edges from LLM-generated summaries"""
        edges = []
        
        # From column summaries - join_key_candidates
        for col_key, col_data in self.column_summaries.items():
            table_name, col_name = col_key.split('.', 1)
            summary = col_data['summary']
            
            # Check join_key_candidates
            candidates = summary.get('join_key_candidates', [])
            for candidate in candidates:
                if '.' in candidate:
                    target_table, target_col = candidate.split('.', 1)
                    if target_table in self.metadata:
                        # Calculate weight based on semantic role and confidence
                        role = summary.get('semantic_role', 'other')
                        confidence = summary.get('confidence', 0.5)
                        base_weight = 0.7 if role in ['id', 'code'] else 0.5
                        weight = base_weight * confidence
                        
                        edge = RelationshipEdge(
                            from_table=table_name,
                            to_table=target_table,
                            weight=weight,
                            edge_type='fk_candidate_from_summary',
                            relation_type='fk_candidate_from_summary',
                            features={
                                'from_column': col_name,
                                'to_column': target_col,
                                'semantic_role': role,
                                'source': 'column_summary',
                                'confidence': confidence,
                                'stage': 1
                            }
                        )
                        edges.append(edge)
        
        # From table summaries - foreign_keys
        for table_name, table_data in self.table_summaries.items():
            summary = table_data['summary']
            foreign_keys = summary.get('foreign_keys', [])
            
            for fk in foreign_keys:
                if '.' in fk:
                    parts = fk.split('.')
                    if len(parts) == 2:
                        target_table, target_col = parts
                        if target_table in self.metadata:
                            confidence = summary.get('confidence', 0.5)
                            edge = RelationshipEdge(
                                from_table=table_name,
                                to_table=target_table,
                                weight=0.6 * confidence,
                                edge_type='fk_from_table_summary',
                                relation_type='fk_from_table_summary',
                                features={
                                    'to_column': target_col,
                                    'source': 'table_summary',
                                    'confidence': confidence,
                                    'stage': 1
                                }
                            )
                            edges.append(edge)
        
        logger.info(f"Found {len(edges)} summary-based edges")
        return edges
    
    async def _find_table_similarity_edges(self) -> List[RelationshipEdge]:
        """Find similar tables based on table embedding similarity"""
        edges = []
        processed_pairs = set()
        
        # Compare table embeddings
        table_items = list(self.table_summaries.items())
        max_comparisons = 500
        comparison_count = 0
        
        for i, (table1, table1_data) in enumerate(table_items):
            if comparison_count >= max_comparisons:
                break
            
            for table2, table2_data in table_items[i+1:]:
                if comparison_count >= max_comparisons:
                    break
                
                pair = tuple(sorted([table1, table2]))
                if pair in processed_pairs:
                    continue
                
                comparison_count += 1
                
                try:
                    # Calculate similarity between table embeddings
                    similarity = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.embedding_service.compute_similarity,
                            table1_data['embedding'],
                            table2_data['embedding']
                        ),
                        timeout=1.0
                    )
                    
                    # High similarity suggests related tables
                    if similarity > 0.7:
                        # Check if tables have compatible types
                        node1 = self.nodes.get(table1)
                        node2 = self.nodes.get(table2)
                        
                        if node1 and node2:
                            # Determine relationship type based on table types
                            if node1.table_type == 'fact' and node2.table_type == 'dimension':
                                weight = similarity * 0.8
                                edge_type = 'fact_dimension_similarity'
                            elif node1.table_type == 'dimension' and node2.table_type == 'fact':
                                weight = similarity * 0.8
                                edge_type = 'fact_dimension_similarity'
                            else:
                                weight = similarity * 0.6
                                edge_type = 'table_similarity'
                            
                            edge = RelationshipEdge(
                                from_table=table1,
                                to_table=table2,
                                weight=weight,
                                edge_type=edge_type,
                                relation_type=edge_type,
                                features={
                                    'similarity_score': similarity,
                                    'table1_type': node1.table_type,
                                    'table2_type': node2.table_type,
                                    'stage': 1
                                }
                            )
                            edges.append(edge)
                            processed_pairs.add(pair)
                
                except asyncio.TimeoutError:
                    logger.warning(f"Table similarity computation timed out for {table1} vs {table2}")
                except Exception as e:
                    logger.error(f"Error computing table similarity: {e}")
        
        logger.info(f"Found {len(edges)} table similarity edges from {comparison_count} comparisons")
        return edges
    
    async def _find_column_similarity_edges(self) -> List[RelationshipEdge]:
        """Find edges based on column embedding similarity"""
        edges = []
        processed_pairs = set()
        
        # Group columns by semantic role for more targeted comparison
        columns_by_role = defaultdict(list)
        for col_key, col_data in self.column_summaries.items():
            role = col_data['summary'].get('semantic_role', 'other')
            columns_by_role[role].append((col_key, col_data))
        
        # Compare columns within same semantic role
        for role, columns in columns_by_role.items():
            if role == 'other':
                continue  # Skip generic columns
            
            logger.info(f"Comparing {len(columns)} columns with role '{role}'")
            
            for i, (col1_key, col1_data) in enumerate(columns):
                table1, col1 = col1_key.split('.', 1)
                
                for col2_key, col2_data in columns[i+1:]:
                    table2, col2 = col2_key.split('.', 1)
                    
                    # Skip same table
                    if table1 == table2:
                        continue
                    
                    pair = tuple(sorted([table1, table2]))
                    if pair in processed_pairs:
                        continue
                    
                    try:
                        # Calculate similarity
                        similarity = await asyncio.wait_for(
                            asyncio.to_thread(
                                self.embedding_service.compute_similarity,
                                col1_data['embedding'],
                                col2_data['embedding']
                            ),
                            timeout=1.0
                        )
                        
                        # Different thresholds for different roles
                        threshold = {
                            'id': 0.8,
                            'code': 0.8,
                            'dimension': 0.75,
                            'measure': 0.75,
                            'time': 0.8,
                            'flag': 0.7
                        }.get(role, 0.85)
                        
                        if similarity > threshold:
                            # Get column info for type checking
                            col1_info = self._get_column_info(table1, col1)
                            col2_info = self._get_column_info(table2, col2)
                            
                            if col1_info and col2_info and self._are_types_compatible(
                                col1_info['type'], col2_info['type']):
                                
                                # Weight based on role and similarity
                                weight_multiplier = {
                                    'id': 0.9,
                                    'code': 0.85,
                                    'dimension': 0.7,
                                    'time': 0.75,
                                    'measure': 0.6,
                                    'flag': 0.6
                                }.get(role, 0.5)
                                
                                weight = similarity * weight_multiplier
                                
                                edge = RelationshipEdge(
                                    from_table=table1,
                                    to_table=table2,
                                    weight=weight,
                                    edge_type='column_similarity',
                                    relation_type='column_similarity',
                                    features={
                                        'from_column': col1,
                                        'to_column': col2,
                                        'similarity_score': similarity,
                                        'semantic_role': role,
                                        'type_compatible': True,
                                        'stage': 1
                                    }
                                )
                                edges.append(edge)
                                processed_pairs.add(pair)
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Column similarity timed out for {col1_key} vs {col2_key}")
                    except Exception as e:
                        logger.error(f"Error computing column similarity: {e}")
        
        logger.info(f"Found {len(edges)} column similarity edges")
        return edges
    
    def _find_junction_paths(self) -> List[RelationshipEdge]:
        """Find relationships through junction tables"""
        edges = []
        
        # Find junction tables
        junction_tables = [
            table_name for table_name, node in self.nodes.items()
            if node.table_type == 'junction'
        ]
        
        logger.info(f"Found {len(junction_tables)} junction tables")
        
        for junction in junction_tables:
            junction_node = self.nodes[junction]
            
            # Find tables this junction connects (from foreign_keys in summary)
            connected_tables = []
            for fk in junction_node.foreign_keys:
                if '.' in fk:
                    target_table = fk.split('.')[0]
                    if target_table in self.metadata:
                        connected_tables.append(target_table)
            
            # Create edges between connected tables via junction
            for i, table1 in enumerate(connected_tables):
                for table2 in connected_tables[i+1:]:
                    edge = RelationshipEdge(
                        from_table=table1,
                        to_table=table2,
                        weight=0.7,  # Higher weight for junction relationships
                        edge_type='via_junction',
                        relation_type='via_junction',
                        features={
                            'junction_table': junction,
                            'path': f"{table1}->{junction}->{table2}",
                            'stage': 1
                        }
                    )
                    edges.append(edge)
        
        logger.info(f"Found {len(edges)} junction-based edges")
        return edges
    
    def _add_edge(self, edge: RelationshipEdge):
        """Add edge to graph with deduplication"""
        # Check if edge already exists with higher weight
        existing_key = (edge.from_table, edge.to_table)
        existing = self.graph.edges.get(existing_key)
        
        if existing and existing.weight >= edge.weight:
            return
        
        # Add or update edge
        self.graph.add_typed_edge(
            edge.from_table,
            edge.to_table,
            edge.relation_type,
            edge.weight,
            edge.features
        )
        
        # Log edge addition
        logger.debug(f"Added edge: {edge.from_table} -> {edge.to_table} "
                    f"(type: {edge.edge_type}, weight: {edge.weight:.2f})")
    
    def _get_column_info(self, table_name: str, column_name: str) -> Optional[Dict[str, Any]]:
        """Get column info from metadata"""
        if table_name in self.metadata:
            for col in self.metadata[table_name].columns:
                if col['name'] == column_name:
                    return col
        return None
    
    def _are_types_compatible(self, type1: Any, type2: Any) -> bool:
        """Check if two column types are compatible for joining"""
        # Convert to string for comparison
        type1_str = str(type1).upper()
        type2_str = str(type2).upper()
        
        # Exact match
        if type1_str == type2_str:
            return True
        
        # Define type groups
        numeric_types = ['INTEGER', 'BIGINT', 'SMALLINT', 'NUMERIC', 'DECIMAL',
                        'FLOAT', 'REAL', 'DOUBLE', 'NUMBER', 'INT']
        string_types = ['VARCHAR', 'CHAR', 'TEXT', 'STRING', 'NVARCHAR', 'VARCHAR2', 'CLOB']
        date_types = ['DATE', 'DATETIME', 'TIMESTAMP', 'TIME']
        
        # Check if both are numeric
        is_type1_numeric = any(t in type1_str for t in numeric_types)
        is_type2_numeric = any(t in type2_str for t in numeric_types)
        if is_type1_numeric and is_type2_numeric:
            return True
        
        # Check if both are string
        is_type1_string = any(t in type1_str for t in string_types)
        is_type2_string = any(t in type2_str for t in string_types)
        if is_type1_string and is_type2_string:
            return True
        
        # Check if both are date/time
        is_type1_date = any(t in type1_str for t in date_types)
        is_type2_date = any(t in type2_str for t in date_types)
        if is_type1_date and is_type2_date:
            return True
        
        # Special case: Allow numeric-string compatibility for IDs
        # This is based on the semantic role, not name pattern
        if (is_type1_numeric and is_type2_string) or (is_type1_string and is_type2_numeric):
            return True
        
        return False
    
    def _build_joinability_index(self):
        """Build index of top neighbors for each table"""
        for table_name, node in self.nodes.items():
            # Get all edges for this table
            neighbors = []
            
            # Check all typed edges
            for edge_key, edge in self.graph.edges.items():
                if isinstance(edge_key, tuple) and len(edge_key) >= 2:
                    from_table, to_table = edge_key[:2]
                    if from_table == table_name:
                        neighbors.append({
                            'table': to_table,
                            'direction': 'outgoing',
                            'edge': edge
                        })
                    elif to_table == table_name:
                        neighbors.append({
                            'table': from_table,
                            'direction': 'incoming',
                            'edge': edge
                        })
            
            # Sort by weight and edge type preference
            def edge_priority(n):
                edge = n['edge']
                type_priority = {
                    'explicit_fk': 0,
                    'fk_like_inclusion': 1,
                    'column_similarity': 2,
                    'fk_candidate_from_summary': 3,
                    'fk_from_table_summary': 3,
                    'table_similarity': 4,
                    'fact_dimension_similarity': 4,
                    'via_junction': 5,
                    'co_usage': 6
                }
                return (-edge.weight, type_priority.get(edge.edge_type, 10))
            
            neighbors.sort(key=edge_priority)
            
            # Store top 10 neighbors
            node.default_joins = {
                n['table']: {
                    'weight': n['edge'].weight,
                    'type': n['edge'].edge_type,
                    'direction': n['direction'],
                    'join_columns': self._get_join_columns(n['edge'])
                }
                for n in neighbors[:10]
            }
    
    def _get_join_columns(self, edge: RelationshipEdge) -> Dict[str, str]:
        """Extract join column information from edge"""
        features = edge.features
        return {
            'from_column': features.get('from_column', ''),
            'to_column': features.get('to_column', ''),
            'junction': features.get('junction_table', '')
        }
    
    async def _build_stage_2_background(self):
        """Stage 2: Content-based validation (runs in background)"""
        logger.info("Starting Stage 2 background processing...")
        
        try:
            # Get high-confidence column pairs from Stage 1
            candidate_pairs = []
            for edge_key, edge in self.graph.edges.items():
                if edge.edge_type in ['column_similarity', 'fk_candidate_from_summary']:
                    if edge.weight > 0.7:
                        from_col = edge.features.get('from_column')
                        to_col = edge.features.get('to_column')
                        if from_col and to_col:
                            candidate_pairs.append({
                                'from_table': edge.from_table,
                                'from_column': from_col,
                                'to_table': edge.to_table,
                                'to_column': to_col,
                                'current_weight': edge.weight
                            })
            
            logger.info(f"Testing {len(candidate_pairs)} high-confidence pairs for inclusion")
            
            # Test inclusion for candidate pairs
            for pair in candidate_pairs[:50]:  # Limit to prevent long running
                try:
                    inclusion = await asyncio.to_thread(
                        self.schema_analyzer.estimate_inclusion,
                        pair['from_table'], pair['from_column'],
                        pair['to_table'], pair['to_column'],
                        sample_size=200
                    )
                    
                    if inclusion > 0.8:
                        # Create or update edge with higher weight
                        new_weight = 0.85 + (inclusion - 0.8) * 0.5  # 0.85-0.95 range
                        
                        edge = RelationshipEdge(
                            from_table=pair['from_table'],
                            to_table=pair['to_table'],
                            weight=new_weight,
                            edge_type='fk_like_inclusion',
                            relation_type='fk_like_inclusion',
                            features={
                                'from_column': pair['from_column'],
                                'to_column': pair['to_column'],
                                'inclusion_ratio': inclusion,
                                'sample_size': 200,
                                'stage': 2,
                                'validated_at': datetime.now().isoformat(),
                                'previous_weight': pair['current_weight']
                            }
                        )
                        self._add_edge(edge)
                        
                        logger.info(f"Validated inclusion: {pair['from_table']}.{pair['from_column']} -> "
                                  f"{pair['to_table']}.{pair['to_column']} (inclusion: {inclusion:.2f})")
                
                except Exception as e:
                    logger.debug(f"Inclusion test failed for {pair}: {e}")
            
            logger.info("Stage 2 background processing completed")
            
        except Exception as e:
            logger.error(f"Stage 2 processing failed: {e}")