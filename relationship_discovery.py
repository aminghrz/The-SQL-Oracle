import asyncio
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class RelationshipDiscovery:
    def __init__(self, metadata, table_summaries, column_summaries, embedding_service,
                 metadata_extractor, schema_analyzer, connection):
        self.metadata = metadata
        self.table_summaries = table_summaries
        self.column_summaries = column_summaries
        self.embedding_service = embedding_service
        self.metadata_extractor = metadata_extractor
        self.schema_analyzer = schema_analyzer
        self.connection = connection
        
    async def discover_all_relationships(self) -> List[Dict[str, Any]]:
        """Discover relationships using all methods"""
        relationships = []
        
        # 1. Foreign key relationships (weight 1.0)
        fk_rels = await self._discover_foreign_keys()
        relationships.extend(fk_rels)
        logger.info(f"Found {len(fk_rels)} foreign key relationships")
        
        # 2. Column similarity relationships (weight varies)
        similarity_rels = await self._discover_by_column_similarity()
        relationships.extend(similarity_rels)
        logger.info(f"Found {len(similarity_rels)} column similarity relationships")
        
        # 3. Content similarity relationships (weight varies)
        content_rels = await self._discover_by_content_similarity()
        relationships.extend(content_rels)
        logger.info(f"Found {len(content_rels)} content similarity relationships")
        
        # Merge duplicate relationships, keeping highest weight
        merged = self._merge_relationships(relationships)
        return merged
    
    async def _discover_foreign_keys(self) -> List[Dict[str, Any]]:
        """Extract explicit foreign keys"""
        relationships = []
        
        # Use existing method
        foreign_keys = await asyncio.to_thread(
            self.metadata_extractor.extract_foreign_keys
        )
        
        for from_table, to_table in foreign_keys:
            relationships.append({
                'from_table': from_table,
                'to_table': to_table,
                'type': 'explicit_fk',
                'weight': 1.0,
                'features': {
                    'source': 'database_metadata',
                    'confidence': 1.0
                }
            })
        
        return relationships
    
    async def _discover_by_column_similarity(self) -> List[Dict[str, Any]]:
        """Find relationships based on column embedding similarity"""
        relationships = []
        
        # Get column embeddings from column_summaries only
        column_embeddings = {}
        for col_key, data in self.column_summaries.items():
            column_embeddings[col_key] = data['embedding']
        
        # Compare columns across different tables
        processed_pairs = set()
        
        for col1_key, col1_embedding in column_embeddings.items():
            table1, col1 = col1_key.split('.', 1)
            
            # Find similar columns in other tables
            for col2_key, col2_embedding in column_embeddings.items():
                table2, col2 = col2_key.split('.', 1)
                
                # Skip same table or already processed
                if table1 == table2:
                    continue
                
                pair = tuple(sorted([table1, table2]))
                if pair in processed_pairs:
                    continue
                
                # Calculate similarity
                similarity = self.embedding_service.compute_similarity(
                    col1_embedding, col2_embedding
                )
                
                # Check if columns suggest a relationship
                if similarity > 0.85:
                    # Get summaries from column_summaries
                    col1_summary = self.column_summaries[col1_key]['summary']
                    col2_summary = self.column_summaries[col2_key]['summary']
                    
                    is_join_candidate = (
                        col1_summary.get('semantic_role') in ['id', 'code'] or
                        col2_summary.get('semantic_role') in ['id', 'code'] or
                        col1.lower().endswith('_id') or col2.lower().endswith('_id') or
                        col1.lower() == 'id' or col2.lower() == 'id'
                    )
                    
                    if is_join_candidate:
                        relationships.append({
                            'from_table': table1,
                            'to_table': table2,
                            'type': 'column_similarity',
                            'weight': similarity * 0.8,  # Scale down from 1.0
                            'features': {
                                'similar_columns': f"{col1} <-> {col2}",
                                'similarity_score': similarity,
                                'col1_role': col1_summary.get('semantic_role'),
                                'col2_role': col2_summary.get('semantic_role')
                            }
                        })
                        processed_pairs.add(pair)
        
        return relationships
    
    async def _discover_by_content_similarity(self) -> List[Dict[str, Any]]:
        """Find relationships based on actual data content"""
        relationships = []
        tables = list(self.metadata.keys())
        
        # Limit to reasonable number of comparisons
        max_comparisons = 50
        comparison_count = 0
        
        for i, table1 in enumerate(tables):
            if comparison_count >= max_comparisons:
                break
                
            # Get potential join columns for table1
            table1_cols = self._get_join_candidate_columns(table1)
            
            for table2 in tables[i+1:]:
                if comparison_count >= max_comparisons:
                    break
                    
                # Get potential join columns for table2
                table2_cols = self._get_join_candidate_columns(table2)
                
                # Test each column pair
                for col1 in table1_cols:
                    for col2 in table2_cols:
                        comparison_count += 1
                        
                        # Use existing estimate_inclusion method
                        try:
                            inclusion = await asyncio.to_thread(
                                self.schema_analyzer.estimate_inclusion,
                                table1, col1, table2, col2, 100
                            )
                            
                            if inclusion > 0.8:
                                relationships.append({
                                    'from_table': table1,
                                    'to_table': table2,
                                    'type': 'content_inclusion',
                                    'weight': inclusion * 0.7,
                                    'features': {
                                        'join_columns': f"{col1} -> {col2}",
                                        'inclusion_ratio': inclusion,
                                        'sample_size': 100
                                    }
                                })
                                break  # Found one good relationship, skip other columns
                                
                        except Exception as e:
                            logger.debug(f"Content comparison failed for {table1}.{col1} -> {table2}.{col2}: {e}")
        
        return relationships
    
    def _get_join_candidate_columns(self, table_name: str) -> List[str]:
        """Get columns that might be used for joins"""
        candidates = []
        
        if table_name not in self.metadata:
            return candidates
        
        for col in self.metadata[table_name].columns[:10]:  # Limit to first 10
            col_name = col['name']
            col_key = f"{table_name}.{col_name}"
            
            # Check if it's a likely join column
            is_candidate = (
                col_name.lower().endswith('_id') or
                col_name.lower() in ['id', 'code', 'key']
            )
            
            # Also check from summary if available
            if col_key in self.column_summaries:
                summary = self.column_summaries[col_key]['summary']
                if summary.get('semantic_role') in ['id', 'code']:
                    is_candidate = True
            
            if is_candidate:
                candidates.append(col_name)
        
        return candidates
    
    def _merge_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge duplicate relationships, keeping highest weight"""
        merged = {}
        
        for rel in relationships:
            # Create key for both directions
            key1 = (rel['from_table'], rel['to_table'])
            key2 = (rel['to_table'], rel['from_table'])
            
            # Check if we already have this relationship
            existing_key = None
            if key1 in merged:
                existing_key = key1
            elif key2 in merged:
                existing_key = key2
            
            if existing_key:
                # Keep the one with higher weight
                if rel['weight'] > merged[existing_key]['weight']:
                    merged[existing_key] = rel
            else:
                merged[key1] = rel
        
        return list(merged.values())