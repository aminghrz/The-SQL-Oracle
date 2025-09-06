"""Persistence layer using SqliteVecStore for all agent data."""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from sqlite_vec_store import SqliteVecStore
from models import QueryMemory, TableInfo, RelationshipEdge, GraphNode

logger = logging.getLogger(__name__)

class PersistenceManager:
    """Manages all persistence operations for the SQL agent."""
    
    def __init__(self, db_path: str = "the_sql_oracle.sqlite3"):
        self.db_path = db_path
        self.vec_store = SqliteVecStore(db_file=db_path)
        self._init_tables()
        
    def _init_tables(self):
        """Initialize additional tables for non-vector data."""
        conn = self.vec_store._connection
        
        # Query memory table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_memory (
                id TEXT PRIMARY KEY,
                target_db TEXT NOT NULL,
                user_prompt TEXT NOT NULL,
                intent_summary TEXT,
                sql_query TEXT,
                tables_used TEXT,
                result_summary TEXT,
                success_score REAL,
                timestamp TEXT,
                execution_time REAL,
                retry_count INTEGER,
                validation_checks TEXT,
                query_complexity TEXT
            )
        """)
        
        # Insights history table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS insights_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_db TEXT NOT NULL,
                table_name TEXT NOT NULL,
                column_name TEXT,
                insight TEXT,
                confidence_score REAL,
                timestamp TEXT,
                query_context TEXT,
                success_flag INTEGER
            )
        """)
        
        # Relationship graph edges table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS relationship_edges (
                target_db TEXT NOT NULL,
                from_table TEXT NOT NULL,
                to_table TEXT NOT NULL,
                weight REAL,
                edge_type TEXT,
                relation_type TEXT,
                features TEXT,
                usage_count INTEGER,
                usage_contexts TEXT,
                PRIMARY KEY (target_db, from_table, to_table, edge_type)
            )
        """)
        
        conn.execute("""
        CREATE TABLE IF NOT EXISTS table_summaries (
            target_db TEXT NOT NULL,
            table_name TEXT NOT NULL,
            summary_json TEXT NOT NULL,
            summary_embedding TEXT,
            last_updated TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.8,
            PRIMARY KEY (target_db, table_name)
        )
        """)
        
        # Column summaries
        conn.execute("""
            CREATE TABLE IF NOT EXISTS column_summaries (
                target_db TEXT NOT NULL,
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                summary_embedding TEXT,
                last_updated TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.8,
                PRIMARY KEY (target_db, table_name, column_name)
            )
        """)

        # Table metadata cache
        conn.execute("""
            CREATE TABLE IF NOT EXISTS table_metadata (
                target_db TEXT NOT NULL,
                table_name TEXT NOT NULL,
                schema_name TEXT,
                columns TEXT,
                ddl TEXT,
                last_analyzed TEXT,
                analysis_status TEXT,
                statistics TEXT,
                patterns TEXT,
                confidence_score REAL,
                PRIMARY KEY (target_db, table_name)
            )
        """)
        
        # Sample data cache metadata (actual samples stored in vec_store)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sample_cache_meta (
                target_db TEXT NOT NULL,
                table_name TEXT NOT NULL,
                last_updated TEXT,
                ttl_hours INTEGER,
                hits INTEGER,
                size_mb REAL,
                PRIMARY KEY (target_db, table_name)
            )
        """)
        
        # Agent state table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_state (
                target_db TEXT PRIMARY KEY,
                metadata_extracted INTEGER DEFAULT 0,
                last_initialized TEXT,
                total_queries INTEGER DEFAULT 0,
                successful_queries INTEGER DEFAULT 0
            )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_nodes (
        target_db TEXT NOT NULL,
        table_name TEXT NOT NULL,
        purpose TEXT,
        table_type TEXT,
        grain TEXT,
        key_columns TEXT,
        foreign_keys TEXT,
        measures TEXT,
        dimensions TEXT,
        time_columns TEXT,
        row_count INTEGER,
        has_time INTEGER,
        default_joins TEXT,
        last_updated TEXT,
        PRIMARY KEY (target_db, table_name)
        )
        """)
        
        conn.commit()
    
    def get_target_db_namespace(self, target_db: str) -> tuple:
        """Get namespace for a target database."""
        return ("agent", target_db)
    
    def save_query_memory(self, target_db: str, query_memory: QueryMemory):
        """Save query memory to database."""
        conn = self.vec_store._connection
        
        # Save to relational table
        conn.execute("""
            INSERT OR REPLACE INTO query_memory 
            (id, target_db, user_prompt, intent_summary, sql_query, tables_used,
             result_summary, success_score, timestamp, execution_time, retry_count,
             validation_checks, query_complexity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            query_memory.id,
            target_db,
            query_memory.user_prompt,
            json.dumps(query_memory.intent_summary),
            query_memory.sql_query,
            json.dumps(query_memory.tables_used),
            query_memory.result_summary,
            query_memory.success_score,
            query_memory.timestamp.isoformat(),
            query_memory.execution_time,
            query_memory.retry_count,
            json.dumps(query_memory.validation_checks),
            query_memory.query_complexity.value
        ))
        
        # Save embedding to vec_store
        namespace = self.get_target_db_namespace(target_db)
        self.vec_store.put(
            namespace + ("query_embeddings",),
            query_memory.id,
            {
                "prompt": query_memory.user_prompt,
                "embedding": query_memory.prompt_embedding,
                "timestamp": query_memory.timestamp.isoformat()
            }
        )
        
        conn.commit()
    
    def get_query_history(self, target_db: str) -> List[QueryMemory]:
        """Get query history for a target database."""
        conn = self.vec_store._connection
        cursor = conn.execute(
            "SELECT * FROM query_memory WHERE target_db = ? ORDER BY timestamp DESC",
            (target_db,)
        )
        
        queries = []
        for row in cursor:
            # Get embedding from vec_store
            namespace = self.get_target_db_namespace(target_db)
            embedding_item = self.vec_store.get(
                namespace + ("query_embeddings",),
                row['id']
            )
            
            query_memory = QueryMemory(
                id=row['id'],
                user_prompt=row['user_prompt'],
                prompt_embedding=embedding_item.value['embedding'] if embedding_item else [],
                intent_summary=json.loads(row['intent_summary']),
                sql_query=row['sql_query'],
                tables_used=json.loads(row['tables_used']),
                result_summary=row['result_summary'],
                success_score=row['success_score'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                execution_time=row['execution_time'],
                retry_count=row['retry_count'],
                validation_checks=json.loads(row['validation_checks']),
                query_complexity=row['query_complexity']
            )
            queries.append(query_memory)
        
        return queries
    
    def save_table_metadata(self, target_db: str, table_name: str, table_info: TableInfo):
        """Save table metadata."""
        conn = self.vec_store._connection
        
        # Convert columns to JSON-serializable format
        columns_data = []
        for col in table_info.columns:
            col_dict = {
                'name': col.get('name'),
                'type': str(col.get('type')),  # Convert SQLAlchemy type to string
                'nullable': col.get('nullable', True),
                'default': str(col.get('default')) if col.get('default') else None,
                'autoincrement': col.get('autoincrement', False)
            }
            columns_data.append(col_dict)

        conn.execute("""
            INSERT OR REPLACE INTO table_metadata
            (target_db, table_name, schema_name, columns, ddl, last_analyzed,
            analysis_status, statistics, patterns, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            target_db,
            table_name,
            table_info.schema,
            json.dumps(columns_data),
            table_info.ddl,
            table_info.last_analyzed.isoformat() if table_info.last_analyzed else None,
            "analyzed",
            json.dumps({}),  # Statistics can be added later
            json.dumps({}),  # Patterns can be added later
            table_info.confidence_score
        ))
        conn.commit()
    
    def get_table_metadata(self, target_db: str) -> Dict[str, TableInfo]:
        """Get all table metadata for a target database."""
        conn = self.vec_store._connection
        cursor = conn.execute(
            "SELECT * FROM table_metadata WHERE target_db = ?",
            (target_db,)
        )
        
        metadata = {}
        for row in cursor:
            table_info = TableInfo(
                name=row['table_name'],
                schema=row['schema_name'],
                columns=json.loads(row['columns']),
                ddl=row['ddl'],
                last_analyzed=datetime.fromisoformat(row['last_analyzed']) if row['last_analyzed'] else None,
                confidence_score=row['confidence_score']
            )
            metadata[row['table_name']] = table_info
        
        return metadata
    
    def save_schema_summary(self, target_db: str, table_name: str, column_name: Optional[str],
                        summary_json: Dict[str, Any], embedding: List[float], confidence: float = 0.8):
        """Save schema summary."""
        if not summary_json:
            raise ValueError("Cannot save NULL or empty summary_json")
        
        conn = self.vec_store._connection
        
        # Ensure column_name is None for table summaries, not empty string
        if column_name == '':
            column_name = None
        
        conn.execute("""
            INSERT OR REPLACE INTO schema_summaries
            (target_db, table_name, column_name, summary_json, summary_embedding, last_updated, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            target_db,
            table_name,
            column_name,
            json.dumps(summary_json),
            json.dumps(embedding),
            datetime.now().isoformat(),
            confidence
        ))
        conn.commit()
    
    def save_relationship_edge(self, target_db: str, edge: RelationshipEdge):
        """Save relationship edge."""
        conn = self.vec_store._connection
        conn.execute("""
            INSERT OR REPLACE INTO relationship_edges
            (target_db, from_table, to_table, weight, edge_type, relation_type, features, usage_count, usage_contexts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            target_db,
            edge.from_table,
            edge.to_table,
            edge.weight,
            edge.edge_type,
            edge.relation_type,
            json.dumps(edge.features),
            edge.usage_count,
            json.dumps(edge.usage_contexts)
        ))
        conn.commit()
    
    def get_relationship_edges(self, target_db: str) -> List[RelationshipEdge]:
        """Get all relationship edges for a target database."""
        conn = self.vec_store._connection
        cursor = conn.execute(
            "SELECT * FROM relationship_edges WHERE target_db = ?",
            (target_db,)
        )
        
        edges = []
        for row in cursor:
            edge = RelationshipEdge(
                from_table=row['from_table'],
                to_table=row['to_table'],
                weight=row['weight'],
                edge_type=row['edge_type'],
                relation_type=row['relation_type'] or row['edge_type'],
                features=json.loads(row['features'] or '{}'),
                usage_count=row['usage_count'],
                usage_contexts=json.loads(row['usage_contexts'])
            )
            edges.append(edge)
        return edges
    
    def save_sample_data(self, target_db: str, table_name: str, sample_data: Dict[str, Any]):
        """Save sample data to vec_store."""
        namespace = self.get_target_db_namespace(target_db)
        self.vec_store.put(
            namespace + ("sample_data",),
            table_name,
            sample_data
        )
        
        # Update metadata
        conn = self.vec_store._connection
        conn.execute("""
            INSERT OR REPLACE INTO sample_cache_meta
            (target_db, table_name, last_updated, ttl_hours, hits, size_mb)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            target_db,
            table_name,
            datetime.now().isoformat(),
            24,  # Default TTL
            1,   # Initial hit count
            1.0  # Estimated size
        ))
        conn.commit()
    
    def get_sample_data(self, target_db: str, table_name: str) -> Optional[Dict[str, Any]]:
        """Get sample data from vec_store."""
        namespace = self.get_target_db_namespace(target_db)
        item = self.vec_store.get(
            namespace + ("sample_data",),
            table_name
        )
        
        if item:
            # Update hit count
            conn = self.vec_store._connection
            conn.execute("""
                UPDATE sample_cache_meta
                SET hits = hits + 1
                WHERE target_db = ? AND table_name = ?
            """, (target_db, table_name))
            conn.commit()
            
            return item.value
        
        return None
    
    def get_agent_state(self, target_db: str) -> Dict[str, Any]:
        """Get agent state for a target database."""
        conn = self.vec_store._connection
        cursor = conn.execute(
            "SELECT * FROM agent_state WHERE target_db = ?",
            (target_db,)
        )
        row = cursor.fetchone()
        
        if row:
            return {
                'metadata_extracted': bool(row['metadata_extracted']),
                'last_initialized': row['last_initialized'],
                'total_queries': row['total_queries'],
                'successful_queries': row['successful_queries']
            }
        
        return {
            'metadata_extracted': False,
            'last_initialized': None,
            'total_queries': 0,
            'successful_queries': 0
        }
    
    def update_agent_state(self, target_db: str, **kwargs):
        """Update agent state."""
        conn = self.vec_store._connection
        
        # Ensure record exists
        conn.execute("""
            INSERT OR IGNORE INTO agent_state (target_db)
            VALUES (?)
        """, (target_db,))
        
        # Update fields
        for key, value in kwargs.items():
            if key in ['metadata_extracted', 'last_initialized', 
                    'total_queries', 'successful_queries']:
                conn.execute(f"""
                    UPDATE agent_state
                    SET {key} = ?
                    WHERE target_db = ?
                """, (value, target_db))
        
        conn.commit()
    
    def save_table_summary(self, target_db: str, table_name: str, 
                        summary_json: Dict[str, Any], embedding: List[float], 
                        confidence: float = 0.8):
        """Save table summary."""
        if not summary_json:
            raise ValueError("Cannot save NULL or empty summary_json")
        
        conn = self.vec_store._connection
        conn.execute("""
            INSERT OR REPLACE INTO table_summaries
            (target_db, table_name, summary_json, summary_embedding, last_updated, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            target_db,
            table_name,
            json.dumps(summary_json),
            json.dumps(embedding),
            datetime.now().isoformat(),
            confidence
        ))
        conn.commit()
    
    def save_column_summary(self, target_db: str, table_name: str, column_name: str,
                        summary_json: Dict[str, Any], embedding: List[float], 
                        confidence: float = 0.8):
        """Save column summary."""
        if not summary_json:
            raise ValueError("Cannot save NULL or empty summary_json")
        
        conn = self.vec_store._connection
        conn.execute("""
            INSERT OR REPLACE INTO column_summaries
            (target_db, table_name, column_name, summary_json, summary_embedding, last_updated, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            target_db,
            table_name,
            column_name,
            json.dumps(summary_json),
            json.dumps(embedding),
            datetime.now().isoformat(),
            confidence
        ))
        conn.commit()
    
    def get_table_summaries(self, target_db: str) -> Dict[str, Dict[str, Any]]:
        """Get all table summaries. Returns dict with table_name as key."""
        conn = self.vec_store._connection
        cursor = conn.execute(
            "SELECT * FROM table_summaries WHERE target_db = ?",
            (target_db,)
        )
        
        summaries = {}
        for row in cursor:
            try:
                # Handle both dict-like and tuple results
                if hasattr(row, 'keys'):
                    # Row is dict-like (Row object)
                    table_name = row['table_name']
                    summary_json = row['summary_json']
                    summary_embedding = row['summary_embedding']
                    confidence = row['confidence']
                    last_updated = row['last_updated']
                else:
                    # Row is a tuple - need to map by column order
                    # Assuming column order: target_db, table_name, summary_json, summary_embedding, last_updated, confidence
                    table_name = row[1]
                    summary_json = row[2]
                    summary_embedding = row[3]
                    last_updated = row[4]
                    confidence = row[5] if len(row) > 5 else 0.8
                
                summaries[table_name] = {
                    'summary': json.loads(summary_json),
                    'embedding': json.loads(summary_embedding) if summary_embedding else [],
                    'confidence': confidence,
                    'last_updated': last_updated
                }
            except (json.JSONDecodeError, IndexError) as e:
                logger.error(f"Failed to parse table summary for {table_name if 'table_name' in locals() else 'unknown'}: {e}")
                continue
        
        return summaries

    def get_column_summaries(self, target_db: str) -> Dict[str, Dict[str, Any]]:
        """Get all column summaries. Returns dict with 'table.column' as key."""
        conn = self.vec_store._connection
        cursor = conn.execute(
            "SELECT * FROM column_summaries WHERE target_db = ?",
            (target_db,)
        )
        
        summaries = {}
        for row in cursor:
            try:
                # Handle both dict-like and tuple results
                if hasattr(row, 'keys'):
                    # Row is dict-like (Row object)
                    table_name = row['table_name']
                    column_name = row['column_name']
                    summary_json = row['summary_json']
                    summary_embedding = row['summary_embedding']
                    confidence = row['confidence']
                    last_updated = row['last_updated']
                else:
                    # Row is a tuple - need to map by column order
                    # Assuming column order: target_db, table_name, column_name, summary_json, summary_embedding, last_updated, confidence
                    table_name = row[1]
                    column_name = row[2]
                    summary_json = row[3]
                    summary_embedding = row[4]
                    last_updated = row[5]
                    confidence = row[6] if len(row) > 6 else 0.8
                
                key = f"{table_name}.{column_name}"
                summaries[key] = {
                    'summary': json.loads(summary_json),
                    'embedding': json.loads(summary_embedding) if summary_embedding else [],
                    'confidence': confidence,
                    'last_updated': last_updated
                }
            except (json.JSONDecodeError, IndexError) as e:
                logger.error(f"Failed to parse column summary: {e}")
                continue
        
        return summaries
    
    def save_graph_node(self, target_db: str, node: GraphNode):
        """Save graph node information"""
        conn = self.vec_store._connection
        conn.execute("""
            INSERT OR REPLACE INTO graph_nodes
            (target_db, table_name, purpose, table_type, grain, key_columns,
            foreign_keys, measures, dimensions, time_columns, row_count,
            has_time, default_joins, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            target_db,
            node.table_name,
            node.purpose,
            node.table_type,
            node.grain,
            json.dumps(node.key_columns),
            json.dumps(node.foreign_keys),
            json.dumps(node.measures),
            json.dumps(node.dimensions),
            json.dumps(node.time_columns),
            node.row_count,
            1 if node.has_time else 0,
            json.dumps(node.default_joins),
            datetime.now().isoformat()
        ))
        conn.commit()

    def get_graph_nodes(self, target_db: str) -> Dict[str, GraphNode]:
        """Get all graph nodes for a target database"""
        conn = self.vec_store._connection
        cursor = conn.execute(
            "SELECT * FROM graph_nodes WHERE target_db = ?",
            (target_db,)
        )
        
        nodes = {}
        for row in cursor:
            node = GraphNode(
                table_name=row['table_name'],
                purpose=row['purpose'],
                table_type=row['table_type'],
                grain=row['grain'],
                key_columns=json.loads(row['key_columns'] or '[]'),
                foreign_keys=json.loads(row['foreign_keys'] or '[]'),
                measures=json.loads(row['measures'] or '[]'),
                dimensions=json.loads(row['dimensions'] or '[]'),
                time_columns=json.loads(row['time_columns'] or '[]'),
                row_count=row['row_count'],
                has_time=bool(row['has_time']),
                default_joins=json.loads(row['default_joins'] or '{}')
            )
            nodes[row['table_name']] = node
        
        return nodes