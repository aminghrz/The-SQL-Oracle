from typing import Dict, Any, List
from queue import Queue
import threading
import logging
from datetime import datetime, timedelta
import sqlalchemy as sa

logger = logging.getLogger(__name__)

class SchemaAnalyzer:
    def __init__(self, connection, metadata_extractor):
        self.connection = connection
        self.metadata_extractor = metadata_extractor
        self.analyzed_tables: Dict[str, Dict[str, Any]] = {}
        self.analysis_queue = Queue()
        self.sample_cache: Dict[str, Dict[str, Any]] = {}
        self._start_background_worker()

    def _start_background_worker(self):
        """Start background thread for deep analysis"""
        def worker():
            while True:
                try:
                    table_name = self.analysis_queue.get(timeout=60)
                    if table_name:
                        self._perform_deep_analysis(table_name)
                except:
                    continue

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def get_table_analysis(self, table_name: str) -> Dict[str, Any]:
        """Get analysis for a table, performing basic analysis if needed"""
        if table_name not in self.analyzed_tables:
            # Quick metadata only
            basic_info = self.get_basic_metadata(table_name)
            self.analyzed_tables[table_name] = {
                'status': 'basic',
                'metadata': basic_info,
                'confidence': 0.3,
                'last_updated': datetime.now()
            }
            # Queue for deep analysis
            self.analysis_queue.put(table_name)

        return self.analyzed_tables[table_name]

    def get_basic_metadata(self, table_name: str) -> Dict[str, Any]:
        """Get basic metadata for a table using SQLAlchemy"""
        try:
            with self.connection.get_connection() as conn:
                # Parse schema and table name
                if '.' in table_name:
                    schema, table_name_only = table_name.split('.', 1)
                else:
                    schema = None
                    table_name_only = table_name
                
                # Create metadata and reflect table
                metadata = sa.MetaData()
                table = sa.Table(table_name_only, metadata,
                                autoload_with=conn, schema=schema)
                
                # Get column information
                columns = []
                for col in table.columns:
                    columns.append({
                        'name': col.name,
                        'type': col.type,
                        'nullable': col.nullable,
                        'primary_key': col.primary_key
                    })
                
                # Get row count using SQLAlchemy
                count_query = sa.select(sa.func.count()).select_from(table)
                result = conn.execute(count_query)
                row_count = result.scalar()
                
                return {
                    'columns': columns,
                    'row_count': row_count,
                    'analyzed_at': datetime.now()
                }
        
        except Exception as e:
            logger.error(f"Failed to get basic metadata for {table_name}: {e}")
            return {'columns': [], 'row_count': 0, 'error': str(e)}

    def _perform_deep_analysis(self, table_name: str):
        """Perform deep analysis of a table in background"""
        try:
            logger.info(f"Starting deep analysis for {table_name}")

            # Get statistics
            stats = self.metadata_extractor.get_table_statistics(table_name)

            # Get sample data
            sample = self.connection.get_table_sample(table_name, limit=10)

            # Analyze data patterns
            patterns = self._analyze_data_patterns(sample)

            # Update analysis
            self.analyzed_tables[table_name] = {
                'status': 'deep',
                'metadata': self.analyzed_tables[table_name].get('metadata', {}),
                'statistics': stats,
                'patterns': patterns,
                'sample': sample,
                'confidence': 0.8,
                'last_updated': datetime.now()
            }

            logger.info(f"Completed deep analysis for {table_name}")

        except Exception as e:
            logger.error(f"Deep analysis failed for {table_name}: {e}")

    def _analyze_data_patterns(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in sample data"""
        patterns = {
            'has_nulls': {},
            'data_types': {},
            'value_patterns': {}
        }

        if not sample_data['rows']:
            return patterns

        columns = sample_data['columns']
        rows = sample_data['rows']

        for col in columns:
            # Check for nulls
            null_count = sum(1 for row in rows if row.get(col) is None)
            patterns['has_nulls'][col] = null_count > 0

            # Infer data types from values
            values = [row.get(col) for row in rows if row.get(col) is not None]
            if values:
                # Simple type inference
                sample_value = values[0]
                if isinstance(sample_value, (int, float)):
                    patterns['data_types'][col] = 'numeric'
                elif isinstance(sample_value, datetime):
                    patterns['data_types'][col] = 'datetime'
                else:
                    patterns['data_types'][col] = 'text'

                # Look for patterns (e.g., all values follow a format)
                if patterns['data_types'][col] == 'text' and len(values) > 2:
                    # Check if values follow a pattern (simplified)
                    lengths = [len(str(v)) for v in values]
                    if len(set(lengths)) == 1:
                        patterns['value_patterns'][col] = f'fixed_length_{lengths[0]}'

        return patterns
        
    def get_column_stats(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Get basic statistics for a column using SQLAlchemy"""
        try:
            with self.connection.get_connection() as conn:
                # Parse schema and table name
                if '.' in table_name:
                    schema, table_name_only = table_name.split('.', 1)
                else:
                    schema = None
                    table_name_only = table_name
                
                # Create table object
                metadata = sa.MetaData()
                table = sa.Table(table_name_only, metadata,
                                autoload_with=conn, schema=schema)
                col = table.c[column_name]
                
                # Get distinct count
                distinct_query = sa.select(
                    sa.func.count(sa.distinct(col))
                )
                result = conn.execute(distinct_query)
                distinct_count = result.scalar()
                
                # Get min/max for appropriate types
                col_type = str(col.type)
                min_val = None
                max_val = None
                
                if any(t in col_type.upper() for t in ['INT', 'FLOAT', 'NUMERIC', 'NUMBER', 'DATE', 'TIME']):
                    minmax_query = sa.select(
                        sa.func.min(col),
                        sa.func.max(col)
                    )
                    result = conn.execute(minmax_query)
                    row = result.first()
                    if row:
                        min_val, max_val = row
                
                # Get top values for categorical columns
                top_values = []
                if distinct_count < 100: # Only for low cardinality
                    top_query = sa.select(
                        col,
                        sa.func.count().label('cnt')
                    ).where(
                        col.isnot(None)
                    ).group_by(
                        col
                    ).order_by(
                        sa.desc('cnt')
                    ).limit(10)
                    
                    result = conn.execute(top_query)
                    top_values = [(row[0], row[1]) for row in result]
                
                return {
                    'distinct_count': distinct_count,
                    'min': min_val,
                    'max': max_val,
                    'top_values': top_values
                }
        
        except Exception as e:
            logger.error(f"Failed to get stats for {table_name}.{column_name}: {e}")
            return {}

    def estimate_inclusion(self, table_a: str, col_a: str, table_b: str, col_b: str,
                        sample_size: int = 1000) -> float:
        """Estimate if values in col_a are included in col_b using SQLAlchemy"""
        try:
            with self.connection.get_connection() as conn:
                # Parse schemas and table names
                schema_a = None
                schema_b = None
                
                if '.' in table_a:
                    schema_a, table_a_only = table_a.split('.', 1)
                else:
                    table_a_only = table_a
                
                if '.' in table_b:
                    schema_b, table_b_only = table_b.split('.', 1)
                else:
                    table_b_only = table_b
                
                # Create table objects
                metadata = sa.MetaData()
                table_obj_a = sa.Table(table_a_only, metadata,
                                    autoload_with=conn, schema=schema_a)
                table_obj_b = sa.Table(table_b_only, metadata,
                                    autoload_with=conn, schema=schema_b)
                
                # Get column objects
                col_obj_a = table_obj_a.c[col_a]
                col_obj_b = table_obj_b.c[col_b]
                
                # Get sample of distinct values from table A
                sample_query = sa.select(col_obj_a).distinct().where(
                    col_obj_a.isnot(None)
                ).limit(sample_size)
                
                result_a = conn.execute(sample_query)
                values_a = [row[0] for row in result_a]
                
                if not values_a:
                    return 0.0
                
                # Check how many exist in table B
                # Use IN clause for efficiency
                check_values = values_a[:100] # Limit to 100 for performance
                exists_query = sa.select(
                    sa.func.count(sa.distinct(col_obj_b))
                ).where(
                    col_obj_b.in_(check_values)
                )
                
                result = conn.execute(exists_query)
                found_count = result.scalar()
                
                inclusion_ratio = found_count / len(check_values)
                return inclusion_ratio
        
        except Exception as e:
            logger.error(f"Failed to estimate inclusion {table_a}.{col_a} -> {table_b}.{col_b}: {e}")
            return 0.0

    def compute_join_candidates(self, table_name: str, target_tables: List[str]) -> List[Dict[str, Any]]:
        """Find potential join columns between tables."""
        candidates = []
        
        try:
            # Get columns for source table
            with self.connection.get_connection() as conn:
                inspector = sa.inspect(conn)
                source_columns = inspector.get_columns(table_name)
            
            for target_table in target_tables:
                if target_table == table_name:
                    continue
                    
                # Get columns for target table
                target_columns = inspector.get_columns(target_table)
                
                # Check each column pair
                for src_col in source_columns:
                    for tgt_col in target_columns:
                        # Skip if types are incompatible
                        if not self._are_types_compatible(src_col['type'], tgt_col['type']):
                            continue
                        
                        # Check name similarity (simple check)
                        if src_col['name'].lower() == tgt_col['name'].lower():
                            candidates.append({
                                'source': f"{table_name}.{src_col['name']}",
                                'target': f"{target_table}.{tgt_col['name']}",
                                'confidence': 0.8,
                                'reason': 'exact_name_match'
                            })
                        elif src_col['name'].lower().endswith('_id') and tgt_col['name'].lower() == 'id':
                            # Check inclusion
                            inclusion = self.estimate_inclusion(
                                table_name, src_col['name'], 
                                target_table, tgt_col['name']
                            )
                            if inclusion > 0.9:
                                candidates.append({
                                    'source': f"{table_name}.{src_col['name']}",
                                    'target': f"{target_table}.{tgt_col['name']}",
                                    'confidence': inclusion,
                                    'reason': 'fk_pattern_with_inclusion'
                                })
        
        except Exception as e:
            logger.error(f"Failed to compute join candidates for {table_name}: {e}")
        
        return candidates

    def _are_types_compatible(self, type_a: Any, type_b: Any) -> bool:
        """Check if two column types are compatible for joining."""
        # Convert to string for comparison
        type_a_str = str(type_a).upper()
        type_b_str = str(type_b).upper()
        
        # Both numeric
        numeric_types = ['INTEGER', 'BIGINT', 'SMALLINT', 'NUMERIC', 'DECIMAL', 'FLOAT', 'REAL', 'DOUBLE']
        if any(t in type_a_str for t in numeric_types) and any(t in type_b_str for t in numeric_types):
            return True
        
        # Both string
        string_types = ['VARCHAR', 'CHAR', 'TEXT', 'STRING']
        if any(t in type_a_str for t in string_types) and any(t in type_b_str for t in string_types):
            return True
        
        # Exact match
        return type_a_str == type_b_str

    def should_refresh_analysis(self, table_name: str, max_age_hours: int = 24) -> bool:
        """Check if analysis should be refreshed"""
        if table_name not in self.analyzed_tables:
            return True

        analysis = self.analyzed_tables[table_name]
        age = datetime.now() - analysis.get('last_updated', datetime.min)

        return age > timedelta(hours=max_age_hours)