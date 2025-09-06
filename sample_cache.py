from typing import Dict, Any, Optional, List
import logging
import json
from datetime import datetime, date

logger = logging.getLogger(__name__)

class SampleDataCache:
    """Sample data cache that uses persistence manager."""
    
    def __init__(self, persistence_manager, target_db: str):
        self.persistence_manager = persistence_manager
        self.target_db = target_db
    
    def get(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get sample data from persistent storage."""
        return self.persistence_manager.get_sample_data(self.target_db, table_name)
    
    def put(self, table_name: str, sample_data: Dict[str, Any], ttl_hours: int = 24):
        """Store sample data in persistent storage."""
        # Serialize datetime objects before saving
        serialized_data = self._serialize_sample_data(sample_data)
        self.persistence_manager.save_sample_data(self.target_db, table_name, serialized_data)
    
    def _serialize_sample_data(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert datetime objects to strings for JSON serialization."""
        if not sample_data or 'rows' not in sample_data:
            return sample_data
        
        # Create a copy to avoid modifying the original
        serialized = {
            'columns': sample_data.get('columns', []),
            'row_count': sample_data.get('row_count', 0),
            'error': sample_data.get('error')
        }
        
        # Serialize each row
        serialized_rows = []
        for row in sample_data.get('rows', []):
            serialized_row = {}
            for key, value in row.items():
                if isinstance(value, datetime):
                    serialized_row[key] = value.isoformat()
                elif isinstance(value, date):
                    serialized_row[key] = value.isoformat()
                elif hasattr(value, 'isoformat'):  # Handle other datetime-like objects
                    serialized_row[key] = value.isoformat()
                else:
                    serialized_row[key] = value
            serialized_rows.append(serialized_row)
        
        serialized['rows'] = serialized_rows
        return serialized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics from database."""
        conn = self.persistence_manager.vec_store._connection
        cursor = conn.execute("""
            SELECT COUNT(*) as entries, 
                   COALESCE(SUM(hits), 0) as total_hits, 
                   COALESCE(AVG(hits), 0) as avg_hits
            FROM sample_cache_meta
            WHERE target_db = ?
        """, (self.target_db,))
        
        row = cursor.fetchone()
        return {
            'entries': row['entries'] or 0,
            'total_hits': row['total_hits'] or 0,
            'avg_hits': float(row['avg_hits'] or 0)
        }
    
    def clear_cache(self, table_name: Optional[str] = None) -> int:
        """Clear cached samples for a specific table or all tables"""
        conn = self.persistence_manager.vec_store._connection
        
        if table_name:
            cursor = conn.execute("""
            DELETE FROM sample_cache_meta 
            WHERE target_db = ? AND table_name = ?
            """, (self.target_db, table_name))
            
            namespace = self.persistence_manager.get_target_db_namespace(self.target_db)
            try:
                self.persistence_manager.vec_store.delete(
                    namespace + ("sample_data",),
                    table_name
                )
            except:
                pass
            
            deleted_count = cursor.rowcount
            logger.info(f"Cleared cache for table {table_name}")
        else:
            cursor = conn.execute("""
            DELETE FROM sample_cache_meta 
            WHERE target_db = ?
            """, (self.target_db,))
            
            deleted_count = cursor.rowcount
            logger.info(f"Cleared cache for all {deleted_count} tables")
        
        conn.commit()
        return deleted_count
    
    def get_cached_tables(self) -> List[str]:
        """Get list of tables with cached samples"""
        conn = self.persistence_manager.vec_store._connection
        cursor = conn.execute("""
        SELECT table_name 
        FROM sample_cache_meta 
        WHERE target_db = ?
        ORDER BY table_name
        """, (self.target_db,))
        
        return [row[0] for row in cursor.fetchall()]
    
