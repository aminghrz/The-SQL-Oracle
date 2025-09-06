import sqlalchemy as sa
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
from models import TableInfo

logger = logging.getLogger(__name__)

class MetadataExtractor:
    def __init__(self, connection):
        self.connection = connection
        self.metadata_cache = {}
    
    def extract_all_metadata(self) -> Dict[str, TableInfo]:
        """Extract metadata for all tables in all accessible schemas using SQLAlchemy"""
        tables = {}
        
        try:
            with self.connection.get_connection() as conn:
                inspector = sa.inspect(conn)
                
                # Get all schema names
                schemas = inspector.get_schema_names()
                logger.info(f"Found {len(schemas)} schemas in database")
                
                # Process each schema
                for schema in schemas:
                    try:
                        # Skip system schemas based on dialect
                        if self._is_system_schema(schema):
                            logger.debug(f"Skipping system schema: {schema}")
                            continue
                        
                        # Get tables from this schema
                        table_names = inspector.get_table_names(schema=schema)
                        
                        if table_names:
                            logger.info(f"Processing {len(table_names)} tables in schema '{schema}'")
                            
                            # Create metadata object for this schema
                            metadata = sa.MetaData()
                            
                            # Reflect all tables from this schema
                            metadata.reflect(bind=conn, schema=schema)
                            
                            # Process each table
                            for table_name in table_names:
                                try:
                                    full_name = f"{schema}.{table_name}"
                                    table = metadata.tables.get(full_name)
                                    
                                    if table is not None:
                                        # Convert SQLAlchemy columns to dict format
                                        columns = []
                                        for col in table.columns:
                                            columns.append({
                                                'name': col.name,
                                                'type': col.type,
                                                'nullable': col.nullable,
                                                'default': col.default,
                                                'autoincrement': col.autoincrement,
                                                'primary_key': col.primary_key
                                            })
                                        
                                        # Get DDL using SQLAlchemy
                                        from sqlalchemy.schema import CreateTable
                                        ddl = str(CreateTable(table).compile(self.connection.engine))
                                        
                                        table_info = TableInfo(
                                            name=table_name,
                                            schema=schema,
                                            columns=columns,
                                            ddl=ddl,
                                            last_analyzed=datetime.now()
                                        )
                                        
                                        tables[full_name] = table_info
                                        
                                except Exception as e:
                                    logger.warning(f"Failed to process table {schema}.{table_name}: {e}")
                                    
                    except Exception as e:
                        logger.warning(f"Cannot access schema '{schema}': {e}")
                
                # Also check default schema (None)
                try:
                    default_tables = inspector.get_table_names()
                    if default_tables:
                        logger.info(f"Processing {len(default_tables)} tables in default schema")
                        
                        metadata = sa.MetaData()
                        metadata.reflect(bind=conn)
                        
                        for table_name, table in metadata.tables.items():
                            # Skip if already processed with schema prefix
                            if '.' not in table_name and table_name not in [t.split('.')[-1] for t in tables.keys()]:
                                try:
                                    # Convert SQLAlchemy columns to dict format
                                    columns = []
                                    for col in table.columns:
                                        columns.append({
                                            'name': col.name,
                                            'type': col.type,
                                            'nullable': col.nullable,
                                            'default': col.default,
                                            'autoincrement': col.autoincrement,
                                            'primary_key': col.primary_key
                                        })
                                    
                                    # Get DDL using SQLAlchemy
                                    from sqlalchemy.schema import CreateTable
                                    ddl = str(CreateTable(table).compile(self.connection.engine))
                                    
                                    table_info = TableInfo(
                                        name=table_name,
                                        schema=None,
                                        columns=columns,
                                        ddl=ddl,
                                        last_analyzed=datetime.now()
                                    )
                                    
                                    tables[table_name] = table_info
                                    
                                except Exception as e:
                                    logger.warning(f"Failed to process table {table_name}: {e}")
                                    
                except Exception as e:
                    logger.warning(f"Cannot access default schema: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
        
        logger.info(f"Extracted metadata for {len(tables)} tables across all schemas")
        return tables
    
    def _is_system_schema(self, schema: str) -> bool:
        """Check if schema is a system schema that should be skipped"""
        schema_lower = schema.lower()
        
        # Common system schemas across different databases
        system_schemas = {
            'information_schema', 'pg_catalog', 'pg_toast', 'pg_temp',  # PostgreSQL
            'sys', 'mysql', 'performance_schema',  # MySQL
            'system', 'sysaux', 'sys',  # Oracle
            'master', 'tempdb', 'model', 'msdb',  # SQL Server
            'temp', 'main'  # SQLite
        }
        
        return schema_lower in system_schemas
    
    def extract_foreign_keys(self) -> List[Tuple[str, str]]:
        """Extract all foreign key relationships across all schemas using SQLAlchemy"""
        relationships = []
        
        try:
            with self.connection.get_connection() as conn:
                inspector = sa.inspect(conn)
                
                # Get all schemas
                schemas = inspector.get_schema_names()
                
                # Process each schema
                for schema in schemas:
                    if self._is_system_schema(schema):
                        continue
                        
                    try:
                        # Get foreign keys for each table in schema
                        table_names = inspector.get_table_names(schema=schema)
                        
                        for table_name in table_names:
                            from_table = f"{schema}.{table_name}"
                            
                            # Get foreign keys
                            fks = inspector.get_foreign_keys(table_name, schema=schema)
                            
                            for fk in fks:
                                referred_schema = fk.get('referred_schema', schema)
                                referred_table = fk['referred_table']
                                
                                if referred_schema:
                                    to_table = f"{referred_schema}.{referred_table}"
                                else:
                                    to_table = referred_table
                                    
                                relationships.append((from_table, to_table))
                                
                    except Exception as e:
                        logger.warning(f"Failed to extract foreign keys from schema {schema}: {e}")
                
                # Also check default schema
                try:
                    default_tables = inspector.get_table_names()
                    for table_name in default_tables:
                        fks = inspector.get_foreign_keys(table_name)
                        
                        for fk in fks:
                            referred_schema = fk.get('referred_schema')
                            referred_table = fk['referred_table']
                            
                            if referred_schema:
                                to_table = f"{referred_schema}.{referred_table}"
                            else:
                                to_table = referred_table
                                
                            relationships.append((table_name, to_table))
                            
                except Exception as e:
                    logger.warning(f"Failed to extract foreign keys from default schema: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to extract foreign keys: {e}")
        
        logger.info(f"Found {len(relationships)} foreign key relationships across all schemas")
        return relationships
    
    def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get table statistics using SQLAlchemy"""
        stats = {}
        
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
                
                # Get row count
                count_query = sa.select(sa.func.count()).select_from(table)
                result = conn.execute(count_query)
                stats['row_count'] = result.scalar()
                
                # Get column statistics
                column_stats = {}
                for col in table.columns:
                    col_stat = {
                        'data_type': str(col.type),
                        'nullable': col.nullable,
                        'primary_key': col.primary_key
                    }
                    
                    # Get distinct count for small tables
                    if stats['row_count'] < 10000:
                        distinct_query = sa.select(
                            sa.func.count(sa.distinct(col))
                        ).select_from(table)
                        result = conn.execute(distinct_query)
                        col_stat['distinct_count'] = result.scalar()
                    
                    column_stats[col.name] = col_stat
                
                stats['columns'] = column_stats
                
        except Exception as e:
            logger.error(f"Failed to get statistics for {table_name}: {e}")
        
        return stats