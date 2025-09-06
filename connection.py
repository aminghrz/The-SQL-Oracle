import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from typing import Optional, Dict, Any
import logging
from contextlib import contextmanager
from datetime import datetime, date

logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self, uri: str):
        self.uri = uri
        self.engine: Optional[Engine] = None
        self._metadata_cache: Dict[str, Any] = {}
        self.dialect_name: Optional[str] = None
        
    def connect(self):
        """Establish database connection with connection pooling"""
        try:
            self.engine = sa.create_engine(
                self.uri,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600
            )
            
            # Detect dialect
            self.dialect_name = self.engine.dialect.name
            logger.info(f"Detected database dialect: {self.dialect_name}")
            
            # Test connection with dialect-specific query
            with self.engine.connect() as conn:
                if self.dialect_name == 'oracle':
                    conn.execute(sa.text("SELECT 1 FROM DUAL"))
                elif self.dialect_name == 'postgresql':
                    conn.execute(sa.text("SELECT 1"))
                elif self.dialect_name == 'mysql':
                    conn.execute(sa.text("SELECT 1"))
                elif self.dialect_name == 'sqlite':
                    conn.execute(sa.text("SELECT 1"))
                else:
                    # Generic fallback
                    conn.execute(sa.text("SELECT 1"))
                    
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        if not self.engine:
            self.connect()
            
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute a query with optional timeout"""
        try:
            with self.get_connection() as conn:
                # Handle timeout based on dialect
                if timeout:
                    if self.dialect_name == 'sqlite':
                        conn.execute(sa.text(f"PRAGMA busy_timeout = {timeout * 1000}"))
                    elif self.dialect_name == 'postgresql':
                        conn.execute(sa.text(f"SET statement_timeout = {timeout * 1000}"))
                    elif self.dialect_name == 'oracle':
                        # Oracle doesn't have a simple statement timeout
                        # You might need to use Resource Manager or other approaches
                        pass
                    elif self.dialect_name == 'mysql':
                        conn.execute(sa.text(f"SET SESSION max_execution_time = {timeout * 1000}"))
                
                result = conn.execute(sa.text(query))
                
                # Fetch results
                columns = list(result.keys())
                rows = result.fetchall()
                
                return {
                    'columns': columns,
                    'rows': [dict(zip(columns, row)) for row in rows],
                    'row_count': len(rows),
                    'error': None
                }
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {
                'columns': [],
                'rows': [],
                'row_count': 0,
                'error': str(e)
            }
    
    def get_table_sample(self, table_name: str, limit: int = 10) -> Dict[str, Any]:
        """Fetch sample data from a table using SQLAlchemy"""
        try:
            with self.get_connection() as conn:
                # Parse schema and table name
                if '.' in table_name:
                    schema, table_name_only = table_name.split('.', 1)
                else:
                    schema = None
                    table_name_only = table_name

                # Create table object
                metadata = sa.MetaData()
                # Try to reflect the table
                try:
                    table = sa.Table(table_name_only, metadata,
                                autoload_with=conn, schema=schema)
                except sa.exc.NoSuchTableError:
                    # If table not found with schema, try without schema
                    if schema:
                        logger.warning(f"Table {schema}.{table_name_only} not found, trying default schema")
                        table = sa.Table(table_name_only, metadata, autoload_with=conn)
                    else:
                        raise

                # Filter out problematic column types for sampling
                safe_columns = []
                for col in table.columns:
                    col_type_str = str(col.type).upper()
                    # Skip BLOB, CLOB, and other binary types
                    if any(blob_type in col_type_str for blob_type in ['BLOB', 'CLOB', 'LONG RAW', 'BFILE']):
                        logger.debug(f"Skipping column {col.name} with type {col.type} for sampling")
                        continue
                    safe_columns.append(col)

                if not safe_columns:
                    # If no safe columns, just get the first few non-blob columns
                    safe_columns = [col for col in table.columns if 'BLOB' not in str(col.type).upper()][:5]

                # Create query with only safe columns
                if safe_columns:
                    query = sa.select(*safe_columns).limit(limit)
                else:
                    # Fallback: select count
                    query = sa.select(sa.func.count()).select_from(table)

                # Execute query
                result = conn.execute(query)

                # Get column names from safe columns
                columns = [col.name for col in safe_columns] if safe_columns else ['count']

                # Fetch rows and convert to dict format with safe handling
                rows = []
                for row in result:
                    row_dict = {}
                    for i, col_name in enumerate(columns):
                        try:
                            value = row[i]
                            # Handle different data types safely
                            if isinstance(value, (datetime, date)):
                                row_dict[col_name] = value.isoformat()
                            elif hasattr(value, 'isoformat'):
                                row_dict[col_name] = value.isoformat()
                            elif isinstance(value, bytes):
                                # Handle binary data
                                try:
                                    # Try to decode as UTF-8
                                    row_dict[col_name] = value.decode('utf-8')
                                except UnicodeDecodeError:
                                    # If that fails, represent as hex or placeholder
                                    if len(value) > 50:
                                        row_dict[col_name] = f"<binary_data_{len(value)}_bytes>"
                                    else:
                                        row_dict[col_name] = value.hex()
                            elif isinstance(value, str):
                                # Ensure string is clean
                                clean_value = ''.join(c if ord(c) >= 32 or c in '\t\n\r' else '?' for c in value)
                                row_dict[col_name] = clean_value
                            else:
                                row_dict[col_name] = value
                        except Exception as e:
                            logger.warning(f"Error processing column {col_name}: {e}")
                            row_dict[col_name] = "<error_reading_value>"
                    
                    rows.append(row_dict)

                return {
                    'columns': columns,
                    'rows': rows,
                    'row_count': len(rows),
                    'error': None
                }

        except Exception as e:
            logger.error(f"Failed to get sample for {table_name}: {e}")
            return {
                'columns': [],
                'rows': [],
                'row_count': 0,
                'error': str(e)
            }
    
    def get_table_ddl(self, table_name: str) -> str:
        """Get table DDL using SQLAlchemy's CreateTable"""
        try:
            # Split schema and table name if present
            if '.' in table_name:
                schema_name, table_name_only = table_name.split('.', 1)
            else:
                schema_name = None
                table_name_only = table_name
            
            # Create metadata object
            metadata = sa.MetaData()
            
            # Reflect the table structure
            table = sa.Table(
                table_name_only, 
                metadata, 
                autoload_with=self.engine, 
                schema=schema_name
            )
            
            # Generate DDL using CreateTable
            from sqlalchemy.schema import CreateTable
            create_table_stmt = CreateTable(table)
            
            # Compile the statement for the specific dialect
            ddl = str(create_table_stmt.compile(self.engine))
            logger.info(f"got DDL for {table_name}")
            return ddl
            
        except sa.exc.NoSuchTableError:
            logger.error(f"Table {table_name} does not exist")
            return f"-- Table {table_name} not found"
        except Exception as e:
            logger.error(f"Failed to get DDL for {table_name}: {e}")
            return f"-- DDL not available for {table_name}: {str(e)}"