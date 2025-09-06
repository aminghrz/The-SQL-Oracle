import time
from typing import Any, Optional
import pandas as pd
import logging
from models import QueryResult
from connection import DatabaseConnection
from config import config

logger = logging.getLogger(__name__)

class QueryExecutor:
    def __init__(self, connection: DatabaseConnection):
        self.connection = connection
        self.dialect = connection.dialect_name

    def execute(self, sql_query: str, intent: Any,
                timeout: Optional[int] = None) -> QueryResult:
        """Execute SQL query with validation"""
        start_time = time.time()
        # Determine timeout based on query complexity
        if timeout is None:
            timeout = (config.QUERY_TIMEOUT_SIMPLE
                       if intent.query_complexity.value == 'simple'
                       else config.QUERY_TIMEOUT_COMPLEX)
        # First, execute test query with LIMIT
        test_result = self._execute_test_query(sql_query, timeout=5)
        if test_result.error:
            return test_result
        # Validate test results
        test_score, validation_checks = self._validate_test_results(
            test_result.data, intent
        )
        if test_score < 0.7:
            logger.warning(f"Test query validation failed: {validation_checks}")
            test_result.validation_score = test_score
            test_result.validation_checks = validation_checks
            return test_result
        # Execute full query
        full_result = self._execute_full_query(sql_query, timeout)
        # Add execution time
        full_result.execution_time = time.time() - start_time
        # Validate full results
        validation_score, validation_checks = self._validate_results(
            full_result.data, intent, sql_query
        )
        full_result.validation_score = validation_score
        full_result.validation_checks = validation_checks
        return full_result

    def _execute_test_query(self, sql_query: str, timeout: int) -> QueryResult:
        """Execute query with LIMIT for testing"""
        # Add LIMIT if not present
        test_query = sql_query
        sql_upper = sql_query.upper()
        
        # Check if query already has limiting clause
        has_limit = any(keyword in sql_upper for keyword in ['LIMIT', 'FETCH', 'TOP', 'ROWNUM'])
        
        if not has_limit:
            # For test query only, add a small limit
            # Remove trailing semicolon if present
            test_query = sql_query.rstrip(';')
            
            # Add dialect-specific limiting clause
            if self.dialect == 'oracle':
                # For Oracle, wrap the entire query in a subquery with ROWNUM
                test_query = f"SELECT * FROM ({test_query}) WHERE ROWNUM <= 10"
            elif self.dialect == 'mssql':
                # For SQL Server, add TOP after SELECT
                select_idx = test_query.upper().find('SELECT')
                if select_idx != -1:
                    test_query = (test_query[:select_idx + 6] +
                                ' TOP 10' +
                                test_query[select_idx + 6:])
            else:
                # PostgreSQL, MySQL, SQLite use LIMIT
                test_query += ' LIMIT 10'
        
        result_dict = self.connection.execute_query(test_query, timeout)
        
        # Convert to DataFrame for easier manipulation
        if result_dict['error']:
            return QueryResult(
                sql=test_query,
                data=pd.DataFrame(),
                execution_time=0,
                error=result_dict['error']
            )
        
        df = pd.DataFrame(result_dict['rows'])
        return QueryResult(
            sql=test_query,
            data=df,
            execution_time=0,
            error=None
        )

    def _execute_full_query(self, sql_query: str, timeout: int) -> QueryResult:
        """Execute the full query"""
        result_dict = self.connection.execute_query(sql_query, timeout)
        if result_dict['error']:
            return QueryResult(
                sql=sql_query,
                data=pd.DataFrame(),
                execution_time=0,
                error=result_dict['error']
            )
        df = pd.DataFrame(result_dict['rows'])
        return QueryResult(
            sql=sql_query,
            data=df,
            execution_time=0,
            error=None
        )

    def _validate_test_results(self, results: pd.DataFrame, intent: Any) -> tuple:
        """Validate test query results"""
        validation_checks = {
            'not_empty': len(results) > 0,
            'has_columns': len(results.columns) > 0 if not results.empty else False,
            'no_all_nulls': True,
            'no_errors': True
        }
        if not results.empty and intent.metrics:
            # Check if requested metrics are in columns
            validation_checks['has_requested_columns'] = any(
                metric.lower() in col.lower()
                for col in results.columns
                for metric in intent.metrics
            )
            # Check for all nulls in important columns
            if validation_checks['has_requested_columns']:
                important_cols = [col for col in results.columns
                                  if any(metric.lower() in col.lower()
                                         for metric in intent.metrics)]
                validation_checks['no_all_nulls'] = not results[important_cols].isnull().all().all()
        # Calculate score
        test_score = (
            0.3 * validation_checks['not_empty'] +
            0.3 * validation_checks.get('has_requested_columns', 1.0) +
            0.2 * validation_checks['no_all_nulls'] +
            0.2 * validation_checks['no_errors']
        )
        return test_score, validation_checks

    def _validate_results(self, results: pd.DataFrame, intent: Any,
                           query: str) -> tuple:
        """Comprehensive result validation"""
        validation_checks = {
            'not_empty': len(results) > 0,
            'has_requested_columns': True,
            'no_all_nulls': True,
            'row_count_matches_intent': True,
            'data_types_valid': True
        }
        if results.empty:
            validation_checks['has_requested_columns'] = False
            validation_checks['no_all_nulls'] = False
            validation_checks['row_count_matches_intent'] = False
            validation_checks['data_types_valid'] = False
        else:
            # Check columns
            if intent.metrics:
                validation_checks['has_requested_columns'] = any(
                    metric.lower() in col.lower()
                    for col in results.columns
                    for metric in intent.metrics
                )
            # Check nulls
            validation_checks['no_all_nulls'] = not results.isnull().all().all()
            # Check row count expectations
            validation_checks['row_count_matches_intent'] = self._check_row_count_expectation(
                results, intent
            )
            # Check data types
            validation_checks['data_types_valid'] = self._validate_data_types(
                results, intent
            )
        # Calculate weighted score
        weights = {
            'not_empty': 0.3,
            'has_requested_columns': 0.3,
            'no_all_nulls': 0.2,
            'row_count_matches_intent': 0.1,
            'data_types_valid': 0.1
        }
        validation_score = sum(
            weights[k] * (1.0 if v else 0.0)
            for k, v in validation_checks.items()
        )
        return validation_score, validation_checks

    def _check_row_count_expectation(self, results: pd.DataFrame, intent: Any) -> bool:
        """Check if row count matches intent expectations"""
        row_count = len(results)
        if intent.expects_single_value and row_count == 1:
            return True
        elif intent.expects_list and row_count > 1:
            return True
        elif intent.expects_aggregation and row_count <= 100:
            return True
        # Default: any non-empty result is acceptable
        return row_count > 0

    def _validate_data_types(self, results: pd.DataFrame, intent: Any) -> bool:
        """Validate that data types are appropriate"""
        if results.empty:
            return False
        # Check if numeric columns exist for aggregation queries
        if intent.intent_type.value == 'aggregation':
            numeric_dtypes = ['int64', 'float64', 'int32', 'float32']
            has_numeric = any(dtype in numeric_dtypes for dtype in results.dtypes)
            return has_numeric
        return True