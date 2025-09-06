from openai import OpenAI
import os
from typing import Dict, List, Any, Tuple, Optional
import logging
from models import Intent, TableInfo
from config import config

logger = logging.getLogger(__name__)

class SQLGenerator:
    def __init__(self, llm_model: str = config.LLM_MODEL, dialect: str = None):
        self.llm_model = llm_model
        self.dialect = dialect or "sql" # Default to generic SQL if not specified
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"))
        logger.info(f"SQLGenerator initialized with dialect: {self.dialect}")

    def generate_sql(self, user_prompt: str, intent: Intent,
                    table_info: Dict[str, TableInfo],
                    similar_examples: List[Dict[str, Any]],
                    sample_data: Dict[str, Dict[str, Any]],
                    retry_context: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        """Generate SQL query using LLM with DDL and sample data"""
        # Build prompt
        prompt = self._build_prompt(
            user_prompt, intent, table_info, similar_examples, sample_data, retry_context
        )
        
        # Generate SQL
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=500
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the SQL
            sql_query = self._clean_sql(sql_query)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                sql_query, table_info, similar_examples, sample_data, retry_context
            )
            
            return sql_query, confidence
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            # Return a simple fallback query
            fallback_sql = self._generate_fallback_sql(table_info)
            return fallback_sql, 0.3
        
    def _get_system_prompt(self) -> str:
        """Get dialect-specific system prompt"""
        base_prompt = """You are an expert SQL developer who generates precise, syntactically correct SQL queries.

    CRITICAL TEXT FILTERING RULES:
    - When filtering text/name fields, ALWAYS use case-insensitive matching with LOWER()
    - ALWAYS use LIKE with % wildcards for partial matching
    - If the search term contains non-English characters, ALWAYS provide BOTH versions:
    - Original language version
    - English transliteration
    - Combine both with OR: LOWER(col) LIKE '%original%' OR LOWER(col) LIKE '%transliterated%'
    - This applies to any text search, especially names and user-provided text values

    """
        
        if self.dialect == "oracle":
            return base_prompt + """Oracle SQL specific rules:
    - Use TO_DATE() for date conversions, not DATE()
    - Use SUBSTR() instead of LEFT() or RIGHT()
    - Use SYSDATE for current date, not CURDATE()
    - Use date arithmetic with numbers (SYSDATE - 30 for 30 days ago)
    - Use TRUNC() to remove time from dates
    - Use || for string concatenation, not CONCAT()
    - ROWNUM is applied before GROUP BY, so use subqueries if needed
    - No LIMIT clause, use ROWNUM or FETCH FIRST n ROWS ONLY (12c+)

    CRITICAL AGGREGATION RULES:
    - ALWAYS read the column analysis carefully
    - If a column is marked as "measure" or contains "pre-aggregated values", use SUM(column_name) NOT COUNT(*)
    - If you see columns which their names contains like these values: CNT, COUNT, TOTAL, AMOUNT, AMNT, SUM, QTY, NUM - these often contain pre-calculated values
    - For counting records: COUNT(*) counts rows, SUM(count_column) sums pre-calculated counts
    - Pay attention to column descriptions and semantic roles

    CRITICAL DATE HANDLING:
    - ALWAYS examine the sample data to understand the date format before using any date functions
    - If dates look like Persian/Solar calendar (years like 1400-1404), treat them as strings
    - If dates are in non-standard formats, use string functions (SUBSTR) instead of date functions
    - Only use TO_DATE when you're certain the dates are in a standard Oracle-compatible format"""
        
        elif self.dialect == "postgresql":
            return base_prompt + """PostgreSQL specific rules:
    - Use CURRENT_DATE for current date
    - Use date arithmetic with INTERVAL (CURRENT_DATE - INTERVAL '30 days')
    - Use LIMIT for row limiting
    - Use :: for type casting
    - Use ILIKE for case-insensitive pattern matching (alternative to LOWER() + LIKE)"""
        
        elif self.dialect == "mysql":
            return base_prompt + """MySQL specific rules:
    - Use CURDATE() for current date
    - Use DATE_SUB() for date arithmetic
    - Use LIMIT for row limiting
    - Collation can affect case sensitivity - use LOWER() for consistency"""
        
        else:
            return base_prompt + f"Generate precise, syntactically correct {self.dialect.upper()} SQL queries."
    def _build_prompt(self, user_prompt: str, intent: Intent,
                    table_info: Dict[str, TableInfo],
                    similar_examples: List[Dict[str, Any]],
                    sample_data: Dict[str, Dict[str, Any]],
                    retry_context: Optional[Dict[str, Any]] = None) -> str:
        """Build comprehensive prompt for SQL generation"""
        dialect_name = (self.dialect or "SQL").upper()
        prompt_parts = [
            f"Database: {dialect_name}",
            f"Question: {user_prompt}",
            f"Expected result type: {intent.expected_result_type}",
            ""
        ]
        
        if retry_context:
            prompt_parts.extend([
                "⚠️ IMPORTANT: This is a retry attempt after previous query returned no results.",
                "\nPrevious Failed Attempt:"
            ])
            
            if retry_context.get('previous_query'):
                prompt_parts.extend([
                    f"Failed Query: {retry_context['previous_query']}",
                    f"Result: {retry_context.get('previous_result', {}).get('row_count', 0)} rows"
                ])
            
            if retry_context.get('validation_issues'):
                prompt_parts.extend([
                    "Issues identified:",
                    *[f"- {issue}" for issue in retry_context['validation_issues'][:3]]
                ])
            
            prompt_parts.extend([
                "\nBased on the failure:",
                "1. The previous query logic might be incorrect",
                "2. Check if the filter conditions are too restrictive",
                "3. Ensure proper use of OR/AND operators",
                "4. Consider different column names or approaches",
                "5. Make sure to properly filter for ALL entities mentioned in the query",
                ""
            ])
        
        if self.dialect == "oracle":
            prompt_parts.extend([
                "Oracle SQL Requirements:",
                "- Use SUBSTR(column, 1, 10) instead of LEFT(column, 10)",
                "- Check sample data format before using date functions",
                "- For limiting rows with GROUP BY, use a subquery",
                ""
            ])
        
        if intent.expects_single_value:
            prompt_parts.append("Note: User expects a single value result.")
        elif intent.expects_list:
            prompt_parts.append("Note: User expects a list of results.")
        elif intent.expects_aggregation:
            prompt_parts.append("Note: User expects aggregated results.")
        
        if similar_examples:
            prompt_parts.append("\nSimilar successful queries:")
            for i, ex in enumerate(similar_examples[:3]):
                prompt_parts.append(f"\nExample {i+1}:")
                prompt_parts.append(f"Question: {ex['prompt']}")
                prompt_parts.append(f"SQL: {ex['sql']}")
        
        prompt_parts.append("\nAvailable tables:")
        for table_name, info in table_info.items():
            prompt_parts.append(f"\n-- Table: {table_name}")
            
            if hasattr(self, 'table_summaries') and table_name in self.table_summaries:
                table_summary = self.table_summaries[table_name]['summary']
                prompt_parts.append(f"-- Purpose: {table_summary.get('purpose', 'Unknown')}")
                prompt_parts.append(f"-- Table type: {table_summary.get('table_type', 'Unknown')}")
                prompt_parts.append(f"-- Grain: {table_summary.get('grain', 'Unknown')}")
            
            prompt_parts.append(info.ddl)
            
            if hasattr(self, 'column_summaries'):
                prompt_parts.append("\n-- Column Details:")
                for col in info.columns[:10]:
                    col_name = col['name']
                    col_key = f"{table_name}.{col_name}"
                    if col_key in self.column_summaries:
                        col_summary = self.column_summaries[col_key]['summary']
                        if col_summary.get('semantic_role') in ['text', 'dimension'] or 'name' in col_name.lower():
                            prompt_parts.append(f"-- {col_name}: {col_summary.get('description', '')} "
                                            f"[Role: {col_summary.get('semantic_role', 'unknown')}]")
            
            if table_name in sample_data and sample_data[table_name].get('rows'):
                sample = sample_data[table_name]
                prompt_parts.append("\n-- Sample data:")
                columns = sample['columns']
                prompt_parts.append(f"-- {' | '.join(columns)}")
                
                for row in sample['rows'][:5]:
                    row_values = []
                    for col in columns:
                        value = row.get(col, 'NULL')
                        if isinstance(value, str) and len(value) > 20:
                            value = value[:17] + "..."
                        row_values.append(str(value))
                    prompt_parts.append(f"-- {' | '.join(row_values)}")
            
            date_analysis = self._analyze_table_date_columns(table_name, sample)
            if date_analysis:
                prompt_parts.append(f"\n-- Date Column Analysis for {table_name}:")
                for col_name, format_info in date_analysis.items():
                    prompt_parts.append(f"-- Column '{col_name}': {format_info}")
        
        prompt_parts.extend([
            "",
            f"Generate a syntactically correct {dialect_name} query for the question above.",
            "Use only the tables and columns shown in the schemas.",
            "",
            "CRITICAL RULES FOR TEXT/NAME FILTERING:",
            "1. When filtering on name fields, person names, or any text that could be in different languages:",
            " - ALWAYS use LOWER() function for case-insensitive matching",
            " - ALWAYS use LIKE with % wildcards for partial matching",
            " - ALWAYS provide BOTH the original language version AND English transliteration",
            " - Use OR to combine both conditions",
            "",
            "2. Pattern to follow:",
            " WHERE LOWER(column_name) LIKE '%original_text%' OR LOWER(column_name) LIKE '%english_transliteration%'",
            "",
            "3. Examples:",
            " - If user searches for 'شکری', generate: WHERE LOWER(name) LIKE '%شکری%' OR LOWER(name) LIKE '%shokri%'",
            " - If user searches for 'احمد', generate: WHERE LOWER(name) LIKE '%احمد%' OR LOWER(name) LIKE '%ahmad%'",
            " - If user searches for 'John', generate: WHERE LOWER(name) LIKE '%john%'",
            " - If user searches for '王' (Wang), generate: WHERE LOWER(name) LIKE '%王%' OR LOWER(name) LIKE '%wang%'",
            "",
            "4. Apply this pattern to:",
            " - Columns with 'name' in their column name (username, customer_name, etc.)",
            " - Columns identified as containing person names or text in the column summaries",
            " - Any text field where the user is searching for a specific value",
            "",
            "5. Detection:",
            " - Look at the user's query to identify if they're searching for names or text values",
            " - Check if the search term contains non-English characters",
            " - If yes, provide both versions; if no, still use LOWER() and LIKE for consistency",
            "",
            "CRITICAL RULES FOR DATE HANDLING:",
            "1. ALWAYS check the sample data to understand date formats",
            "2. If dates are in Persian/Solar calendar format (YYYY/MM/DD with years like 1400-1403):",
            " - DO NOT use date functions like TO_DATE or date arithmetic",
            " - Use string operations: SUBSTR for extraction, string comparison for filtering",
            " - For 'last 30 days' with Persian dates, use string comparison with a calculated Persian date",
            "3. For the questions/queries containing date:",
            " - If the date column contains Persian dates, you need to:",
            " a) Find the maximum date using MAX(date_column)",
            " b) Use string comparison, not date arithmetic",
            " c) Consider using a different approach like sorting and limiting results",
            "4. Example for Persian date filtering:",
            " Instead of: t_date >= MAX(t_date) - 30",
            " Use: t_date >= '1403/08/20' (where you calculate the Persian date 30 days ago)",
            " Or: Use row limiting after sorting by date DESC",
            "",
            "Return only the SQL query without any explanation.",
            "DO NOT add semicolon (;) at the end of the query"
        ])
        
        return '\n'.join(prompt_parts)
    
    def _analyze_table_date_columns(self, table_name: str, sample_data: Dict[str, Any]) -> Dict[str, str]:
        """Analyze date columns for a specific table"""
        date_columns = {}
        
        if not sample_data.get('rows'):
            return date_columns
        
        columns = sample_data.get('columns', [])
        rows = sample_data.get('rows', [])
        
        for col in columns:
            # Check if column name suggests it's a date
            if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 't_date']):
                # Get sample values
                sample_values = [row.get(col) for row in rows[:5] if row.get(col)]
                if sample_values:
                    format_info = self._detect_date_format(sample_values[0])
                    date_columns[col] = format_info
                    
                    # Add specific handling hints for Persian dates
                    if "Persian" in format_info:
                        date_columns[col] += " - Use string operations only!"
        
        return date_columns

    def _analyze_date_columns(self, sample_data: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Analyze columns that might contain date/time data across all tables"""
        date_columns = {}
        for table_name, table_sample in sample_data.items():
            if not table_sample.get('rows'):
                continue
            columns = table_sample.get('columns', [])
            rows = table_sample.get('rows', [])
            for col in columns:
                # Check if column name suggests it's a date
                if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'modified']):
                    # Get sample values
                    sample_values = [row.get(col) for row in rows[:3] if row.get(col)]
                    if sample_values:
                        format_info = self._detect_date_format(sample_values[0])
                        date_columns[f"{table_name}.{col}"] = format_info
                else:
                    # Check if values look like dates (only if not already identified by name)
                    sample_values = [row.get(col) for row in rows[:3] if row.get(col)]
                    if sample_values and isinstance(sample_values[0], str):
                        if self._looks_like_date(sample_values[0]):
                            format_info = self._detect_date_format(sample_values[0])
                            date_columns[f"{table_name}.{col}"] = format_info
        return date_columns

    def _looks_like_date(self, value: str) -> bool:
        """Check if a string value looks like a date"""
        if not value or not isinstance(value, str):
            return False
        # Common date patterns
        date_patterns = [
            # Has slashes or dashes with numbers
            lambda v: ('/' in v or '-' in v) and any(c.isdigit() for c in v),
            # Starts with year-like number (1300-2100)
            lambda v: v[:4].isdigit() and 1300 <= int(v[:4]) <= 2100,
            # Contains time-like pattern
            lambda v: ':' in v and any(c.isdigit() for c in v)
        ]
        return any(pattern(value) for pattern in date_patterns)

    def _detect_date_format(self, date_str: str) -> str:
        """Detect the format of a date string"""
        if not date_str or not isinstance(date_str, str):
            return "unknown format"
        # Check for Persian date format (year 1400+)
        if date_str[:4].isdigit() and 1400 <= int(date_str[:4]) <= 1410:
            if '/' in date_str and ':' in date_str:
                return "Persian calendar with time (YYYY/MM/DD HH:MI) - use string operations, not date functions"
            elif '/' in date_str:
                return "Persian calendar (YYYY/MM/DD) - use string operations, not date functions"
        # Check for standard formats
        if '-' in date_str:
            if 'T' in date_str:
                return "ISO format (YYYY-MM-DDTHH:MI:SS) - can use TO_DATE"
            elif len(date_str.split('-')) == 3:
                return "Standard date (YYYY-MM-DD) - can use TO_DATE"
        # Check for Oracle default format
        if '/' in date_str and date_str[:4].isdigit() and int(date_str[:4]) > 1900:
            return "Date with slashes (YYYY/MM/DD) - can use TO_DATE with format mask"
        return f"Non-standard format: '{date_str}' - use string operations"

    def _clean_sql(self, sql: str) -> str:
        """Clean and format the generated SQL"""
        # Remove markdown code blocks if present
        sql = sql.replace('```sql', '').replace('```', '')
        # Remove leading/trailing whitespace
        sql = sql.strip()
        # Don't add semicolon as per requirement
        if sql.endswith(';'):
            sql = sql[:-1].strip()
        return sql

    def _calculate_confidence(self, sql_query: str,
                            table_info: Dict[str, TableInfo],
                            similar_examples: List[Dict[str, Any]],
                            sample_data: Dict[str, Dict[str, Any]],
                            retry_context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate confidence score for generated SQL"""
        # Sample data quality score
        tables_with_samples = sum(1 for t in table_info if t in sample_data and sample_data[t].get('rows'))
        sample_data_quality = tables_with_samples / len(table_info) if table_info else 0
        
        # Similarity to examples score
        example_similarity = 0.5 # Default if no examples
        if similar_examples:
            # Simple check: do we use similar tables?
            sql_tables = self._extract_tables_from_sql(sql_query)
            example_tables = set()
            for ex in similar_examples[:3]:
                example_tables.update(ex.get('tables', []))
            
            if sql_tables and example_tables:
                overlap = len(sql_tables & example_tables) / len(sql_tables)
                example_similarity = overlap
        
        # Table count penalty
        table_count = len(table_info)
        table_count_penalty = 1.0 / (1 + 0.1 * max(0, table_count - 3))
        
        # Query complexity check
        has_join = 'JOIN' in sql_query.upper()
        has_where = 'WHERE' in sql_query.upper()
        has_group = 'GROUP BY' in sql_query.upper()
        
        complexity_score = 1.0
        if has_join and table_count > 5:
            complexity_score *= 0.8
        if has_group and not has_where:
            complexity_score *= 0.9
        
        # Retry penalty
        retry_penalty = 1.0
        if retry_context:
            # Lower confidence for retries
            retry_penalty = 0.8
            # Further lower if multiple failures
            if retry_context.get('previous_failures'):
                retry_penalty = 0.7
        
        # Calculate final confidence
        confidence = (
            0.3 * example_similarity +
            0.2 * table_count_penalty +
            0.2 * sample_data_quality +
            0.2 * complexity_score +
            0.1 * retry_penalty
        )
        
        return confidence

    def _extract_tables_from_sql(self, sql: str) -> set:
        """Extract table names from SQL query"""
        tables = set()
        # Simple extraction - could be improved with proper SQL parsing
        sql_upper = sql.upper()
        # Look for FROM clause
        from_idx = sql_upper.find('FROM')
        if from_idx != -1:
            # Extract until WHERE, GROUP BY, ORDER BY, or end
            end_keywords = ['WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'FETCH']
            end_idx = len(sql)
            for keyword in end_keywords:
                idx = sql_upper.find(keyword, from_idx)
                if idx != -1 and idx < end_idx:
                    end_idx = idx
            from_clause = sql[from_idx + 4:end_idx].strip()
            # Parse tables (simplified)
            parts = from_clause.split(',')
            for part in parts:
                # Remove aliases and joins
                part = part.strip()
                if ' ' in part:
                    part = part.split()[0]
                if part and not part.upper() in ['JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER', 'DUAL']:
                    tables.add(part)
        # Look for JOIN clauses
        join_keywords = ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN']
        for keyword in join_keywords:
            idx = 0
            while True:
                idx = sql_upper.find(keyword, idx)
                if idx == -1:
                    break
                # Extract table name after JOIN
                start = idx + len(keyword)
                end = sql_upper.find(' ', start + 1)
                if end == -1:
                    end = len(sql)
                table = sql[start:end].strip()
                if table and not table.upper() in ['ON', 'USING']:
                    tables.add(table)
                idx = end
        return tables

    def _generate_fallback_sql(self, table_info: Dict[str, TableInfo]) -> str:
        """Generate a simple fallback SQL query"""
        if not table_info:
            if self.dialect == 'oracle':
                return "SELECT 1 FROM DUAL"
            else:
                return "SELECT 1"
        # Use the first table
        first_table = list(table_info.keys())[0]
        if self.dialect == 'oracle':
            return f"SELECT * FROM {first_table} WHERE ROWNUM <= 10"
        elif self.dialect == 'mssql':
            return f"SELECT TOP 10 * FROM {first_table}"
        else:
            return f"SELECT * FROM {first_table} LIMIT 10"
        
    def _generate_join_clause(self, table1: str, table2: str, 
                            relationship_graph: Optional[Any] = None) -> str:
        """Generate JOIN clause using graph information"""
        if not relationship_graph:
            # Fallback to simple join
            return f"{table1} JOIN {table2}"
        
        # Get join template from graph
        template = relationship_graph.get_join_template(table1, table2)
        if template:
            if 'via_junction' in template:
                junction = template['via_junction']
                # Generate two joins through junction
                return f"{table1} JOIN {junction} ON {table1}.id = {junction}.{table1}_id JOIN {table2} ON {junction}.{table2}_id = {table2}.id"
            else:
                return f"{table1} JOIN {table2} ON {template['on_clause']}"
        
        # Fallback
        return f"{table1} JOIN {table2}"