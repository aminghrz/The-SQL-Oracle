from typing import Dict, Any, List, Tuple
import json
import logging
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

class SchemaSummarizer:
    def __init__(self, embedding_service, persistence_manager, target_db: str, connection=None):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL")
        )
        self.embedding_service = embedding_service
        self.persistence_manager = persistence_manager
        self.target_db = target_db
        self.connection = connection

    def summarize_column(self, table_name: str, column_info: Dict[str, Any],
                        sample_values: List[Any], stats: Dict[str, Any],
                        peer_columns: List[Dict[str, Any]],
                        table_ddl: str = None,
                        table_sample_data: Dict[str, Any] = None) -> Tuple[Dict[str, Any], List[float]]:
        """Generate column summary with full table context."""
        
        # Filter and clean sample values
        clean_sample_values = self._clean_sample_values(sample_values)
        clean_table_sample = self._filter_sample_data_for_prompt(table_sample_data) if table_sample_data else None
        
        # Build comprehensive prompt with table context
        prompt = f"""Analyze this database column with its full table context.

    Table: {table_name}
    Column: {column_info['name']}
    Type: {column_info['type']}
    Nullable: {column_info.get('nullable', True)}

    Table DDL:
    {table_ddl if table_ddl else 'Not available'}

    Sample values from this column (first 10): {clean_sample_values[:10]}

    Full table sample (first 3 rows):
    {self._format_table_sample(clean_table_sample) if clean_table_sample else 'Not available'}

    Statistics:
    - Distinct count: {stats.get('distinct_count', 'unknown')}
    - Min: {stats.get('min', 'unknown')}
    - Max: {stats.get('max', 'unknown')}
    - Top values: {stats.get('top_values', [])}

    Other columns in this table: {[col['name'] for col in peer_columns[:10]]}

    Analyze the column considering:
    1. Its role within this specific table
    2. How it relates to other columns in the table
    3. Whether it appears to reference other tables
    4. Data patterns visible in the full table context

    IMPORTANT: 
    - Pay special attention to columns that might contain pre-aggregated counts
    - Column names like CNT, COUNT, TOTAL, NUM, QTY often contain pre-calculated values
    - Return ONLY valid JSON with no additional text

    Return a JSON object with these fields:
    {{
        "canonical_name": "snake_case English name (1-3 words)",
        "acronym_expansion": "full expansion if column name is acronym, else same as canonical",
        "aliases": ["list", "of", "alternative", "names"],
        "semantic_role": "one of: id, code, dimension, measure, time, flag, text, other",
        "unit": "unit of measurement if applicable, else null",
        "description": "brief description of what this column contains IN THIS TABLE'S CONTEXT",
        "value_domain": ["up to 10 example categories if categorical, else null"],
        "join_key_candidates": ["table.column suggestions if this looks like a foreign key"],
        "table_relationship": "how this column relates to the table's purpose",
        "peer_relationships": ["relationships with other columns in same table"],
        "pii": false,
        "confidence": 0.8
    }}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a database expert. Return only valid JSON with no additional text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=600,
                timeout=30.0
            )
            
            response_text = response.choices[0].message.content.strip()
            response_text = self._clean_json_response(response_text)
            
            try:
                summary = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for column {table_name}.{column_info['name']}: {e}")
                fixed_response = self._fix_json_response(response_text)
                summary = json.loads(fixed_response)

            # Create embedding text
            embedding_text = self._make_column_embedding_text(table_name, column_info['name'], summary)
            embedding = self.embedding_service.embed_text(embedding_text)

            # Save to persistence
            self.persistence_manager.save_column_summary(
                self.target_db,
                table_name,
                column_info['name'],
                summary,
                embedding.tolist(),
                summary['confidence']
            )

            return summary, embedding.tolist()

        except Exception as e:
            logger.error(f"Failed to summarize column {table_name}.{column_info['name']}: {e}")
            return None, None

    def _clean_sample_values(self, sample_values: List[Any]) -> List[Any]:
        """Clean sample values to avoid JSON issues."""
        clean_values = []
        for value in sample_values:
            if value is None:
                clean_values.append(None)
            elif isinstance(value, bytes):
                clean_values.append("<binary_data>")
            elif isinstance(value, str):
                # Truncate very long strings and remove problematic characters
                if len(value) > 100:
                    clean_value = value[:97] + "..."
                else:
                    clean_value = value
                # Remove non-printable characters except common whitespace
                clean_value = ''.join(c if ord(c) >= 32 or c in '\t\n\r' else '?' for c in clean_value)
                clean_values.append(clean_value)
            else:
                clean_values.append(value)
        return clean_values
        
    def _format_table_sample(self, sample_data: Dict[str, Any], max_rows: int = 3) -> str:
        """Format table sample data for prompt."""
        if not sample_data or not sample_data.get('rows'):
            return "No sample data available"
        
        rows = sample_data['rows'][:max_rows]
        columns = sample_data['columns']
        
        # Format as readable table
        lines = []
        lines.append(" | ".join(columns))
        lines.append("-" * (len(lines[0]) + 10))
        
        for row in rows:
            values = []
            for col in columns:
                val = row.get(col, 'NULL')
                if isinstance(val, str) and len(val) > 20:
                    val = val[:17] + "..."
                values.append(str(val))
            lines.append(" | ".join(values))
        
        return "\n".join(lines)
    
    def summarize_table(self, table_name: str, table_info: Dict[str, Any],
                    sample_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[float]]:
        """Generate table summary using already-generated column summaries."""
        # Get all column summaries for this table
        column_summaries = {}
        columns = table_info.columns if hasattr(table_info, 'columns') else table_info.get('columns', [])
        
        for col in columns:
            col_name = col.get('name') if isinstance(col, dict) else getattr(col, 'name', None)
            if col_name:
                col_key = f"{table_name}.{col_name}"
                col_summary = self.persistence_manager.get_column_summaries(self.target_db).get(col_key)
                if col_summary:
                    column_summaries[col_name] = col_summary['summary']
        
        # Build rich column analysis from summaries
        column_analysis = self._analyze_column_summaries(column_summaries)
        
        # Filter out problematic data types from sample data
        filtered_sample_data = self._filter_sample_data_for_prompt(sample_data)
        
        # Analyze sample for table type detection
        sample_analysis = self._analyze_sample_for_table_type(filtered_sample_data)
        
        prompt = f"""Analyze this database table using the detailed column summaries below.

    Table: {table_name}
    Row count: {len(sample_data.get('rows', [])) if sample_data else 'unknown'}

    Column Analysis from Summaries:
    {json.dumps(column_analysis, indent=2)}

    Sample Data Analysis:
    - Actual rows in sample: {sample_analysis['row_count']}
    - Is complete table (< 1000 rows): {sample_analysis['is_complete']}
    - Has numeric measures: {sample_analysis['has_measures']}
    - Has time columns: {sample_analysis['has_time']}
    - Foreign key count: {sample_analysis['fk_count']}
    - Unique identifier columns: {sample_analysis['id_count']}

    Sample Data (first 5 rows):
    {self._format_table_sample(filtered_sample_data, max_rows=5)}

    Based on the column summaries and their relationships, determine:
    1. The table's primary business purpose
    2. Table type classification:
    - catalog: Reference/lookup data, typically < 1000 rows, high reuse
    - reference: Static reference data, codes, types
    - dimension: Descriptive attributes, moderate rows
    - fact: Transactional data, measures, many rows
    - transaction: Event/log data, time-based
    - junction: Connects other tables, mostly FKs
    - configuration: System settings, parameters
    - other: Doesn't fit clear category
    3. Key relationships with other tables
    4. Common access patterns

    IMPORTANT: Return ONLY valid JSON. Do not include any explanatory text before or after the JSON.

    Return a JSON object with these fields:
    {{
    "purpose": "main business purpose based on column analysis",
    "table_type": "catalog|reference|dimension|fact|transaction|junction|configuration|other",
    "typical_use_cases": ["list of common use cases for this table"],
    "grain": "what each row represents",
    "key_columns": ["primary key columns based on column summaries"],
    "foreign_keys": ["table.column pairs identified from column summaries"],
    "measures": ["columns identified as measures from summaries"],
    "dimensions": ["columns identified as dimensions from summaries"],
    "common_filters": ["columns likely used in WHERE based on their roles"],
    "data_characteristics": {{
    "is_catalog": true/false,
    "is_time_series": true/false,
    "has_measures": true/false,
    "is_descriptive": true/false,
    "update_frequency": "static|daily|real-time|unknown"
    }},
    "quality_indicators": {{
    "completeness": "high|medium|low",
    "reliability": 0.0-1.0
    }},
    "data_quality_notes": "any patterns or issues noticed",
    "confidence": 0.8
    }}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a database expert. Analyze the column summaries to understand the table's purpose and classify its type. Return ONLY valid JSON with no additional text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800,
                timeout=30.0
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Clean the response more thoroughly
            response_text = self._clean_json_response(response_text)
            
            try:
                summary = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for table {table_name}: {e}")
                logger.error(f"Response text: {response_text[:500]}...")
                # Try to fix common JSON issues
                fixed_response = self._fix_json_response(response_text)
                summary = json.loads(fixed_response)
            
            # Create embedding text incorporating column insights and table type
            embedding_text_parts = [
                table_name,
                summary.get('purpose', ''),
                summary.get('grain', ''),
                summary.get('table_type', ''),
                f"use_cases: {' '.join(summary.get('typical_use_cases', [])[:3])}"
            ]
            
            # Add key column descriptions
            for col_name, col_summary in column_summaries.items():
                if isinstance(col_summary, dict):
                    embedding_text_parts.append(col_summary.get('description', ''))
                    embedding_text_parts.append(col_summary.get('canonical_name', col_name))
            
            embedding_text = " ".join(filter(None, embedding_text_parts))
            embedding = self.embedding_service.embed_text(embedding_text).tolist()
            
            # Save to persistence
            self.persistence_manager.save_table_summary(
                self.target_db,
                table_name,
                summary,
                embedding,
                summary.get('confidence', 0.8)
            )
            
            logger.info(f"Successfully generated table summary for {table_name} (type: {summary.get('table_type')})")
            return summary, embedding
            
        except Exception as e:
            logger.error(f"Failed to summarize table {table_name}: {e}")
            return None, None

    def _analyze_sample_for_table_type(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sample data to help determine table type"""
        if not sample_data or not sample_data.get('rows'):
            return {
                'row_count': 0,
                'is_complete': True,
                'has_measures': False,
                'has_time': False,
                'fk_count': 0,
                'id_count': 0
            }
        
        rows = sample_data['rows']
        columns = sample_data['columns']
        row_count = len(rows)
        
        analysis = {
            'row_count': row_count,
            'is_complete': row_count < 1000,
            'has_measures': False,
            'has_time': False,
            'fk_count': 0,
            'id_count': 0
        }
        
        # Analyze columns
        for col in columns:
            col_lower = col.lower()
            
            # Check for measures (numeric columns)
            if rows and rows[0].get(col) is not None:
                if isinstance(rows[0].get(col), (int, float)):
                    analysis['has_measures'] = True
            
            # Check for time columns
            if any(time_word in col_lower for time_word in ['date', 'time', 'created', 'updated']):
                analysis['has_time'] = True
            
            # Count potential foreign keys
            if col_lower.endswith('_id') or col_lower.endswith('_code'):
                analysis['fk_count'] += 1
            
            # Count ID columns
            if col_lower in ['id', 'uid', 'guid'] or col_lower.endswith('_id'):
                analysis['id_count'] += 1
        
        return analysis
    
    def _filter_sample_data_for_prompt(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out problematic data types from sample data for prompt generation."""
        if not sample_data or not sample_data.get('rows'):
            return sample_data
        
        filtered_data = {
            'columns': [],
            'rows': []
        }
        
        # Identify problematic columns (BLOB, CLOB, etc.)
        safe_columns = []
        for col in sample_data.get('columns', []):
            # Skip columns that might contain binary data or very long text
            is_safe = True
            for row in sample_data['rows'][:3]:  # Check first few rows
                value = row.get(col)
                if value is not None:
                    # Skip if value looks like binary data or is extremely long
                    if isinstance(value, bytes):
                        is_safe = False
                        break
                    elif isinstance(value, str) and len(value) > 1000:
                        is_safe = False
                        break
                    # Check for non-printable characters that might break JSON
                    elif isinstance(value, str) and any(ord(c) < 32 and c not in '\t\n\r' for c in value):
                        is_safe = False
                        break
            
            if is_safe:
                safe_columns.append(col)
        
        filtered_data['columns'] = safe_columns
        
        # Filter rows to only include safe columns
        for row in sample_data['rows']:
            filtered_row = {}
            for col in safe_columns:
                value = row.get(col)
                # Additional safety check
                if isinstance(value, str) and len(value) > 200:
                    filtered_row[col] = value[:197] + "..."
                else:
                    filtered_row[col] = value
            filtered_data['rows'].append(filtered_row)
        
        return filtered_data

    def _clean_json_response(self, response_text: str) -> str:
        """Clean JSON response from LLM."""
        # Remove markdown code blocks
        if "```json" in response_text:
            start_idx = response_text.find("```json") + 7
            end_idx = response_text.find("```", start_idx)
            if end_idx > start_idx:
                response_text = response_text[start_idx:end_idx].strip()
        elif "```" in response_text:
            parts = response_text.split("```")
            if len(parts) >= 2:
                response_text = parts[1].strip()
        
        # Remove any text before the first {
        first_brace = response_text.find('{')
        if first_brace > 0:
            response_text = response_text[first_brace:]
        
        # Remove any text after the last }
        last_brace = response_text.rfind('}')
        if last_brace > 0:
            response_text = response_text[:last_brace + 1]
        
        return response_text.strip()

    def _fix_json_response(self, response_text: str) -> str:
        """Attempt to fix common JSON issues."""
        try:
            # Try to fix unterminated strings by adding closing quotes
            lines = response_text.split('\n')
            fixed_lines = []
            
            for line in lines:
                # If line has an unterminated string, try to fix it
                if line.count('"') % 2 == 1 and not line.strip().endswith(','):
                    # Add closing quote and comma if needed
                    if ':' in line and not line.strip().endswith('"'):
                        line = line.rstrip() + '"'
                    if not line.strip().endswith(',') and not line.strip().endswith('}'):
                        line = line + ','
                fixed_lines.append(line)
            
            fixed_response = '\n'.join(fixed_lines)
            
            # Try to balance braces
            open_braces = fixed_response.count('{')
            close_braces = fixed_response.count('}')
            
            if open_braces > close_braces:
                fixed_response += '}' * (open_braces - close_braces)
            
            return fixed_response
            
        except Exception as e:
            logger.error(f"Failed to fix JSON response: {e}")
            # Return a minimal valid JSON as fallback
            return '{"purpose": "unknown", "table_type": "other", "grain": "unknown", "key_columns": [], "foreign_keys": [], "measures": [], "dimensions": [], "common_filters": [], "data_quality_notes": "analysis failed", "confidence": 0.3}'
    
    def _analyze_column_summaries(self, column_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze column summaries to extract patterns and insights."""
        analysis = {
            "total_columns": len(column_summaries),
            "columns_by_role": {},
            "foreign_key_candidates": [],
            "measures": [],
            "dimensions": [],
            "identifiers": [],
            "time_columns": [],
            "text_columns": [],
            "flags": [],
            "pii_columns": [],
            "column_details": []
        }
        for col_name, summary in column_summaries.items():
            if not isinstance(summary, dict):
                continue

            # Categorize by semantic role
            role = summary.get('semantic_role', 'other')
            if role not in analysis['columns_by_role']:
                analysis['columns_by_role'][role] = []
            analysis['columns_by_role'][role].append(col_name)

            # Collect specific types
            if role == 'measure':
                analysis['measures'].append({
                    'name': col_name,
                    'canonical': summary.get('canonical_name'),
                    'unit': summary.get('unit'),
                    'description': summary.get('description')
                })
            elif role == 'dimension':
                analysis['dimensions'].append({
                    'name': col_name,
                    'canonical': summary.get('canonical_name'),
                    'description': summary.get('description')
                })
            elif role == 'id':
                analysis['identifiers'].append(col_name)
            elif role == 'time':
                analysis['time_columns'].append(col_name)
            elif role == 'flag':
                analysis['flags'].append(col_name)
            elif role == 'text':
                analysis['text_columns'].append(col_name)

            # Check for foreign keys
            fk_candidates = summary.get('join_key_candidates', [])
            if fk_candidates:
                for fk in fk_candidates:
                    analysis['foreign_key_candidates'].append({
                        'column': col_name,
                        'references': fk,
                        'confidence': summary.get('confidence', 0.5)
                    })

            # Check for PII
            if summary.get('pii', False):
                analysis['pii_columns'].append(col_name)

            # Add detailed column info
            analysis['column_details'].append({
                'name': col_name,
                'canonical_name': summary.get('canonical_name'),
                'role': role,
                'description': summary.get('description'),
                'table_relationship': summary.get('table_relationship'),
                'peer_relationships': summary.get('peer_relationships', [])
            })
        return analysis
    
    def _make_column_embedding_text(self, table: str, column: str, summary: Dict[str, Any]) -> str:
        """Create text for column embedding."""
        parts = [
            f"{table}.{column}",
            summary['canonical_name'],
            summary['acronym_expansion'],
            summary['semantic_role'],
            summary.get('description', ''),
            summary.get('table_relationship', '')
        ]
        if summary.get('aliases'):
            parts.extend(summary['aliases'][:3])
        if summary.get('unit'):
            parts.append(summary['unit'])
        # Add peer relationships for better context
        if summary.get('peer_relationships'):
            parts.extend(summary['peer_relationships'][:2])
        return " ".join(filter(None, parts))