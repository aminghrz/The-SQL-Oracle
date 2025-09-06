from typing import Dict, Tuple, Any
import json
import logging
from openai import OpenAI
from models import Intent, QueryResult
import os

logger = logging.getLogger(__name__)

class ResultValidatorLLM:
    def __init__(self, model: str = "gpt-4.1"):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL")
        )
        self.model = model
    
    def validate(self, result: QueryResult, intent: Intent, 
                user_prompt: str, sql_query: str) -> Tuple[float, Dict[str, Any]]:
        """Validate query results using LLM"""
        
        if result.error:
            return 0.0, {'execution_failed': True, 'error': result.error}
        
        prompt = self._build_validation_prompt(result, intent, user_prompt, sql_query)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data quality expert who validates SQL query results."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            
            # Check for None response
            if not response_text:
                logger.error("Empty response from LLM validation")
                return self._basic_validation(result, intent)
            
            response_text = response_text.strip()
            
            # Clean JSON
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            validation_result = json.loads(response_text.strip())
            
            score = float(validation_result.get('validation_score', 0.5))
            checks = validation_result.get('checks', {})
            issues = validation_result.get('issues', [])
            suggestions = validation_result.get('suggestions', [])
            
            # Log validation details
            logger.info(f"Validation score: {score:.2f}")
            if issues:
                logger.warning(f"Validation issues: {', '.join(issues)}")
            
            return score, {
                'checks': checks,
                'issues': issues,
                'suggestions': suggestions,
                'retry_strategy': validation_result.get('retry_strategy', 'none')
            }
            
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            # Fallback to basic validation
            return self._basic_validation(result, intent)
    
    def _build_validation_prompt(self, result: QueryResult, intent: Intent,
                                 user_prompt: str, sql_query: str) -> str:
        """Build validation prompt"""
        
        # Get sample of results
        df = result.data
        sample_size = min(5, len(df))
        sample_data = df.head(sample_size).to_dict('records') if not df.empty else []
        
        prompt_parts = [
            f"Validate these SQL query results:",
            f"\nOriginal Question: \"{user_prompt}\"",
            f"\nQuery Intent:",
            f"- Type: {intent.intent_type.value}",
            f"- Expected result: {intent.expected_result_type}",
            f"- Entities: {intent.entities}",
            f"- Metrics: {intent.metrics}",
            f"\nSQL Query executed:",
            f"{sql_query}",
            f"\nResults Summary:",
            f"- Row count: {len(df)}",
            f"- Columns: {list(df.columns) if not df.empty else 'No results'}",
        ]
        
        if sample_data:
            prompt_parts.append(f"\nSample data (first {sample_size} rows):")
            prompt_parts.append(json.dumps(sample_data, indent=2, default=str))
        
        prompt_parts.extend([
            "\nValidation Checklist:",
            "1. Does the result answer the original question?",
            "2. Are the expected entities/metrics present in results?",
            "3. Is the row count reasonable for the query type?",
            "4. Are there any obvious data quality issues (nulls, duplicates)?",
            "5. Do the data types match what's expected?",
            "",
            "Return a JSON object with:",
            "{",
            '  "validation_score": 0.0-1.0,',
            '  "checks": {',
            '    "answers_question": true/false,',
            '    "has_expected_data": true/false,',
            '    "row_count_reasonable": true/false,',
            '    "data_quality_good": true/false',
            '  },',
            '  "issues": ["list of specific problems found"],',
            '  "suggestions": ["list of suggestions to fix issues"],',
            '  "retry_strategy": "none" | "add_filters" | "change_tables" | "modify_aggregation"',
            "}"
        ])
        
        return '\n'.join(prompt_parts)
    
    def _basic_validation(self, result: QueryResult, intent: Intent) -> Tuple[float, Dict[str, Any]]:
        """Basic fallback validation"""
        df = result.data
        checks = {
            'not_empty': len(df) > 0,
            'has_columns': len(df.columns) > 0 if not df.empty else False,
            'reasonable_size': len(df) < 10000
        }
        
        score = sum(checks.values()) / len(checks)
        
        return score, {
            'checks': checks,
            'issues': ['Basic validation only'],
            'suggestions': [],
            'retry_strategy': 'none' if score > 0.5 else 'change_tables'
        }