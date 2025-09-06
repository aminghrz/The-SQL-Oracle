import json
import logging
import os
from openai import OpenAI
from models import Intent, QueryIntent, QueryComplexity

logger = logging.getLogger(__name__)

class IntentAnalyzer:
    def __init__(self, model: str = "gpt-4.1"):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL")
        )
        self.model = model
    
    def analyze_intent(self, user_prompt: str) -> Intent:
        """Analyze user intent using LLM"""
        logger.info(f"Analyzing intent for prompt: {user_prompt[:50]}...")
        
        # Add language detection hint
        prompt = f"""
        Analyze this SQL query request and extract the following information.
        Note: The query might be in Persian/Farsi or English. Focus on understanding the intent regardless of language.
        
        1. Intent type: Choose from [aggregation, comparison, trend, distribution, lookup]
        - aggregation: queries involving SUM, COUNT, AVG, MAX, MIN
        - comparison: comparing values between different entities
        - trend: time-based analysis, changes over time
        - distribution: grouping, categorization, breakdown by categories
        - lookup: simple data retrieval, finding specific records
        
        2. Entities: List of main tables/objects mentioned (e.g., customers, orders, products)
        3. Metrics: Values to calculate or retrieve (e.g., total_sales, average_price, count)
        4. Filters: Any conditions or constraints mentioned
        5. Time range: Extract any time-related constraints (e.g., "last month", "2023", "between Jan and Mar", "30 روز آخر" means "last 30 days")
        6. Expected result type: Choose from [single_value, list, aggregation]
        - single_value: expecting one number or value
        - list: expecting multiple rows/records
        - aggregation: expecting grouped or summarized data
        
        Query: "{user_prompt}"
        
        Key Persian/Farsi terms that might appear:
        - خطا/خطای = error
        - روز = day
        - آخر = last
        - تفکیک = breakdown/grouping
        - نوع = type
        - چند = how many/count
        
        Return ONLY a valid JSON object with this structure:
        {{
            "type": "one of the intent types listed above",
            "entities": ["list", "of", "entities"],
            "metrics": ["list", "of", "metrics"],
            "filters": ["list", "of", "filter conditions"],
            "time_range": "time range if mentioned, otherwise null",
            "expected_result": "one of the result types listed above",
            "confidence": 0.0 to 1.0,
            "complexity": 0.0 to 1.0
        }}
        """
        
        try:
            logger.info("Calling LLM for intent analysis...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes SQL queries in multiple languages and returns structured JSON responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                timeout=30.0  # Add timeout
            )
            
            logger.info("LLM response received")
            response_text = response.choices[0].message.content
        
            
            # Clean the response to ensure it's valid JSON
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse the JSON response
            parsed_response = json.loads(response_text)
            
            # Convert filters list to dict format expected by Intent
            filters_dict = {}
            if isinstance(parsed_response.get('filters'), list):
                # Convert list of filter strings to a dict
                for i, filter_str in enumerate(parsed_response.get('filters', [])):
                    filters_dict[f'filter_{i}'] = filter_str
            elif isinstance(parsed_response.get('filters'), dict):
                filters_dict = parsed_response.get('filters', {})
            
            # Determine complexity
            complexity_score = float(parsed_response.get('complexity', 0.5))
            query_complexity = QueryComplexity.SIMPLE if complexity_score < 0.5 else QueryComplexity.COMPLEX
            
            # Determine expected result booleans
            expected_result = parsed_response.get('expected_result', 'list')
            expects_single_value = expected_result == 'single_value'
            expects_list = expected_result == 'list'
            expects_aggregation = expected_result == 'aggregation'
            
            # Create Intent object with correct field names
            intent = Intent(
                intent_type=QueryIntent(parsed_response.get('type', 'lookup')),
                entities=parsed_response.get('entities', []),
                metrics=parsed_response.get('metrics', []),
                filters=filters_dict,
                time_range=parsed_response.get('time_range'),
                expected_result_type=expected_result,
                expects_single_value=expects_single_value,
                expects_list=expects_list,
                expects_aggregation=expects_aggregation,
                intent_confidence=float(parsed_response.get('confidence', 0.8)),
                query_complexity=query_complexity
            )
            
            logger.info(f"LLM Intent Analysis - Type: {intent.intent_type.value}, "
                       f"Entities: {intent.entities}, Metrics: {intent.metrics}")
            
            return intent
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response_text}")
            return self._create_fallback_intent(user_prompt)
        except Exception as e:
            logger.error(f"Error in LLM intent analysis: {e}")
            return self._create_fallback_intent(user_prompt)
    
    def _create_fallback_intent(self, user_prompt: str) -> Intent:
        """Create a basic fallback intent when LLM analysis fails"""
        # Simple keyword-based fallback
        user_prompt_lower = user_prompt.lower()
        
        intent_type = 'lookup'
        if any(word in user_prompt_lower for word in ['sum', 'total', 'count', 'average', 'avg']):
            intent_type = 'aggregation'
        elif any(word in user_prompt_lower for word in ['compare', 'versus', 'vs', 'difference']):
            intent_type = 'comparison'
        elif any(word in user_prompt_lower for word in ['trend', 'over time', 'growth', 'change']):
            intent_type = 'trend'
        elif any(word in user_prompt_lower for word in ['by', 'group by', 'distribution', 'breakdown']):
            intent_type = 'distribution'
        
        # Determine expected result type
        expected_result = 'list'
        expects_single_value = False
        expects_list = True
        expects_aggregation = False
        
        if intent_type == 'aggregation':
            expected_result = 'aggregation'
            expects_aggregation = True
            expects_list = False
            # Check if it's asking for a single value
            if any(word in user_prompt_lower for word in ['total', 'sum', 'count']) and \
               not any(word in user_prompt_lower for word in ['by', 'group', 'each']):
                expected_result = 'single_value'
                expects_single_value = True
                expects_aggregation = False
        
        # Simple entity extraction
        entities = []
        entity_keywords = ['actor', 'film', 'customer', 'payment', 'store', 'rental', 'category', 'inventory']
        for keyword in entity_keywords:
            if keyword in user_prompt_lower or keyword + 's' in user_prompt_lower:
                entities.append(keyword)
        
        # Simple metric extraction
        metrics = []
        metric_keywords = ['amount', 'total', 'count', 'average', 'sum', 'revenue', 'duration']
        for keyword in metric_keywords:
            if keyword in user_prompt_lower:
                metrics.append(keyword)
        
        return Intent(
            intent_type=QueryIntent(intent_type),
            entities=entities,
            metrics=metrics,
            filters={},  # Empty dict for filters
            time_range=None,
            expected_result_type=expected_result,
            expects_single_value=expects_single_value,
            expects_list=expects_list,
            expects_aggregation=expects_aggregation,
            intent_confidence=0.5,
            query_complexity=QueryComplexity.SIMPLE
        )