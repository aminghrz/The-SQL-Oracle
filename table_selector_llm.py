from typing import List, Tuple, Dict, Any, Optional
import json
import logging
import os
from openai import OpenAI
from models import Intent

logger = logging.getLogger(__name__)

class TableSelectorLLM:
    def __init__(self, model: str = "gpt-4.1"):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL")
        )
        self.model = model
    
    def select_tables(self, user_prompt: str, intent: Intent,
                    all_tables_info: Dict[str, Any],
                    similar_queries: List[Dict[str, Any]],
                    retry_context: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Use LLM to select relevant tables with semantic search and quality filtering."""
        # Store intent and retry context for use in semantic search
        self._current_intent = intent
        self._retry_context = retry_context
        self._column_match_details = {}
        self._table_quality_details = {}
        
        # Store reference to sql_agent if available (for accessing sample cache)
        if hasattr(self, 'sql_agent'):
            self._sql_agent_ref = self.sql_agent
        
        # First, do semantic search with quality filtering
        semantic_candidates = self._semantic_search(user_prompt, all_tables_info)
        
        # Apply progressive filtering based on iteration
        iteration = retry_context.get('iteration', 0) if retry_context else 0
        filtered_candidates = self._apply_iteration_based_filtering(
            semantic_candidates, iteration, retry_context
        )
        
        # If this is a retry, adjust confidence scores for previously failed tables
        if retry_context and retry_context.get('previous_tables'):
            # Penalize previously failed tables in semantic candidates
            penalized_candidates = []
            for table_name, similarity in filtered_candidates:
                if table_name in retry_context['previous_tables']:
                    # Reduce confidence significantly for failed tables
                    penalized_similarity = similarity * 0.3
                    logger.info(f"Penalizing previously failed table {table_name}: {similarity:.2f} -> {penalized_similarity:.2f}")
                    penalized_candidates.append((table_name, penalized_similarity))
                else:
                    penalized_candidates.append((table_name, similarity))
            filtered_candidates = penalized_candidates
        
        # Build prompt with retry context and quality details
        prompt = self._build_prompt_with_quality(user_prompt, intent, all_tables_info,
                                            similar_queries, filtered_candidates, retry_context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_retry_aware_system_prompt(retry_context)},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4 if retry_context else 0.1,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            if not response_text:
                logger.error("Empty response from LLM")
                return self._quality_aware_fallback_selection(intent, all_tables_info, retry_context)
            
            response_text = response_text.strip()
            
            # Clean JSON response
            if "```json" in response_text:
                start_idx = response_text.find("```json") + 7
                end_idx = response_text.find("```", start_idx)
                if end_idx > start_idx:
                    response_text = response_text[start_idx:end_idx].strip()
            elif "```" in response_text:
                parts = response_text.split("```")
                if len(parts) >= 2:
                    response_text = parts[1].strip()
            
            # Ensure the JSON is complete
            if response_text.count('{') > response_text.count('}'):
                response_text += '"}]}'
            
            # Try to parse JSON
            try:
                parsed_response = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.error(f"Response was: {response_text[:500]}...")
                return self._extract_tables_from_text(response_text, intent, all_tables_info, retry_context)
            
            # Extract selected tables
            selected_tables = []
            for table in parsed_response.get('selected_tables', []):
                table_name = table.get('name')
                confidence = float(table.get('confidence', 0.5))
                reason = table.get('reason', '')
                
                # Apply penalty if this table was previously selected and failed
                if retry_context and table_name in retry_context.get('previous_tables', []):
                    original_confidence = confidence
                    confidence *= 0.3  # Significant penalty
                    logger.warning(f"Penalizing {table_name} due to previous failure: {original_confidence:.2f} -> {confidence:.2f}")
                
                if table_name:
                    selected_tables.append((table_name, confidence))
                    logger.info(f"Selected table '{table_name}' (confidence: {confidence:.2f}): {reason}")
            
            if not selected_tables:
                logger.warning("No tables selected by LLM, using fallback")
                return self._quality_aware_fallback_selection(intent, all_tables_info, retry_context)
            
            # Sort by confidence and filter out very low confidence tables on retry
            if retry_context:
                # On retry, only keep tables with reasonable confidence
                selected_tables = [(t, c) for t, c in selected_tables if c > 0.4]
                if not selected_tables:
                    logger.warning("All selected tables have low confidence after retry penalty")
                    # Force selection of different tables
                    return self._force_alternative_tables_with_quality(intent, all_tables_info, retry_context)
            
            selected_tables.sort(key=lambda x: x[1], reverse=True)
            
            if hasattr(self, 'relationship_graph'):
                selected_tables = self._enhance_with_graph_info(selected_tables, all_tables_info)
            
            # Clean up temporary attributes
            self._current_intent = None
            self._retry_context = None
            
            return selected_tables
            
        except Exception as e:
            logger.error(f"LLM table selection failed: {e}")
            return self._quality_aware_fallback_selection(intent, all_tables_info, retry_context)

    def _apply_iteration_based_filtering(self, candidates: List[Tuple[str, float]], 
                                    iteration: int, retry_context: Optional[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Apply progressive filtering based on iteration"""
        if not hasattr(self, '_table_quality_details'):
            return candidates
        
        filtered = []
        
        for table_name, score in candidates:
            quality_info = self._table_quality_details.get(table_name, {})
            quality_score = quality_info.get('quality_score', 0.5)
            table_type = quality_info.get('table_type', 'unknown')
            
            # Progressive quality thresholds
            if iteration <= 1:
                # Strict quality for first attempts
                if quality_score >= 0.5 or table_type in ['catalog', 'reference']:
                    filtered.append((table_name, score))
            elif iteration == 2:
                # Relax quality threshold
                if quality_score >= 0.3 or table_type in ['catalog', 'reference', 'dimension']:
                    filtered.append((table_name, score))
            elif iteration == 3:
                # Include more table types
                if quality_score >= 0.2 or table_type in ['catalog', 'reference', 'dimension', 'junction']:
                    filtered.append((table_name, score))
            else:
                # Last resort - any table with some data
                if quality_score > 0.1:
                    filtered.append((table_name, score))
        
        # Ensure we have at least some candidates
        if not filtered and candidates:
            # Take top 5 regardless of quality
            filtered = candidates[:5]
        
        return filtered

    def _build_prompt_with_quality(self, user_prompt: str, intent: Intent,
                                all_tables_info: Dict[str, Any],
                                similar_queries: List[Dict[str, Any]],
                                semantic_candidates: List[Tuple[str, float]],
                                retry_context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt with quality information"""
        prompt_parts = [
            f"Select the most relevant database tables for this query: \"{user_prompt}\"",
            f"\nQuery Intent Analysis:",
            f"- Type: {intent.intent_type.value}",
            f"- Entities mentioned: {', '.join(intent.entities) if intent.entities else 'none identified'}",
            f"- Metrics requested: {', '.join(intent.metrics) if intent.metrics else 'none identified'}",
            f"- Expected result: {intent.expected_result_type}",
        ]
        
        # Add detailed retry context if available
        if retry_context:
            prompt_parts.extend([
                f"\n🚨 CRITICAL: This is retry attempt #{retry_context['iteration']}",
                f"\n❌ FAILED TABLES (DO NOT SELECT THESE AGAIN):"
            ])
            
            # List all previously failed tables prominently
            for failed_table in retry_context.get('previous_tables', []):
                prompt_parts.append(f" ❌ {failed_table} - RETURNED NO RESULTS")
            
            # Add all previous failure details
            prompt_parts.append(f"\nFailure Analysis:")
            for i, failure in enumerate(retry_context.get('previous_failures', [])):
                prompt_parts.extend([
                    f"\nAttempt {failure['iteration']}:",
                    f"- Tables used: {', '.join(failure['tables'])}",
                    f"- Issues: {'; '.join(failure['issues'][:3])}"
                ])
            
            # Add the most recent failed query if available
            if retry_context.get('failed_query'):
                prompt_parts.extend([
                    f"\nFailed SQL Query:",
                    f"{retry_context['failed_query'][:200]}...",
                    f"Result: {retry_context['failed_result']['row_count']} rows"
                ])
            
            prompt_parts.extend([
                "\n⚠️ RETRY REQUIREMENTS:",
                "1. You MUST select DIFFERENT tables than the failed ones listed above",
                "2. The data might be in tables with different naming conventions",
                "3. Consider tables that might have the data under different column names",
                "4. Look for junction tables or related tables that might contain the information",
                "5. DO NOT select any table marked with ❌ above",
                ""
            ])
        
        # Add semantic search results with quality indicators
        if semantic_candidates:
            prompt_parts.append("\nSemantically similar tables (with quality assessment):")
            shown_count = 0
            for table_name, similarity in semantic_candidates[:10]:
                # Skip failed tables in semantic results
                if retry_context and table_name in retry_context.get('previous_tables', []):
                    continue
                
                # Get table summary
                summary_info = ""
                if hasattr(self, 'table_summaries') and table_name in self.table_summaries:
                    summary = self.table_summaries[table_name]['summary']
                    summary_info = f" - {summary.get('purpose', 'No description')}"
                
                # Add quality indicators
                quality_info = ""
                if hasattr(self, '_table_quality_details') and table_name in self._table_quality_details:
                    quality_data = self._table_quality_details[table_name]
                    quality_category = quality_data['quality_category']
                    table_type = quality_data['table_type']
                    quality_info = f" [Type: {table_type}, Quality: {quality_category}]"
                
                # Add column match details
                column_info = ""
                if hasattr(self, '_column_match_details') and table_name in self._column_match_details:
                    matched_cols = self._column_match_details[table_name]
                    if matched_cols:
                        top_cols = matched_cols[:3]
                        col_details = []
                        for col in top_cols:
                            col_details.append(f"{col['name']} ({col['semantic_role']}, score: {col['score']:.2f})")
                        column_info = f"\n Matching columns: {', '.join(col_details)}"
                
                prompt_parts.append(f"- {table_name} (similarity: {similarity:.2f}){quality_info}{summary_info}{column_info}")
                shown_count += 1
                if shown_count >= 5:
                    break
        
        # Add similar query examples
        if similar_queries:
            prompt_parts.append("\nSimilar successful queries used these tables:")
            for i, example in enumerate(similar_queries[:3]):
                prompt_parts.append(f"{i+1}. \"{example['prompt']}\" -> Tables: {', '.join(example['tables'])}")
        
        # Add available tables with quality indicators
        prompt_parts.append(f"\nAvailable tables in database ({len(all_tables_info)} total):")
        
        # If retry, show previously failed tables first with warning
        if retry_context and retry_context.get('previous_tables'):
            prompt_parts.append("\n❌ FAILED TABLES (DO NOT SELECT):")
            for table_name in retry_context['previous_tables']:
                if table_name in all_tables_info:
                    table_info = all_tables_info[table_name]
                    columns = table_info.columns[:5]
                    col_names = [col['name'] for col in columns]
                    
                    # Add quality warning if available
                    quality_warning = ""
                    if hasattr(self, '_table_quality_details') and table_name in self._table_quality_details:
                        quality_data = self._table_quality_details[table_name]
                        if quality_data['quality_category'] in ['low', 'very_low']:
                            quality_warning = f" [LOW QUALITY: {quality_data['quality_category']}]"
                    
                    prompt_parts.append(f"- ❌ {table_name}: {', '.join(col_names)}... [FAILED - NO RESULTS]{quality_warning}")
        
        # Show other tables with quality indicators
        prompt_parts.append("\n✅ OTHER AVAILABLE TABLES (CONSIDER THESE):")
        shown_count = 0
        
        # Group tables by quality/type for better organization
        high_quality_tables = []
        catalog_tables = []
        medium_quality_tables = []
        other_tables = []
        
        for table_name, table_info in all_tables_info.items():
            # Skip if already shown as failed
            if retry_context and table_name in retry_context.get('previous_tables', []):
                continue
            
            quality_data = self._table_quality_details.get(table_name, {})
            quality_category = quality_data.get('quality_category', 'unknown')
            table_type = quality_data.get('table_type', 'unknown')
            
            if table_type in ['catalog', 'reference']:
                catalog_tables.append((table_name, table_info))
            elif quality_category == 'high':
                high_quality_tables.append((table_name, table_info))
            elif quality_category == 'medium':
                medium_quality_tables.append((table_name, table_info))
            else:
                other_tables.append((table_name, table_info))
        
        # Show tables in priority order
        all_grouped_tables = [
            ("⭐ HIGH QUALITY TABLES:", high_quality_tables),
            ("📚 CATALOG/REFERENCE TABLES:", catalog_tables),
            ("✓ MEDIUM QUALITY TABLES:", medium_quality_tables),
            ("OTHER TABLES:", other_tables)
        ]
        
        for group_label, tables in all_grouped_tables:
            if tables and shown_count < 40:
                prompt_parts.append(f"\n{group_label}")
                for table_name, table_info in tables[:10]:
                    if shown_count >= 40:
                        break
                    
                    columns = table_info.columns[:5]
                    
                    # Build column description with match indicators
                    col_descriptions = []
                    has_column_match = False
                    for col in columns:
                        col_name = col['name']
                        col_desc = col_name
                        
                        # Check if this column matched in semantic search
                        col_key = f"{table_name}.{col_name}"
                        if hasattr(self, '_column_match_details') and table_name in self._column_match_details:
                            for matched_col in self._column_match_details[table_name]:
                                if matched_col['name'] == col_name:
                                    col_desc = f"{col_name} ⭐"
                                    has_column_match = True
                                    break
                        
                        # Check column summaries
                        if hasattr(self, 'column_summaries'):
                            if col_key in self.column_summaries:
                                col_summary = self.column_summaries[col_key]['summary']
                                canonical = col_summary.get('canonical_name')
                                if canonical and canonical != col_name:
                                    col_desc = f"{col_name} ({canonical})"
                                
                                # Add semantic role
                                role = col_summary.get('semantic_role')
                                if role and role != 'other':
                                    col_desc += f" [{role}]"
                        
                        col_descriptions.append(col_desc)
                    
                    # Add table with enhanced description
                    table_desc = f"- {table_name}: {', '.join(col_descriptions)}..."
                    if has_column_match:
                        table_desc = "⭐ " + table_desc
                    
                    # Add table purpose from summaries
                    if hasattr(self, 'table_summaries') and table_name in self.table_summaries:
                        table_summary = self.table_summaries[table_name]['summary']
                        purpose = table_summary.get('purpose')
                        if purpose:
                            table_desc += f" // {purpose}"
                    
                    # Add quality indicator
                    quality_data = self._table_quality_details.get(table_name, {})
                    if quality_data:
                        table_desc += f" [Quality: {quality_data['quality_category']}]"
                    
                    prompt_parts.append(table_desc)
                    shown_count += 1
        
        prompt_parts.extend([
            "\nInstructions:",
            "1. Select ONLY the tables necessary to answer the query",
            "2. Include tables needed for JOINs to connect the data",
            "3. Consider the query intent type when selecting tables",
            "4. For aggregations, include fact tables with metrics",
            "5. Use the semantic similarity scores and table purposes as hints",
            "6. Pay special attention to tables with ⭐ (column matches) indicators",
            "7. Consider canonical column names when matching to user's query terms",
            "8. Consider table quality - prefer HIGH/MEDIUM quality tables unless CATALOG tables are needed",
            "9. For lookup queries, CATALOG/REFERENCE tables can be valuable even with few rows",
            "10. Maximum 10 tables (prefer fewer if possible)",
        ])
        
        if retry_context:
            prompt_parts.extend([
                "\n🚨 CRITICAL RETRY RULES:",
                "11. DO NOT select any table marked with ❌ (they returned no results)",
                "12. You MUST choose DIFFERENT tables than previous attempts",
                "13. Consider that the data might be named differently than expected",
                "14. Look for alternative tables that might contain user/level information",
                "15. Be creative - the data might be in unexpected places",
                "16. Consider CATALOG or REFERENCE tables that might contain the lookup data"
            ])
        
        prompt_parts.extend([
            "",
            "Return a JSON object with this structure:",
            "{",
            ' "selected_tables": [',
            ' {"name": "table_name", "confidence": 0.9, "reason": "contains customer data mentioned in query, has matching columns: customer_name (dimension), total_amount (measure), high quality table"},',
            ' {"name": "another_table", "confidence": 0.7, "reason": "catalog table with user roles, might contain support manager information"}',
            " ]",
            "}"
        ])
        
        return '\n'.join(prompt_parts)

    def _quality_aware_fallback_selection(self, intent: Intent, all_tables_info: Dict[str, Any],
                                        retry_context: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Fallback selection with quality awareness"""
        selected = []
        previous_tables = set(retry_context.get('previous_tables', [])) if retry_context else set()
        
        for table_name in all_tables_info:
            # Skip previously failed tables
            if table_name in previous_tables:
                continue
            
            score = 0.0
            table_lower = table_name.lower()
            
            # Get quality info
            quality_data = self._table_quality_details.get(table_name, {})
            quality_score = quality_data.get('quality_score', 0.5)
            table_type = quality_data.get('table_type', 'unknown')
            
            # Skip very low quality tables unless they're catalogs
            if quality_score < 0.2 and table_type not in ['catalog', 'reference']:
                continue
            
            # Check entity matches
            for entity in intent.entities:
                if entity.lower() in table_lower or table_lower in entity.lower():
                    score = 0.8
                    break
            
            # Intent-based scoring
            if intent.intent_type.value == 'lookup' and table_type in ['catalog', 'reference']:
                score = max(score, 0.7)
            elif intent.intent_type.value == 'aggregation' and table_type == 'fact':
                score = max(score, 0.6)
            elif 'payment' in table_lower and any(m in ['payment', 'revenue', 'amount'] for m in intent.metrics):
                score = max(score, 0.7)
            
            # Apply quality factor
            if score > 0:
                score *= (0.7 + 0.3 * quality_score)  # Quality affects 30% of score
                selected.append((table_name, score))
        
        # If nothing selected, try catalog tables for lookup
        if not selected and intent.intent_type.value == 'lookup':
            for table_name in all_tables_info:
                if table_name in previous_tables:
                    continue
                quality_data = self._table_quality_details.get(table_name, {})
                if quality_data.get('table_type') in ['catalog', 'reference']:
                    selected.append((table_name, 0.5))
        
        # Last resort
        if not selected:
            for table_name in list(all_tables_info.keys())[:5]:
                if table_name not in previous_tables:
                    selected.append((table_name, 0.3))
        
        return sorted(selected, key=lambda x: x[1], reverse=True)[:10]

    def _force_alternative_tables_with_quality(self, intent: Intent, all_tables_info: Dict[str, Any],
                                            retry_context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Force selection of alternative tables with quality consideration"""
        logger.info("Forcing alternative table selection with quality awareness")
        previous_tables = set(retry_context.get('previous_tables', []))
        alternatives = []
        
        # Get all tables except previously failed ones
        for table_name in all_tables_info:
            if table_name in previous_tables:
                continue
            
            # Get quality info
            quality_data = self._table_quality_details.get(table_name, {})
            quality_score = quality_data.get('quality_score', 0.5)
            table_type = quality_data.get('table_type', 'unknown')
            
            score = 0.5  # Base score for alternatives
            
            # Boost based on table type for intent
            if intent.intent_type.value == 'lookup':
                if table_type in ['catalog', 'reference']:
                    score += 0.3
                elif table_type == 'dimension':
                    score += 0.2
            elif intent.intent_type.value == 'aggregation':
                if table_type == 'fact':
                    score += 0.3
                elif table_type == 'transaction':
                    score += 0.2
            
            # Check for entity matches
            table_lower = table_name.lower()
            for entity in intent.entities:
                if entity.lower() in table_lower:
                    score += 0.2
            
            # Apply quality factor
            score *= (0.6 + 0.4 * quality_score)
            
            alternatives.append((table_name, min(score, 0.9)))
        
        # Sort by score and return top alternatives
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        if not alternatives:
            logger.error("No alternative tables available!")
            # Last resort: return any table that's not in previous_tables
            for table_name in list(all_tables_info.keys())[:5]:
                if table_name not in previous_tables:
                    alternatives.append((table_name, 0.3))
        
        return alternatives[:5]
    
    def _get_retry_aware_system_prompt(self, retry_context: Optional[Dict[str, Any]]) -> str:
        """Get system prompt that emphasizes retry requirements"""
        base_prompt = "You are a database expert who selects the most relevant tables for SQL queries. Always return valid JSON."
        
        if retry_context and retry_context.get('iteration', 0) > 0:
            return base_prompt + """
            
CRITICAL: This is a RETRY attempt. The previous table selection did NOT work.
- You MUST select DIFFERENT tables than before
- The previously selected tables returned NO results
- Consider that the data might be in completely different tables than expected
- Look for alternative table names that might contain the same information
- Be creative and consider non-obvious table choices"""
        
        return base_prompt
    
    def _force_alternative_tables(self, intent: Intent, all_tables_info: Dict[str, Any],
                                 retry_context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Force selection of alternative tables when all else fails"""
        logger.info("Forcing alternative table selection")
        
        previous_tables = set(retry_context.get('previous_tables', []))
        alternatives = []
        
        # Get all tables except previously failed ones
        for table_name in all_tables_info:
            if table_name not in previous_tables:
                score = 0.5  # Base score for alternatives
                
                # Boost score based on keywords
                table_lower = table_name.lower()
                
                # Check for entity matches
                for entity in intent.entities:
                    if entity.lower() in table_lower:
                        score += 0.2
                
                # Check for common patterns
                if any(keyword in table_lower for keyword in ['user', 'person', 'member', 'account']):
                    score += 0.1
                if any(keyword in table_lower for keyword in ['level', 'grade', 'rank', 'tier']):
                    score += 0.1
                
                alternatives.append((table_name, min(score, 0.9)))
        
        # Sort by score and return top alternatives
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        if not alternatives:
            logger.error("No alternative tables available!")
            # Last resort: return any table that's not in previous_tables
            for table_name in list(all_tables_info.keys())[:5]:
                if table_name not in previous_tables:
                    alternatives.append((table_name, 0.3))
        
        return alternatives[:5]  # Return top 5 alternatives
    
    def _fallback_selection(self, intent: Intent, all_tables_info: Dict[str, Any],
                           retry_context: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Simple fallback selection based on entity matching"""
        selected = []
        previous_tables = set(retry_context.get('previous_tables', [])) if retry_context else set()
        
        for table_name in all_tables_info:
            # Skip previously failed tables
            if table_name in previous_tables:
                continue
                
            score = 0.0
            table_lower = table_name.lower()
            
            # Check entity matches
            for entity in intent.entities:
                if entity.lower() in table_lower or table_lower in entity.lower():
                    score = 0.8
                    break
            
            # Common patterns
            if intent.intent_type.value == 'aggregation' and 'fact' in table_lower:
                score = max(score, 0.6)
            elif 'payment' in table_lower and any(m in ['payment', 'revenue', 'amount'] for m in intent.metrics):
                score = max(score, 0.7)
            
            if score > 0:
                selected.append((table_name, score))
        
        # If nothing selected and not a retry, return top 5 tables with low confidence
        if not selected and not retry_context:
            for table_name in list(all_tables_info.keys())[:5]:
                selected.append((table_name, 0.3))
        elif not selected and retry_context:
            # On retry, force different tables
            return self._force_alternative_tables(intent, all_tables_info, retry_context)
        
        return sorted(selected, key=lambda x: x[1], reverse=True)[:10]
    
    def _extract_tables_from_text(self, text: str, intent: Intent, all_tables_info: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Extract table names from text when JSON parsing fails"""
        import re
        
        selected_tables = []
        
        # Look for patterns like "main.table_name" or just "table_name"
        pattern = r'"name"\s*:\s*"([^"]+)"'
        matches = re.findall(pattern, text)
        
        for match in matches:
            if match in all_tables_info:
                selected_tables.append((match, 0.7))  # Default confidence when extracted from text
                logger.info(f"Extracted table from text: {match}")
        
        if not selected_tables:
            return self._fallback_selection(intent, all_tables_info)
        
        return selected_tables
    
    def _build_prompt(self, user_prompt: str, intent: Intent,
                    all_tables_info: Dict[str, Any],
                    similar_queries: List[Dict[str, Any]],
                    semantic_candidates: List[Tuple[str, float]],
                    retry_context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for table selection with column match details."""
        prompt_parts = [
            f"Select the most relevant database tables for this query: \"{user_prompt}\"",
            f"\nQuery Intent Analysis:",
            f"- Type: {intent.intent_type.value}",
            f"- Entities mentioned: {', '.join(intent.entities) if intent.entities else 'none identified'}",
            f"- Metrics requested: {', '.join(intent.metrics) if intent.metrics else 'none identified'}",
            f"- Expected result: {intent.expected_result_type}",
        ]
        
        # Add detailed retry context if available
        if retry_context:
            prompt_parts.extend([
                f"\n🚨 CRITICAL: This is retry attempt #{retry_context['iteration']}",
                f"\n❌ FAILED TABLES (DO NOT SELECT THESE AGAIN):"
            ])
            
            # List all previously failed tables prominently
            for failed_table in retry_context.get('previous_tables', []):
                prompt_parts.append(f" ❌ {failed_table} - RETURNED NO RESULTS")
            
            # Add all previous failure details
            prompt_parts.append(f"\nFailure Analysis:")
            for i, failure in enumerate(retry_context.get('previous_failures', [])):
                prompt_parts.extend([
                    f"\nAttempt {failure['iteration']}:",
                    f"- Tables used: {', '.join(failure['tables'])}",
                    f"- Issues: {'; '.join(failure['issues'][:3])}"
                ])
            
            # Add the most recent failed query if available
            if retry_context.get('failed_query'):
                prompt_parts.extend([
                    f"\nFailed SQL Query:",
                    f"{retry_context['failed_query'][:200]}...",
                    f"Result: {retry_context['failed_result']['row_count']} rows"
                ])
            
            prompt_parts.extend([
                "\n⚠️ RETRY REQUIREMENTS:",
                "1. You MUST select DIFFERENT tables than the failed ones listed above",
                "2. The data might be in tables with different naming conventions",
                "3. Consider tables that might have the data under different column names",
                "4. Look for junction tables or related tables that might contain the information",
                "5. DO NOT select any table marked with ❌ above",
                ""
            ])
        
        # Add semantic search results with column match details
        if semantic_candidates:
            prompt_parts.append("\nSemantically similar tables (based on table AND column analysis):")
            shown_count = 0
            for table_name, similarity in semantic_candidates[:10]:
                # Skip failed tables in semantic results
                if retry_context and table_name in retry_context.get('previous_tables', []):
                    continue
                
                # Get table summary
                summary_info = ""
                if hasattr(self, 'table_summaries') and table_name in self.table_summaries:
                    summary = self.table_summaries[table_name]['summary']
                    summary_info = f" - {summary.get('purpose', 'No description')}"
                
                # Add column match details
                column_info = ""
                if hasattr(self, '_column_match_details') and table_name in self._column_match_details:
                    matched_cols = self._column_match_details[table_name]
                    if matched_cols:
                        top_cols = matched_cols[:3] # Show top 3 matching columns
                        col_details = []
                        for col in top_cols:
                            col_details.append(f"{col['name']} ({col['semantic_role']}, score: {col['score']:.2f})")
                        column_info = f"\n  Matching columns: {', '.join(col_details)}"
                
                prompt_parts.append(f"- {table_name} (similarity: {similarity:.2f}){summary_info}{column_info}")
                shown_count += 1
                if shown_count >= 5:
                    break
        
        # Add similar query examples
        if similar_queries:
            prompt_parts.append("\nSimilar successful queries used these tables:")
            for i, example in enumerate(similar_queries[:3]):
                prompt_parts.append(f"{i+1}. \"{example['prompt']}\" -> Tables: {', '.join(example['tables'])}")
        
        # Add available tables with enhanced column information
        prompt_parts.append(f"\nAvailable tables in database ({len(all_tables_info)} total):")
        
        # If retry, show previously failed tables first with warning
        if retry_context and retry_context.get('previous_tables'):
            prompt_parts.append("\n❌ FAILED TABLES (DO NOT SELECT):")
            for table_name in retry_context['previous_tables']:
                if table_name in all_tables_info:
                    table_info = all_tables_info[table_name]
                    columns = table_info.columns[:5]
                    col_names = [col['name'] for col in columns]
                    prompt_parts.append(f"- ❌ {table_name}: {', '.join(col_names)}... [FAILED - NO RESULTS]")
        
        # Show other tables with column match indicators
        prompt_parts.append("\n✅ OTHER AVAILABLE TABLES (CONSIDER THESE):")
        shown_count = 0
        for table_name, table_info in all_tables_info.items():
            # Skip if already shown as failed
            if retry_context and table_name in retry_context.get('previous_tables', []):
                continue
            
            if shown_count >= 30: # Show more tables on retry
                break
            
            columns = table_info.columns[:5]
            
            # Build column description with match indicators
            col_descriptions = []
            has_column_match = False
            for col in columns:
                col_name = col['name']
                col_desc = col_name
                
                # Check if this column matched in semantic search
                col_key = f"{table_name}.{col_name}"
                if hasattr(self, '_column_match_details') and table_name in self._column_match_details:
                    for matched_col in self._column_match_details[table_name]:
                        if matched_col['name'] == col_name:
                            col_desc = f"{col_name} ⭐" # Star indicates semantic match
                            has_column_match = True
                            break
                
                # Check column summaries
                if hasattr(self, 'column_summaries'):
                    if col_key in self.column_summaries:
                        col_summary = self.column_summaries[col_key]['summary']
                        canonical = col_summary.get('canonical_name')
                        if canonical and canonical != col_name:
                            col_desc = f"{col_name} ({canonical})"
                        
                        # Add semantic role
                        role = col_summary.get('semantic_role')
                        if role and role != 'other':
                            col_desc += f" [{role}]"
                
                col_descriptions.append(col_desc)
            
            # Add table with enhanced description
            table_desc = f"- {table_name}: {', '.join(col_descriptions)}..."
            if has_column_match:
                table_desc = "⭐ " + table_desc # Highlight tables with column matches
            
            # Add table purpose from summaries
            if hasattr(self, 'table_summaries') and table_name in self.table_summaries:
                table_summary = self.table_summaries[table_name]['summary']
                purpose = table_summary.get('purpose')
                if purpose:
                    table_desc += f" // {purpose}"
            
            prompt_parts.append(table_desc)
            shown_count += 1
        
        prompt_parts.extend([
            "\nInstructions:",
            "1. Select ONLY the tables necessary to answer the query",
            "2. Include tables needed for JOINs to connect the data",
            "3. Consider the query intent type when selecting tables",
            "4. For aggregations, include fact tables with metrics",
            "5. Use the semantic similarity scores and table purposes as hints",
            "6. Pay special attention to tables with ⭐ (column matches) indicators",
            "7. Consider canonical column names when matching to user's query terms",
            "8. Maximum 10 tables (prefer fewer if possible)",
        ])
        
        if retry_context:
            prompt_parts.extend([
                "\n🚨 CRITICAL RETRY RULES:",
                "9. DO NOT select any table marked with ❌ (they returned no results)",
                "10. You MUST choose DIFFERENT tables than previous attempts",
                "11. Consider that the data might be named differently than expected",
                "12. Look for alternative tables that might contain user/level information",
                "13. Be creative - the data might be in unexpected places"
            ])
        
        prompt_parts.extend([
            "",
            "Return a JSON object with this structure:",
            "{",
            ' "selected_tables": [',
            ' {"name": "table_name", "confidence": 0.9, "reason": "contains customer data mentioned in query, has matching columns: customer_name (dimension), total_amount (measure)"},',
            ' {"name": "another_table", "confidence": 0.7, "reason": "needed to join customer with orders, column order_id matches"}',
            " ]",
            "}"
        ])
        
        return '\n'.join(prompt_parts)

    def _fallback_selection(self, intent: Intent, all_tables_info: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Simple fallback selection based on entity matching"""
        selected = []
        
        for table_name in all_tables_info:
            score = 0.0
            table_lower = table_name.lower()
            
            # Check entity matches
            for entity in intent.entities:
                if entity.lower() in table_lower or table_lower in entity.lower():
                    score = 0.8
                    break
            
            # Common patterns
            if intent.intent_type.value == 'aggregation' and 'fact' in table_lower:
                score = max(score, 0.6)
            elif 'payment' in table_lower and any(m in ['payment', 'revenue', 'amount'] for m in intent.metrics):
                score = max(score, 0.7)
            
            if score > 0:
                selected.append((table_name, score))
        
        # If nothing selected, return top 5 tables with low confidence
        if not selected:
            for table_name in list(all_tables_info.keys())[:5]:
                selected.append((table_name, 0.3))
        
        return sorted(selected, key=lambda x: x[1], reverse=True)[:10]
    
    def _semantic_search(self, user_prompt: str, all_tables_info: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Search for tables using semantic similarity with enhanced column-level search and quality filtering."""
        try:
            if not hasattr(self, 'embedding_service') or not hasattr(self, 'table_summaries'):
                return []
            
            # Store quality assessments for later use
            self._table_quality_scores = {}
            
            # 1. Embed user prompt (once)
            prompt_embedding = self.embedding_service.embed_text(user_prompt)
            
            # 2. Table-level search (existing)
            table_candidates = self._table_level_search(prompt_embedding, all_tables_info)
            
            # 3. Column-level search (new)
            column_candidates = self._column_level_search(prompt_embedding, all_tables_info)
            
            # 4. Assess quality for all candidates
            all_candidate_tables = set(table_candidates.keys()) | set(column_candidates.keys())
            for table_name in all_candidate_tables:
                # Get cached sample if available
                sample_data = None
                if hasattr(self, '_sql_agent_ref') and hasattr(self._sql_agent_ref, 'sample_cache'):
                    sample_data = self._sql_agent_ref.sample_cache.get(table_name)
                
                # Get table type from summary
                table_type = None
                if table_name in self.table_summaries:
                    table_type = self.table_summaries[table_name]['summary'].get('table_type')
                
                # Assess quality
                if sample_data:
                    quality_assessment = self._assess_table_quality_from_sample_local(
                        table_name, sample_data, table_type
                    )
                    self._table_quality_scores[table_name] = quality_assessment
            
            # 5. Combine scores with quality filtering
            combined_candidates = self._combine_search_results_with_quality(
                table_candidates, 
                column_candidates,
                all_tables_info
            )
            
            # 6. Return enhanced candidates
            return combined_candidates[:15] # Return top 15 candidates
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _assess_table_quality_from_sample_local(self, table_name: str, sample_data: Dict[str, Any], 
                                            table_type: Optional[str] = None) -> Dict[str, Any]:
        """Local version of quality assessment for table selector"""
        if not sample_data or not sample_data.get('rows'):
            return {
                'quality_score': 0.0,
                'row_count_estimate': 0,
                'is_exact_count': True,
                'fill_rate': 0.0,
                'has_meaningful_data': False,
                'empty_columns': [],
                'quality_category': 'empty'
            }
        
        rows = sample_data['rows']
        columns = sample_data['columns']
        actual_row_count = len(rows)
        
        # Estimate total rows
        if actual_row_count < 1000:
            row_count_estimate = actual_row_count
            is_exact_count = True
            row_category = 'small' if actual_row_count < 100 else 'medium'
        else:
            row_count_estimate = '1000+'
            is_exact_count = False
            row_category = 'large'
        
        # Calculate fill rates
        column_fill_rates = {}
        empty_columns = []
        
        for col in columns:
            non_null_count = sum(1 for row in rows if row.get(col) is not None and str(row.get(col)).strip())
            fill_rate = non_null_count / actual_row_count if actual_row_count > 0 else 0
            column_fill_rates[col] = fill_rate
            
            if fill_rate == 0:
                empty_columns.append(col)
        
        # Overall fill rate
        overall_fill_rate = sum(column_fill_rates.values()) / len(columns) if columns else 0
        
        # Check if has meaningful data
        has_meaningful_data = any(rate > 0.5 for rate in column_fill_rates.values())
        
        # Calculate quality score based on table type
        quality_score = self._calculate_type_aware_quality_score_local(
            table_type=table_type,
            row_count=actual_row_count,
            row_category=row_category,
            fill_rate=overall_fill_rate,
            empty_column_ratio=len(empty_columns) / len(columns) if columns else 1,
            has_meaningful_data=has_meaningful_data
        )
        
        return {
            'quality_score': quality_score,
            'row_count_estimate': row_count_estimate,
            'is_exact_count': is_exact_count,
            'row_category': row_category,
            'fill_rate': overall_fill_rate,
            'empty_columns': empty_columns,
            'has_meaningful_data': has_meaningful_data,
            'quality_category': 'high' if quality_score > 0.7 else ('medium' if quality_score > 0.4 else 'low')
        }

    def _calculate_type_aware_quality_score_local(self, table_type: Optional[str], row_count: int,
                                                row_category: str, fill_rate: float,
                                                empty_column_ratio: float, has_meaningful_data: bool) -> float:
        """Local version of type-aware quality scoring"""
        if not table_type:
            table_type = 'unknown'
        
        # Type-specific scoring
        if table_type in ['catalog', 'reference', 'configuration']:
            # Catalog tables: row count doesn't matter, fill rate is critical
            quality_score = (
                0.5 * fill_rate +
                0.3 * (1.0 if has_meaningful_data else 0.0) +
                0.2 * (1 - empty_column_ratio)
            )
        elif table_type in ['fact', 'transaction']:
            # Fact tables: need many rows and reasonable fill rate
            row_score = 1.0 if row_category == 'large' else (0.5 if row_category == 'medium' else 0.2)
            quality_score = (
                0.3 * row_score +
                0.3 * fill_rate +
                0.2 * (1.0 if has_meaningful_data else 0.0) +
                0.2 * (1 - empty_column_ratio)
            )
        elif table_type == 'dimension':
            # Dimension tables: moderate rows, high fill rate
            row_score = 1.0 if row_count >= 10 else (row_count / 10)
            quality_score = (
                0.2 * row_score +
                0.4 * fill_rate +
                0.2 * (1 - empty_column_ratio) +
                0.2 * (1.0 if has_meaningful_data else 0.0)
            )
        else:
            # Unknown type: balanced scoring
            row_score = min(1.0, row_count / 100) if row_count < 1000 else 1.0
            quality_score = (
                0.25 * row_score +
                0.25 * fill_rate +
                0.25 * (1.0 if has_meaningful_data else 0.0) +
                0.25 * (1 - empty_column_ratio)
            )
        
        return quality_score

    def _combine_search_results_with_quality(self, table_scores: Dict[str, float], 
                                        column_matches: Dict[str, Dict[str, Any]],
                                        all_tables_info: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Combine table and column search results with quality filtering."""
        combined_scores = {}
        
        # Get all tables that appeared in either search
        all_matched_tables = set(table_scores.keys()) | set(column_matches.keys())
        
        for table_name in all_matched_tables:
            # Get individual scores
            table_similarity = table_scores.get(table_name, 0)
            column_data = column_matches.get(table_name, {})
            
            best_column_score = column_data.get('best_score', 0)
            avg_top_scores = column_data.get('avg_top_scores', 0)
            coverage_score = column_data.get('coverage_score', 0)
            
            # Get quality assessment
            quality_assessment = self._table_quality_scores.get(table_name, {})
            quality_score = quality_assessment.get('quality_score', 0.5) # Default to 0.5 if no assessment
            
            # Get table type for intent matching
            table_type = None
            if table_name in self.table_summaries:
                table_type = self.table_summaries[table_name]['summary'].get('table_type')
            
            # Calculate semantic role bonus
            semantic_role_bonus = self._calculate_semantic_role_bonus(
                column_data.get('semantic_roles', set())
            )
            
            # Calculate intent-type compatibility
            intent_compatibility = self._calculate_intent_type_compatibility(table_type)
            
            # Apply quality-based filtering
            if quality_score < 0.2 and table_type not in ['catalog', 'reference']:
                # Skip very low quality non-catalog tables
                continue
            
            # Weighted combination with quality factor
            combined_score = (
                0.25 * table_similarity +
                0.30 * best_column_score +
                0.15 * coverage_score +
                0.10 * semantic_role_bonus +
                0.10 * intent_compatibility +
                0.10 * quality_score # Quality factor
            )
            
            # Boost catalog tables for lookup intent
            if hasattr(self, '_current_intent') and self._current_intent:
                if self._current_intent.intent_type.value == 'lookup' and table_type in ['catalog', 'reference']:
                    combined_score *= 1.2 # 20% boost
            
            # Penalize nearly empty tables (unless catalog)
            if quality_assessment.get('row_count_estimate', 0) < 10 and table_type not in ['catalog', 'reference']:
                combined_score *= 0.5
            
            # Store detailed info for later use
            combined_scores[table_name] = {
                'score': combined_score,
                'table_similarity': table_similarity,
                'best_column_match': best_column_score,
                'matched_columns': column_data.get('columns', []),
                'quality_score': quality_score,
                'quality_category': quality_assessment.get('quality_category', 'unknown'),
                'table_type': table_type
            }
        
        # Convert to list format with enhanced info
        candidates = []
        for table_name, score_data in combined_scores.items():
            candidates.append((table_name, score_data['score']))
            # Store column match info for use in prompt building
            if not hasattr(self, '_column_match_details'):
                self._column_match_details = {}
            self._column_match_details[table_name] = score_data['matched_columns']
            
            # Store quality info
            if not hasattr(self, '_table_quality_details'):
                self._table_quality_details = {}
            self._table_quality_details[table_name] = {
                'quality_score': score_data['quality_score'],
                'quality_category': score_data['quality_category'],
                'table_type': score_data['table_type']
            }
        
        # Sort by combined score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates

    def _calculate_intent_type_compatibility(self, table_type: Optional[str]) -> float:
        """Calculate compatibility between table type and query intent"""
        if not hasattr(self, '_current_intent') or not self._current_intent:
            return 0.5
        
        if not table_type:
            return 0.5
        
        intent_type = self._current_intent.intent_type.value
        
        # Define compatibility matrix
        compatibility_matrix = {
            'lookup': {
                'catalog': 1.0, 'reference': 1.0, 'dimension': 0.8,
                'fact': 0.5, 'transaction': 0.5, 'junction': 0.3,
                'configuration': 0.7, 'unknown': 0.5
            },
            'aggregation': {
                'fact': 1.0, 'transaction': 1.0, 'dimension': 0.6,
                'catalog': 0.3, 'reference': 0.3, 'junction': 0.4,
                'configuration': 0.2, 'unknown': 0.5
            },
            'trend': {
                'fact': 1.0, 'transaction': 1.0, 'dimension': 0.5,
                'catalog': 0.2, 'reference': 0.2, 'junction': 0.3,
                'configuration': 0.1, 'unknown': 0.5
            },
            'distribution': {
                'fact': 1.0, 'transaction': 0.9, 'dimension': 0.8,
                'catalog': 0.4, 'reference': 0.4, 'junction': 0.5,
                'configuration': 0.2, 'unknown': 0.5
            },
            'comparison': {
                'fact': 0.8, 'transaction': 0.8, 'dimension': 0.9,
                'catalog': 0.5, 'reference': 0.5, 'junction': 0.4,
                'configuration': 0.3, 'unknown': 0.5
            }
        }
        
        return compatibility_matrix.get(intent_type, {}).get(table_type, 0.5)
    
    def _table_level_search(self, prompt_embedding, all_tables_info: Dict[str, Any]) -> Dict[str, float]:
        """Perform table-level embedding search."""
        table_scores = {}
        
        for table_name in all_tables_info:
            if table_name in self.table_summaries:
                summary_data = self.table_summaries[table_name]
                table_embedding = summary_data['embedding']
                similarity = self.embedding_service.compute_similarity(
                    prompt_embedding, table_embedding
                )
                if similarity > 0.6:
                    table_scores[table_name] = similarity
                    
        return table_scores
    
    def _column_level_search(self, prompt_embedding, all_tables_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Perform column-level embedding search."""
        column_matches_by_table = {}
        
        # Compare with all column embeddings
        for col_key, col_data in self.column_summaries.items():
            table_name, col_name = col_key.split('.', 1)
            
            # Skip if table not in available tables
            if table_name not in all_tables_info:
                continue
                
            # Calculate similarity
            col_embedding = col_data['embedding']
            similarity = self.embedding_service.compute_similarity(
                prompt_embedding, col_embedding
            )
            
            # Use threshold of 0.6 (can be adjusted for retry)
            threshold = 0.6
            if hasattr(self, '_retry_context') and self._retry_context:
                threshold = 0.5  # Lower threshold on retry
                
            if similarity > threshold:
                if table_name not in column_matches_by_table:
                    column_matches_by_table[table_name] = {
                        'columns': [],
                        'best_score': 0,
                        'avg_top_scores': 0,
                        'semantic_roles': set()
                    }
                    
                # Get column summary for semantic role
                col_summary = col_data['summary']
                semantic_role = col_summary.get('semantic_role', 'other')
                
                column_matches_by_table[table_name]['columns'].append({
                    'name': col_name,
                    'score': similarity,
                    'semantic_role': semantic_role,
                    'canonical_name': col_summary.get('canonical_name', col_name),
                    'description': col_summary.get('description', '')
                })
                
                column_matches_by_table[table_name]['semantic_roles'].add(semantic_role)
                
        # Calculate aggregated scores for each table
        for table_name, match_data in column_matches_by_table.items():
            columns = match_data['columns']
            # Sort by score
            columns.sort(key=lambda x: x['score'], reverse=True)
            
            # Best column match
            match_data['best_score'] = columns[0]['score'] if columns else 0
            
            # Average of top-3 columns
            top_columns = columns[:3]
            match_data['avg_top_scores'] = sum(c['score'] for c in top_columns) / len(top_columns)
            
            # Column coverage score
            match_data['coverage_score'] = min(len(columns) / 5.0, 1.0)  # Normalize to max 5 columns
            
            # Keep only top 5 columns for each table
            match_data['columns'] = columns[:5]
            
        return column_matches_by_table
    

    def _combine_search_results(self, table_scores: Dict[str, float], 
                            column_matches: Dict[str, Dict[str, Any]],
                            all_tables_info: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Combine table and column search results with multi-level scoring."""
        combined_scores = {}
        
        # Get all tables that appeared in either search
        all_matched_tables = set(table_scores.keys()) | set(column_matches.keys())
        
        for table_name in all_matched_tables:
            # Get individual scores
            table_similarity = table_scores.get(table_name, 0)
            column_data = column_matches.get(table_name, {})
            
            best_column_score = column_data.get('best_score', 0)
            avg_top_scores = column_data.get('avg_top_scores', 0)
            coverage_score = column_data.get('coverage_score', 0)
            
            # Calculate semantic role bonus
            semantic_role_bonus = self._calculate_semantic_role_bonus(
                column_data.get('semantic_roles', set())
            )
            
            # Weighted combination
            combined_score = (
                0.3 * table_similarity +
                0.4 * best_column_score +
                0.2 * coverage_score +
                0.1 * semantic_role_bonus
            )
            
            # Store detailed info for later use
            combined_scores[table_name] = {
                'score': combined_score,
                'table_similarity': table_similarity,
                'best_column_match': best_column_score,
                'matched_columns': column_data.get('columns', [])
            }
            
        # Convert to list format with enhanced info
        candidates = []
        for table_name, score_data in combined_scores.items():
            candidates.append((table_name, score_data['score']))
            # Store column match info for use in prompt building
            if not hasattr(self, '_column_match_details'):
                self._column_match_details = {}
            self._column_match_details[table_name] = score_data['matched_columns']
            
        # Sort by combined score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates
    
    def _calculate_semantic_role_bonus(self, semantic_roles: set) -> float:
        """Calculate bonus based on semantic roles found."""
        if not hasattr(self, '_current_intent'):
            return 0.5
            
        intent_type = getattr(self._current_intent, 'intent_type', None)
        if not intent_type:
            return 0.5
            
        # Intent-aware scoring
        if intent_type.value == 'aggregation':
            if 'measure' in semantic_roles:
                return 1.0
            elif 'dimension' in semantic_roles:
                return 0.7
        elif intent_type.value == 'lookup':
            if 'id' in semantic_roles or 'dimension' in semantic_roles:
                return 1.0
            elif 'code' in semantic_roles:
                return 0.8
        elif intent_type.value == 'trend':
            if 'time' in semantic_roles and 'measure' in semantic_roles:
                return 1.0
            elif 'time' in semantic_roles:
                return 0.8
                
        # Default bonus for having any meaningful semantic roles
        meaningful_roles = semantic_roles - {'other'}
        if meaningful_roles:
            return 0.6
        return 0.5

    def _get_join_paths(self, selected_tables: List[str]) -> Dict[str, Any]:
        """Find reliable join paths between selected tables"""
        join_paths = {}
        # For each pair of tables, find best path
        for i, table1 in enumerate(selected_tables):
            for table2 in selected_tables[i+1:]:
                path = self.relationship_graph.find_shortest_reliable_path(
                    table1, table2, min_weight=0.6, max_hops=2
                )
                if path:
                    join_paths[f"{table1}->{table2}"] = {
                        'path': path,
                        'total_weight': sum(w for _,_ , w in path),
                        'explanation': self._explain_path(path)
                    }
        return join_paths

    def _explain_path(self, path: List[Tuple[str, str, float]]) -> str:
        """Generate explanation for a join path"""
        explanations = []
        for from_table, to_table, weight in path:
            explanation = self.relationship_graph.get_edge_explanation(from_table, to_table)
            explanations.append(f"{from_table} → {to_table}: {explanation}")
        return " | ".join(explanations)

    def _enhance_with_graph_info(self, selected_tables: List[Tuple[str, float]],
                                all_tables_info: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Enhance table selection using graph information"""
        enhanced = []
        selected_names = [t[0] for t in selected_tables]
        # Check if we need bridge tables
        for table_name, confidence in selected_tables:
            enhanced.append((table_name, confidence))
        # Look for necessary junction tables
        if hasattr(self, 'graph_builder') and hasattr(self.graph_builder, 'nodes'):
            nodes = self.graph_builder.nodes
            # Find junction tables that connect our selected tables
            for junction_name, node in nodes.items():
                if node.table_type == 'junction' and junction_name not in selected_names:
                    # Check if this junction connects any of our selected tables
                    connected_count = 0
                    for fk in node.foreign_keys:
                        if '.' in fk:
                            target_table = fk.split('.')[0]
                            if target_table in selected_names:
                                connected_count += 1
                    if connected_count >= 2:
                        # This junction connects multiple selected tables
                        enhanced.append((junction_name, 0.8))
                        logger.info(f"Added junction table {junction_name} to connect selected tables")
        return enhanced