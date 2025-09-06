from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import re
from models import QueryMemory, InsightHistory, QueryComplexity
from relationship_graph import RelationshipGraph
from config import config

logger = logging.getLogger(__name__)

class LearningPipeline:
    def __init__(self, relationship_graph: RelationshipGraph, persistence_manager=None, target_db: str = None):
        self.relationship_graph = relationship_graph
        self.persistence_manager = persistence_manager
        self.target_db = target_db

        # Load history from persistence
        if self.persistence_manager and self.target_db:
            self.query_history = self.persistence_manager.get_query_history(self.target_db)
        else:
            self.query_history = []

        self.insights_history = []
        self.join_usage_history = []

    def process_query_result(self, query_memory: QueryMemory,
                            query_result: Any,
                            sample_data_used: Dict[str, Any]):
        """Process a completed query for learning"""
        # Calculate success score
        success_score = self._calculate_conservative_success_score(
            query_memory, query_result
        )
        query_memory.success_score = success_score
        
        # Store in history
        self.query_history.append(query_memory)
        if self.persistence_manager and self.target_db:
            self.persistence_manager.save_query_memory(self.target_db, query_memory)
        
        # Learn from queries with reasonable success (lowered threshold)
        if success_score > 0.7:  # Lowered from 0.8
            # Update relationship graph
            self._update_relationship_graph(query_memory, success_score)
            
            # Update metadata insights
            self._update_metadata_insights(
                query_memory, query_result, sample_data_used
            )
        
        # Decay old insights periodically
        if len(self.query_history) % 100 == 0:
            self._decay_insights()

    def _calculate_conservative_success_score(self, query_memory: QueryMemory,
                                               query_result: Any) -> float:
        """Calculate conservative success score using multiplicative approach"""

        success_components = {
            'execution_success': 1.0 if query_result.error is None else 0.0,
            'no_retries': 1.0 if query_memory.retry_count == 0 else (1.0 - query_memory.retry_count/3),
            'validation_passed': query_result.validation_score,
            'execution_time_reasonable': 1.0 if query_memory.execution_time < 5.0 else 0.5,
            'result_size_reasonable': 1.0 if 0 < len(query_result.data) < 10000 else 0.3,
            'sample_data_helped': 1.0 if query_result.used_sample_data else 0.8
        }

        # Multiplicative scoring - all components must be good
        success_score = 1.0
        for component, score in success_components.items():
            success_score *= score

        logger.info(f"Success score calculation: {success_components} -> {success_score:.3f}")

        return success_score

    def _update_relationship_graph(self, query_memory: QueryMemory,
                                   success_score: float):
        """Update relationship graph based on successful JOINs"""

        # Extract JOINs from SQL
        joins = self._extract_joins_from_sql(query_memory.sql_query)

        for table1, table2 in joins:
            # Validate diversity before updating
            if self._validate_relationship_diversity(table1, table2):
                current_weight = self.relationship_graph.get_edge_weight(table1, table2)

                # Only update if not an explicit FK (weight < 1.0)
                if current_weight < 1.0:
                    # Conservative increment
                    new_weight = min(0.9, current_weight + 0.1)
                    self.relationship_graph.update_edge(table1, table2, new_weight)

                # Track usage
                self.relationship_graph.increment_usage_count(table1, table2)
                self.relationship_graph.add_usage_context(table1, table2, {
                    'query_id': query_memory.id,
                    'intent_type': query_memory.intent_summary.get('intent_type'),
                    'success_score': success_score,
                    'timestamp': datetime.now()
                })

                # Store in join history
                self.join_usage_history.append({
                    'table1': table1,
                    'table2': table2,
                    'query_id': query_memory.id,
                    'intent_type': query_memory.intent_summary.get('intent_type'),
                    'success_score': success_score,
                    'timestamp': datetime.now()
                })

    def _validate_relationship_diversity(self, table1: str, table2: str) -> bool:
        """Check if JOIN has been used in diverse contexts"""

        # Get usage contexts for this join
        usage_contexts = [
            entry for entry in self.join_usage_history
            if (entry['table1'] == table1 and entry['table2'] == table2) or
            (entry['table1'] == table2 and entry['table2'] == table1)
        ]

        if len(usage_contexts) < 2:
            return False # Need at least 2 uses

        # Calculate diversity
        intent_types = set(ctx['intent_type'] for ctx in usage_contexts if ctx['intent_type'])
        unique_queries = set(ctx['query_id'] for ctx in usage_contexts)

        diversity_score = (
            len(intent_types) / 3 + # Intent diversity
            min(1.0, len(unique_queries) / 5) / 2 # Usage frequency
        )

        return diversity_score > config.DIVERSITY_THRESHOLD

    def _update_metadata_insights(self, query_memory: QueryMemory,
                                  query_result: Any,
                                  sample_data_used: Dict[str, Any]):
        """Update metadata insights based on successful queries"""

        # Check if we should update metadata
        if not self._should_update_metadata(
            query_memory.success_score,
            query_memory.tables_used,
            query_memory.query_complexity
        ):
            return

        # Generate insights for each table used
        for table_name in query_memory.tables_used:
            # Generate insight
            insight = self._generate_insight_with_samples(
                query_result.data,
                table_name,
                sample_data_used.get(table_name, {})
            )

            if insight:
                # Calculate insight confidence
                insight_confidence = self._calculate_insight_confidence(
                    query_memory.success_score,
                    query_result.validation_score,
                    table_name
                )

                # Check if this insight is better than existing
                existing_insight = self._get_existing_insight(table_name)

                if not existing_insight or insight_confidence > existing_insight.confidence_score:
                    # Store new insight
                    self.insights_history.append(InsightHistory(
                        table_name=table_name,
                        column_name=None, # Could be enhanced to column-level
                        insight=insight,
                        confidence_score=insight_confidence,
                        timestamp=datetime.now(),
                        query_context=query_memory.user_prompt,
                        success_flag=True
                    ))

                    logger.info(f"New insight for {table_name}: {insight} (confidence: {insight_confidence:.3f})")

    def _should_update_metadata(self, success_score: float,
                                tables_used: List[str],
                                query_complexity: QueryComplexity) -> bool:
        """Determine if metadata should be updated"""

        # Adaptive threshold based on complexity
        required_successes = (
            config.SIMPLE_QUERY_SUCCESS_REQUIREMENT
            if query_complexity == QueryComplexity.SIMPLE
            else config.COMPLEX_QUERY_SUCCESS_REQUIREMENT
        )

        # Require high success score
        if success_score < config.METADATA_UPDATE_THRESHOLD:
            return False

        # Check if pattern has been successful multiple times
        for table in tables_used:
            similar_successes = self._count_similar_successes(table)
            if similar_successes >= required_successes:
                return True

        return False

    def _count_similar_successes(self, table_name: str) -> int:
        """Count successful queries using this table"""
        return sum(
            1 for query in self.query_history
            if table_name in query.tables_used and query.success_score > 0.8
        )

    def _generate_insight_with_samples(self, results: Any,
                                       table_name: str,
                                       sample_data: Dict[str, Any]) -> Optional[str]:
        """Generate insight based on query results and sample data"""

        insights = []

        # Analyze result patterns
        if hasattr(results, 'columns') and table_name in str(results.columns):
            insights.append(f"Table {table_name} commonly used for {results.columns.tolist()}")

        # Analyze sample data patterns
        if sample_data and 'rows' in sample_data:
            # Look for common values
            for col in sample_data.get('columns', []):
                values = [row.get(col) for row in sample_data['rows'] if row.get(col)]
                if values:
                    unique_ratio = len(set(values)) / len(values)
                    if unique_ratio < 0.5:
                        insights.append(f"Column {col} has low cardinality")

        return "; ".join(insights) if insights else None

    def _calculate_insight_confidence(self, success_score: float,
                                      validation_score: float,
                                      table_name: str) -> float:
        """Calculate confidence for a new insight"""

        # Get usage statistics
        usage_count = sum(1 for q in self.query_history if table_name in q.tables_used)
        total_queries = len(self.query_history)
        usage_ratio = usage_count / total_queries if total_queries > 0 else 0

        # Sample data quality (simplified)
        sample_quality = 0.8 # Could be enhanced based on actual sample analysis

        confidence = (
            0.4 * success_score +
            0.3 * validation_score +
            0.2 * usage_ratio +
            0.1 * sample_quality
        )

        return confidence

    def _get_existing_insight(self, table_name: str) -> Optional[InsightHistory]:
        """Get the most recent insight for a table"""
        table_insights = [
            i for i in self.insights_history
            if i.table_name == table_name
        ]

        if table_insights:
            # Return the most recent one
            return max(table_insights, key=lambda x: x.timestamp)

        return None

    def _extract_joins_from_sql(self, sql: str) -> List[Tuple[str, str]]:
        """Extract JOIN relationships from SQL"""
        joins = []

        # Simple regex-based extraction
        join_pattern = r'(\w+)\s+(?:INNER\s+)?JOIN\s+(\w+)\s+ON'
        matches = re.findall(join_pattern, sql.upper())

        for match in matches:
            if len(match) >= 2:
                # The first table is usually before JOIN
                # Need to look back to find it
                before_join = sql[:sql.upper().find(f'JOIN {match[1]}')]
                tables_before = re.findall(r'FROM\s+(\w+)', before_join.upper())

                if tables_before:
                    joins.append((tables_before[-1], match[1]))

        return joins

    def _decay_insights(self):
        """Decay confidence of old insights"""
        current_time = datetime.now()

        for insight in self.insights_history:
            days_old = (current_time - insight.timestamp).days
            decay_factor = config.CONFIDENCE_DECAY_RATE ** (days_old / config.DECAY_PERIOD_DAYS)
            insight.confidence_score *= decay_factor

        # Remove insights with very low confidence
        self.insights_history = [
            i for i in self.insights_history
            if i.confidence_score > 0.1
        ]

        logger.info(f"Decayed insights. Remaining: {len(self.insights_history)}")

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning pipeline"""

        total_queries = len(self.query_history)
        successful_queries = sum(1 for q in self.query_history if q.success_score > 0.8)

        # Calculate average success rate by complexity
        simple_queries = [q for q in self.query_history if q.query_complexity == QueryComplexity.SIMPLE]
        complex_queries = [q for q in self.query_history if q.query_complexity == QueryComplexity.COMPLEX]

        stats = {
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
            'simple_query_success_rate': (
                sum(q.success_score > 0.8 for q in simple_queries) / len(simple_queries)
                if simple_queries else 0
            ),
            'complex_query_success_rate': (
                sum(q.success_score > 0.8 for q in complex_queries) / len(complex_queries)
                if complex_queries else 0
            ),
            'total_insights': len(self.insights_history),
            'active_insights': sum(1 for i in self.insights_history if i.confidence_score > 0.5),
            'relationship_updates': len(self.join_usage_history),
            'unique_joins_learned': len(set(
                (j['table1'], j['table2']) for j in self.join_usage_history
            ))
        }

        return stats