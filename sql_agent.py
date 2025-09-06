from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, date
import asyncio
import pandas as pd
import numpy as np
import json
import sqlalchemy as sa


from connection import DatabaseConnection
from metadata_extractor import MetadataExtractor
from sample_cache import SampleDataCache
from embedding_service import EmbeddingService
from relationship_graph import RelationshipGraph
from schema_analyzer import SchemaAnalyzer
from intent_analyzer import IntentAnalyzer
from table_selector_llm import TableSelectorLLM
from sql_generator import SQLGenerator
from query_executor import QueryExecutor
from result_validator_llm import ResultValidatorLLM
from learning_pipeline import LearningPipeline
from chart_generator import ChartGenerator
from models import QueryMemory, QueryComplexity, RelationshipEdge, GraphNode
from config import config

from graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    # Input - Use Annotated to handle potential multiple updates
    user_prompt: str
    
    # Processing state
    intent: Optional[Dict[str, Any]]
    similar_queries: Optional[List[Dict[str, Any]]]
    selected_tables: Optional[List[tuple]]
    table_info: Optional[Dict[str, Any]]
    sample_data: Optional[Dict[str, Any]]
    sql_query: Optional[str]
    query_confidence: Optional[float]
    query_result: Optional[Dict[str, Any]]
    validation_score: Optional[float]
    validation_checks: Optional[Dict[str, bool]]
    
    # Iteration control
    iteration: int
    max_iterations: int
    
    # Output
    response: Optional[Dict[str, Any]]
    visualization: Optional[Dict[str, Any]]
    explanation: Optional[str]
    overall_confidence: Optional[float]

class SQLAgent:
    def __init__(self, database_uri: str):
        # Extract database identifier from URI
        self.target_db = self._extract_db_identifier(database_uri)
        
        # Initialize persistence
        from persistence import PersistenceManager
        self.persistence_manager = PersistenceManager()
        
        # Check if already initialized
        agent_state = self.persistence_manager.get_agent_state(self.target_db)
        
        # Initialize components
        self.connection = DatabaseConnection(database_uri)
        self.connection.connect()
        
        # Get dialect after connection is established
        self.dialect = self.connection.dialect_name
        logger.info(f"Database dialect: {self.dialect}")
        
        # Initialize all components
        self.metadata_extractor = MetadataExtractor(self.connection)
        self.embedding_service = EmbeddingService(persistence_manager=self.persistence_manager)
        self.relationship_graph = RelationshipGraph(
            persistence_manager=self.persistence_manager,
            target_db=self.target_db
        )
        self.schema_analyzer = SchemaAnalyzer(self.connection, self.metadata_extractor)
        
        from schema_summarizer import SchemaSummarizer
        self.schema_summarizer = SchemaSummarizer(
            self.embedding_service,
            self.persistence_manager,
            self.target_db,
            self.connection  # Add connection parameter
        )
        
        self.table_summaries = self.persistence_manager.get_table_summaries(self.target_db)
        self.column_summaries = self.persistence_manager.get_column_summaries(self.target_db)
        
        self.intent_analyzer = IntentAnalyzer()
        
        # Use the new LLM-based components
        self.table_selector_llm = TableSelectorLLM()
        self.table_selector_llm.embedding_service = self.embedding_service
        self.table_selector_llm.table_summaries = self.table_summaries
        self.table_selector_llm.column_summaries = self.column_summaries
        self.table_selector_llm.sql_agent = self
        
        self.result_validator_llm = ResultValidatorLLM()
        
        # Pass dialect to SQL generator - ONLY ONCE
        self.sql_generator = SQLGenerator(dialect=self.dialect)
        
        # Pass connection (which has dialect) to query executor
        self.query_executor = QueryExecutor(self.connection)
        
        self.learning_pipeline = LearningPipeline(
            self.relationship_graph,
            persistence_manager=self.persistence_manager,
            target_db=self.target_db
        )
        
        self.chart_generator = ChartGenerator()
        
        # Initialize metadata and embeddings - THIS MUST COME BEFORE GraphBuilder
        self._initialize(agent_state)
        
        # NOW initialize graph builder after metadata is loaded
        self.graph_builder = GraphBuilder(
            self.relationship_graph,
            self.metadata,
            self.table_summaries,
            self.column_summaries,
            self.embedding_service,
            self.schema_analyzer,
            self.connection
        )
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info(f"Initialization complete:")
        logger.info(f"Metadata tables: {len(self.metadata)}")
        logger.info(f"Table summaries: {len(self.table_summaries)}")
        logger.info(f"Column summaries: {len(self.column_summaries)}")
        logger.info(f"Sample cache initialized: {hasattr(self, 'sample_cache')}")
        logger.info(f"Relationship graph edges: {self.relationship_graph.graph.number_of_edges()}")
 
    def _extract_db_identifier(self, database_uri: str) -> str:
        """Extract a unique identifier from database URI."""
        # Simple extraction - can be enhanced
        import hashlib
        return hashlib.md5(database_uri.encode()).hexdigest()[:16]

    def _initialize(self, agent_state: Dict[str, Any] = None):
        """One-time initialization of metadata and embeddings"""
        logger.info(f"Initializing SQL Agent for target DB: {self.target_db}")
        
        if agent_state is None:
            agent_state = {'metadata_extracted': False, 'embeddings_computed': False}
        
        # Check if already initialized
        if agent_state['metadata_extracted']:
            logger.info("Loading existing metadata from persistence")
            # Load metadata
            self.metadata = self.persistence_manager.get_table_metadata(self.target_db)
            
            if not self.metadata:
                logger.info("No metadata in persistence, extracting from database...")
                self.metadata = self.metadata_extractor.extract_all_metadata()
                # Save metadata
                for table_name, table_info in self.metadata.items():
                    self.persistence_manager.save_table_metadata(self.target_db, table_name, table_info)
                # Update state
                self.persistence_manager.update_agent_state(
                    self.target_db,
                    metadata_extracted=True
                )
        else:
            # Extract metadata
            logger.info("Extracting metadata from database...")
            self.metadata = self.metadata_extractor.extract_all_metadata()
            # Save metadata
            for table_name, table_info in self.metadata.items():
                self.persistence_manager.save_table_metadata(self.target_db, table_name, table_info)
            # Update state
            self.persistence_manager.update_agent_state(
                self.target_db,
                metadata_extracted=True,
                last_initialized=datetime.now().isoformat()
            )

        # Initialize sample cache with persistence
        self.sample_cache = SampleDataCache(self.persistence_manager, self.target_db)

        # Extract foreign keys if not done
        if not hasattr(self.relationship_graph, 'graph') or self.relationship_graph.graph.number_of_nodes() == 0:
            logger.info("Initializing relationship graph...")
            foreign_keys = self.metadata_extractor.extract_foreign_keys()
            self.relationship_graph.initialize_from_metadata(foreign_keys)

        logger.info(f"SQL Agent initialization complete. Tables: {len(self.metadata)}")


    # Update the conditional edge logic to avoid parallel execution conflicts
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_intent", self.analyze_intent)
        workflow.add_node("retrieve_examples", self.retrieve_examples)
        workflow.add_node("select_tables", self.select_tables)
        workflow.add_node("fetch_samples", self.fetch_samples)
        workflow.add_node("generate_sql", self.generate_sql)
        workflow.add_node("execute_query", self.execute_query)
        workflow.add_node("validate_results", self.validate_results)
        workflow.add_node("generate_visualization", self.generate_visualization)
        workflow.add_node("generate_explanation", self.generate_explanation)
        workflow.add_node("synthesize_response", self.synthesize_response)
        workflow.add_node("learn_from_query", self.learn_from_query)
        
        # Define edges
        workflow.add_edge("analyze_intent", "retrieve_examples")
        workflow.add_edge("retrieve_examples", "select_tables")
        workflow.add_edge("select_tables", "fetch_samples")
        workflow.add_edge("fetch_samples", "generate_sql")
        workflow.add_edge("generate_sql", "execute_query")
        workflow.add_edge("execute_query", "validate_results")
        
        # Conditional edge for retry logic
        workflow.add_conditional_edges(
            "validate_results",
            self.should_retry,
            {
                "retry": "select_tables",
                "continue": "generate_visualization"
            }
        )
        
        # Sequential processing instead of parallel
        workflow.add_edge("generate_visualization", "generate_explanation")
        workflow.add_edge("generate_explanation", "synthesize_response")
        
        # Final steps
        workflow.add_edge("synthesize_response", "learn_from_query")
        workflow.add_edge("learn_from_query", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_intent")
        
        # Compile without config parameter
        return workflow.compile()

    async def analyze_intent(self, state: AgentState) -> Dict[str, Any]:
        """Analyze user intent"""
        logger.info(f"Analyzing intent for: {state['user_prompt']}")
        
        intent = self.intent_analyzer.analyze_intent(state['user_prompt'])
        
        # Only return the fields this node updates
        return {
            'intent': {
                'intent_type': intent.intent_type.value,
                'entities': intent.entities,
                'metrics': intent.metrics,
                'filters': intent.filters,
                'time_range': intent.time_range,
                'expected_result_type': intent.expected_result_type,
                'expects_single_value': intent.expects_single_value,
                'expects_list': intent.expects_list,
                'expects_aggregation': intent.expects_aggregation,
                'intent_confidence': intent.intent_confidence,
                'query_complexity': intent.query_complexity.value
            }
        }

    async def retrieve_examples(self, state: AgentState) -> AgentState:
        """Retrieve similar successful queries"""
        logger.info("Retrieving similar queries...")

        # Get query examples from history
        similar_queries = self._get_query_examples(
            state['user_prompt'],
            self.learning_pipeline.query_history
        )

        state['similar_queries'] = similar_queries
        logger.info(f"Found {len(similar_queries)} similar queries")

        return state

    def should_retry(self, state: AgentState) -> str:
        """Determine if query should be retried based on validation"""
        current_iteration = state.get('iteration', 0)
        max_iterations = state.get('max_iterations', 3)
        validation_details = state.get('validation_checks', {})
        
        # Check if we've exceeded max iterations
        if current_iteration >= max_iterations - 1:
            logger.info(f"Reached maximum iterations ({max_iterations}), stopping retries")
            return "continue"
        
        # Check if validation score is too low
        validation_score = state.get('validation_score', 0)
        if validation_score < 0.6:
            retry_strategy = validation_details.get('retry_strategy', 'none')
            
            # Check if we're stuck in a loop with the same table
            selected_tables = state.get('selected_tables', [])
            previous_tables = state.get('previous_selected_tables', [])
            
            if selected_tables and previous_tables and selected_tables == previous_tables:
                logger.info("Same tables selected again, trying different strategy")
                # Force a different strategy or exit
                if retry_strategy == 'change_tables':
                    retry_strategy = 'none'  # Exit the loop
                else:
                    retry_strategy = 'change_tables'
            
            if retry_strategy != 'none':
                logger.info(f"Validation failed (score: {validation_score:.2f}). "
                        f"Retrying with strategy: {retry_strategy} (iteration {current_iteration + 1})")
                return "retry"
        
        return "continue"


    async def select_tables(self, state: AgentState) -> Dict[str, Any]:
        """Select relevant tables using LLM"""
        logger.info("Selecting tables...")
        
        # Store previous selection and failures
        previous_selected = state.get('selected_tables', [])
        previous_failures = state.get('previous_failures', [])
        all_failed_tables = state.get('all_failed_tables', set())
        
        # Get iteration from state (DO NOT INCREMENT HERE)
        iteration = state.get('iteration', 0)
        
        # Get intent
        intent = self._intent_from_state(state['intent'])
        
        # Check which tables have samples cached
        tables_with_samples = []
        tables_without_samples = []
        
        for table_name in self.metadata:
            if self.sample_cache.get(table_name) is not None:
                tables_with_samples.append(table_name)
            else:
                tables_without_samples.append(table_name)
        
        logger.info(f"Tables with cached samples: {len(tables_with_samples)}, without: {len(tables_without_samples)}")
        
        # Build retry context with failure information
        retry_context = None
        if iteration > 0 and previous_selected:
            # Add current failed tables to the set
            for table_name, _ in previous_selected:
                all_failed_tables.add(table_name)
            
            retry_context = {
                'previous_tables': list(all_failed_tables),  # All failed tables from all iterations
                'retry_reason': state.get('validation_checks', {}).get('issues', []),
                'iteration': iteration,  # Use iteration from state
                'previous_failures': previous_failures or []
            }
            
            # Add the failed query and its result to context
            if state.get('sql_query'):
                retry_context['failed_query'] = state['sql_query']
                retry_context['failed_result'] = {
                    'row_count': state.get('query_result', {}).get('row_count', 0),
                    'error': state.get('query_result', {}).get('error')
                }
            
            logger.info(f"Retry context - Iteration: {iteration}, Failed tables so far: {list(all_failed_tables)}")
        
        # Pass graph builder to table selector
        self.table_selector_llm.relationship_graph = self.relationship_graph
        if hasattr(self, 'graph_builder'):
            self.table_selector_llm.graph_builder = self.graph_builder
        
        # Use LLM to select tables - pass ALL metadata, not filtered
        selected_tables = self.table_selector_llm.select_tables(
            state['user_prompt'],
            intent,
            self.metadata,  # Pass all tables
            state['similar_queries'],
            retry_context=retry_context
        )
        
        # Filter out any previously failed tables (safety check)
        if all_failed_tables:
            filtered_tables = []
            for table_name, confidence in selected_tables:
                if table_name in all_failed_tables:
                    logger.warning(f"Removing previously failed table {table_name} from selection")
                else:
                    filtered_tables.append((table_name, confidence))
            selected_tables = filtered_tables
        
        # Prepare table info for selected tables
        table_info = {}
        for table_name, confidence in selected_tables[:20]:  # Limit to top 20
            if table_name in self.metadata:
                table_info[table_name] = self.metadata[table_name]
            else:
                logger.warning(f"Selected table '{table_name}' not found in metadata")
        
        # Track failures for retry context
        if iteration > 0:
            if not previous_failures:
                previous_failures = []
            previous_failures.append({
                'iteration': iteration,
                'tables': [t[0] for t in previous_selected],
                'issues': state.get('validation_checks', {}).get('issues', [])
            })
        
        logger.info(f"Selected {len(selected_tables)} tables")
        
        return {
            'selected_tables': selected_tables,
            'table_info': table_info,
            'iteration': iteration,  # Pass through, don't modify
            'previous_selected_tables': previous_selected,
            'previous_failures': previous_failures,
            'all_failed_tables': all_failed_tables
        }
    
    async def fetch_samples(self, state: AgentState) -> AgentState:
        """Fetch sample data for selected tables using universal fetch function."""
        logger.info("Fetching sample data...")
        sample_data = {}
        
        force_refresh = True
        
        for table_name, _ in state['selected_tables'][:config.MAX_TABLES_TO_SAMPLE]:
            sample = await self.fetch_table_head(
                table_name, 
                config.SAMPLE_SIZE, 
                force_refresh=force_refresh
            )
            
            if sample:
                sample_data[table_name] = sample
        
        state['sample_data'] = sample_data
        logger.info(f"Fetched samples for {len(sample_data)} tables")
        
        # Only generate summaries if enabled and missing
        if config.SUMMARIZATION_ENABLED:
            logger.info("Checking for missing summaries...")
            
            # Check table summaries
            tables_needing_summary = []
            for table_name in sample_data.keys():
                if table_name not in self.table_summaries:
                    logger.info(f"Table {table_name} needs summary generation")
                    table_info = self.metadata.get(table_name)
                    if table_info:
                        tables_needing_summary.append((table_name, table_info))
            
            # Check column summaries
            columns_needing_summary = []
            for table_name in sample_data.keys():
                table_info = self.metadata.get(table_name)
                if table_info:
                    columns = table_info.columns if hasattr(table_info, 'columns') else table_info.get('columns', [])
                    for col in columns[:10]:  # Limit to first 10 columns for performance
                        col_name = col.get('name') if isinstance(col, dict) else getattr(col, 'name', None)
                        if col_name:
                            col_key = f"{table_name}.{col_name}"
                            if col_key not in self.column_summaries:
                                logger.debug(f"Column {col_key} needs summary generation")
                                columns_needing_summary.append((table_name, col, table_info))
            
            # Generate column summaries first (tables depend on them)
            if columns_needing_summary:
                logger.info(f"Generating summaries for {len(columns_needing_summary)} columns...")
                await self._generate_column_summaries_for_fetch(columns_needing_summary, sample_data)
                
                # Reload column summaries after generation
                self.column_summaries = self.persistence_manager.get_column_summaries(self.target_db)
                logger.info(f"Column summaries updated. Total: {len(self.column_summaries)}")
            
            # Generate table summaries after columns are done
            if tables_needing_summary:
                logger.info(f"Generating summaries for {len(tables_needing_summary)} tables...")
                await self._generate_table_summaries_for_fetch(tables_needing_summary, sample_data)
                
                # Reload table summaries
                self.table_summaries = self.persistence_manager.get_table_summaries(self.target_db)
                logger.info(f"Table summaries updated. Total: {len(self.table_summaries)}")
            
            # Update components with new summaries
            if columns_needing_summary or tables_needing_summary:
                self.table_selector_llm.table_summaries = self.table_summaries
                self.table_selector_llm.column_summaries = self.column_summaries
                self.sql_generator.table_summaries = self.table_summaries
                self.sql_generator.column_summaries = self.column_summaries
                logger.info("Updated LLM components with new summaries")
        
        return state
    

    async def _generate_column_summaries_for_fetch(self, columns_needing_summary: List[Tuple[str, Dict, Dict]], 
                                                sample_data: Dict[str, Dict[str, Any]]):
        """Generate column summaries during fetch process."""
        async def generate_column_summary_task(table_name, col_info, table_info):
            try:
                # Get table DDL
                table_ddl = table_info.ddl if hasattr(table_info, 'ddl') else None
                
                # Get sample data for the table
                table_sample = sample_data.get(table_name, {})
                
                # Get column stats
                col_name = col_info.get('name') if isinstance(col_info, dict) else getattr(col_info, 'name')
                stats = await asyncio.to_thread(
                    self.schema_analyzer.get_column_stats, table_name, col_name
                )
                
                # Extract sample values for this column
                col_samples = []
                if table_sample and table_sample.get('rows'):
                    col_samples = [row.get(col_name) for row in table_sample['rows']
                                if row.get(col_name) is not None]
                
                # Get all columns for peer analysis
                all_columns = table_info.columns if hasattr(table_info, 'columns') else table_info.get('columns', [])
                
                # Generate summary with full context
                result = await asyncio.to_thread(
                    self.schema_summarizer.summarize_column,
                    table_name, col_info, col_samples, stats, all_columns,
                    table_ddl, table_sample
                )
                
                # Check if summary was generated successfully
                if result[0] is not None and result[1] is not None:
                    logger.debug(f"✓ Generated summary for column {table_name}.{col_name}")
                    return True
                else:
                    logger.warning(f"Summary generation returned None for {table_name}.{col_name}")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to generate column summary for {table_name}.{col_name}: {e}")
                return False
        
        # Process columns sequentially to avoid overwhelming the system
        successful = 0
        failed = 0
        
        for table_name, col_info, table_info in columns_needing_summary:
            result = await generate_column_summary_task(table_name, col_info, table_info)
            if result:
                successful += 1
            else:
                failed += 1
        
        logger.info(f"Column summary generation complete. Successful: {successful}, Failed: {failed}")

    async def _generate_table_summaries_for_fetch(self, tables_needing_summary: List[Tuple[str, Dict]], 
                                                sample_data: Dict[str, Dict[str, Any]]):
        """Generate table summaries during fetch process."""
        async def generate_table_summary_task(table_name, table_info):
            try:
                # Check if we have column summaries for this table
                columns = table_info.columns if hasattr(table_info, 'columns') else table_info.get('columns', [])
                column_count = sum(1 for col in columns
                                if f"{table_name}.{col.get('name', '')}" in self.column_summaries)
                
                if column_count == 0:
                    logger.warning(f"No column summaries found for {table_name}, skipping table summary")
                    return False
                
                # Get sample data
                sample = sample_data.get(table_name, {})
                
                # Generate table summary (will use existing column summaries)
                result = await asyncio.to_thread(
                    self.schema_summarizer.summarize_table,
                    table_name, table_info, sample
                )
                
                # Check if summary was generated successfully
                if result[0] is not None and result[1] is not None:
                    logger.debug(f"✓ Generated summary for table {table_name}")
                    return True
                else:
                    logger.warning(f"Summary generation returned None for table {table_name}")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to generate table summary for {table_name}: {e}")
                return False
        
        # Process tables sequentially
        successful = 0
        failed = 0
        
        for table_name, table_info in tables_needing_summary:
            result = await generate_table_summary_task(table_name, table_info)
            if result:
                successful += 1
            else:
                failed += 1
        
        logger.info(f"Table summary generation complete. Successful: {successful}, Failed: {failed}")

    async def generate_sql(self, state: AgentState) -> AgentState:
        """Generate SQL query"""
        logger.info("Generating SQL query...")
        intent = self._intent_from_state(state['intent'])
        
        # Pass relationship graph AND summaries to SQL generator
        self.sql_generator.relationship_graph = self.relationship_graph
        self.sql_generator.table_summaries = self.table_summaries
        self.sql_generator.column_summaries = self.column_summaries
        
        # Build retry context for SQL generator
        retry_context = None
        if state.get('iteration', 0) > 0:
            retry_context = {
                'previous_failures': state.get('previous_failures', []),
                'validation_issues': state.get('validation_checks', {}).get('issues', []),
                'previous_query': state.get('sql_query'),
                'previous_result': state.get('query_result', {})
            }
        
        sql_query, confidence = self.sql_generator.generate_sql(
            state['user_prompt'],
            intent,
            state['table_info'],
            state['similar_queries'],
            state['sample_data'],
            retry_context=retry_context  # Pass retry context
        )
        
        state['sql_query'] = sql_query
        state['query_confidence'] = confidence
        logger.info(f"Generated SQL with confidence: {confidence:.2f}")
        logger.debug(f"SQL: {sql_query}")
        return state

    async def execute_query(self, state: AgentState) -> AgentState:
        """Execute the SQL query"""
        logger.info("Executing query...")

        intent = self._intent_from_state(state['intent'])

        query_result = self.query_executor.execute(
            state['sql_query'],
            intent
        )

        # Store result in state
        state['query_result'] = {
            'data': query_result.data.to_dict('records') if not query_result.data.empty else [],
            'columns': list(query_result.data.columns) if not query_result.data.empty else [],
            'row_count': len(query_result.data),
            'error': query_result.error,
            'execution_time': query_result.execution_time,
            'validation_score': query_result.validation_score,
            'validation_checks': query_result.validation_checks
        }

        logger.info(f"Query executed. Rows: {len(query_result.data)}, Error: {query_result.error}")

        return state

    async def validate_results(self, state: AgentState) -> AgentState:
        """Validate query results using LLM"""
        logger.info("Validating results...")
        
        intent = self._intent_from_state(state['intent'])
        query_result = self._query_result_from_state(state['query_result'])
        
        # Use the new LLM validator
        validation_score, validation_details = self.result_validator_llm.validate(
            query_result,
            intent,
            state['user_prompt'],
            state['sql_query']
            )
        
        state['validation_score'] = validation_score
        state['validation_checks'] = validation_details
        
        logger.info(f"Validation score: {validation_score:.2f}")
        if validation_details.get('issues'):
            logger.info(f"Issues found: {validation_details['issues']}")
        
        return state

    def should_retry(self, state: AgentState) -> str:
        """Determine if query should be retried based on validation"""
        current_iteration = state.get('iteration', 0)
        max_iterations = state.get('max_iterations', 3)
        validation_details = state.get('validation_checks', {})
        
        # Check if we've exceeded max iterations
        if current_iteration >= max_iterations - 1:
            logger.info(f"Reached maximum iterations ({max_iterations}), stopping retries")
            return "continue"
        
        # Check if validation score is too low
        validation_score = state.get('validation_score', 0)
        if validation_score < 0.6:
            retry_strategy = validation_details.get('retry_strategy', 'none')
            
            # Check if we're stuck in a loop with the same table
            selected_tables = state.get('selected_tables', [])
            previous_tables = state.get('previous_selected_tables', [])
            
            if selected_tables and previous_tables:
                current_table_names = {t[0] for t in selected_tables}
                previous_table_names = {t[0] for t in previous_tables}
                
                if current_table_names == previous_table_names:
                    logger.warning("Same tables selected again - forcing different strategy")
                    retry_strategy = 'change_tables'
            
            # Check if we've exhausted too many tables
            all_failed_tables = state.get('all_failed_tables', set())
            if len(all_failed_tables) > 5:
                logger.warning(f"Already tried {len(all_failed_tables)} tables without success")
                # Maybe the data doesn't exist
                if validation_score < 0.3:
                    logger.info("Very low validation score after multiple attempts - data might not exist")
                    return "continue"
            
            if retry_strategy != 'none':
                # INCREMENT ITERATION HERE
                state['iteration'] = current_iteration + 1
                logger.info(f"Validation failed (score: {validation_score:.2f}). "
                        f"Retrying with strategy: {retry_strategy} (iteration {state['iteration']})")
                return "retry"
        
        return "continue"

    async def generate_visualization(self, state: AgentState) -> AgentState:
        """Generate visualization if appropriate"""
        logger.info("Generating visualization...")

        if state['query_result']['error'] or state['query_result']['row_count'] == 0:
            state['visualization'] = None
            return state

        # Convert data back to DataFrame
        df = pd.DataFrame(state['query_result']['data'])
        intent = self._intent_from_state(state['intent'])

        visualization = self.chart_generator.generate_visualization(df, intent)
        state['visualization'] = visualization

        if visualization:
            logger.info(f"Generated {visualization['type']} chart")

        return state
    
    async def generate_explanation(self, state: AgentState) -> AgentState:
        """Generate comprehensive explanation using LLM"""
        logger.info("Generating explanation...")
        
        # Build enhanced context with summaries
        context = {
            'user_prompt': state['user_prompt'],
            'intent': state['intent'],
            'sql_query': state['sql_query'],
            'tables_used': [t for t, _ in state['selected_tables'][:5]] if state.get('selected_tables') else [],
            'row_count': state['query_result']['row_count'],
            'columns': state['query_result']['columns'],
            'sample_data': state['query_result']['data'][:3] if state['query_result']['data'] else [],
            'validation_score': state['validation_score'],
            'validation_details': state['validation_checks'],
            'confidence': state['query_confidence'],
            'execution_time': state['query_result']['execution_time']
        }
        
        # ADD TABLE AND COLUMN SUMMARIES
        table_contexts = []
        column_contexts = []
        
        for table_name, _ in state['selected_tables'][:5]:
            # Add table summary
            if table_name in self.table_summaries:
                table_summary = self.table_summaries[table_name]['summary']
                table_contexts.append({
                    'table': table_name,
                    'purpose': table_summary.get('purpose', ''),
                    'type': table_summary.get('table_type', ''),
                    'grain': table_summary.get('grain', '')
                })
            
            # Add column summaries for result columns
            for col_name in state['query_result']['columns']:
                col_key = f"{table_name}.{col_name}"
                if col_key in self.column_summaries:
                    col_summary = self.column_summaries[col_key]['summary']
                    column_contexts.append({
                        'column': f"{table_name}.{col_name}",
                        'canonical_name': col_summary.get('canonical_name', col_name),
                        'description': col_summary.get('description', ''),
                        'semantic_role': col_summary.get('semantic_role', ''),
                        'unit': col_summary.get('unit', '')
                    })

        try:
            prompt = f"""
    Generate a clear, concise explanation of these query results for a non-technical user.
    IMPORTANT: Respond in the same language as the user's question. If the user asked in Persian/Farsi, respond in Persian. If in English, respond in English.

    User asked: "{context['user_prompt']}"

    Table Context:
    {chr(10).join([f"- {tc['table']}: {tc['purpose']} (Type: {tc['type']}, Grain: {tc['grain']})" for tc in table_contexts])}

    Column Context:
    {chr(10).join([f"- {cc['column']} ({cc['canonical_name']}): {cc['description']} [Role: {cc['semantic_role']}]" + (f" Unit: {cc['unit']}" if cc['unit'] else "") for cc in column_contexts])}

    Results summary:
    - Found {context['row_count']} rows
    - Columns: {', '.join(context['columns'])}
    - Query confidence: {context['confidence']:.0%}
    - Validation score: {context['validation_score']:.0%}
    - Execution time: {context['execution_time']:.2f} seconds

    Sample data: {json.dumps(context['sample_data'], default=str)}

    SQL Query executed:
    {context['sql_query']}

    IMPORTANT: Use the table and column context to provide meaningful explanations about what the data represents.

    ALWAYS PROVIDE THESE:
    1. A one paragraph summary of what was found using business terms from the summaries
    2. Key insights from the data (if any patterns are visible)
    3. Any caveats or limitations
    4. Confidence assessment in plain language
    5. If the row count is high, mention that only a sample is shown
    """
            
            response = self.sql_generator.client.chat.completions.create(
                model='gpt-4.1',
                messages=[
                    {"role": "system", "content": "You are a helpful multilingual data analyst who explains technical results in business terms. Always respond in the same language as the user's question."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            explanation = response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM explanation failed: {e}")
            # Enhanced fallback explanation with summaries
            explanation = self._generate_fallback_explanation_with_summaries(context, table_contexts, column_contexts)
        
        state['explanation'] = explanation
        return state

    def _generate_fallback_explanation(self, context: Dict[str, Any]) -> str:
        """Generate basic explanation when LLM fails"""
        parts = []
        
        # Summary
        if context['row_count'] == 0:
            parts.append("No results found for your query.")
        elif context['row_count'] == 1:
            parts.append(f"Found 1 result matching your query.")
        else:
            parts.append(f"Found {context['row_count']} results matching your query.")
        
        # Add validation info
        if context['validation_score'] < 0.7:
            issues = context['validation_details'].get('issues', [])
            if issues:
                parts.append(f"Note: {', '.join(issues[:2])}")
        
        # Confidence
        if context['confidence'] < 0.7:
            parts.append("Results may need verification due to moderate confidence.")
        
        return " ".join(parts)

    async def synthesize_response(self, state: AgentState) -> Dict[str, Any]:
        """Enhanced response synthesis with detailed insights"""
        logger.info("Synthesizing response...")
        
        # Check if there was an error
        has_error = state['query_result'].get('error') is not None
        
        # Build comprehensive summary using LLM only if no error
        if not has_error:
            try:
                summary_prompt = f"""
                Create a response summary for this SQL query execution:
                
                Query: "{state['user_prompt']}"
                SQL executed: {state['sql_query']}
                Tables used: {', '.join([t for t, _ in state['selected_tables'][:5]])}
                Result count: {state['query_result']['row_count']} rows
                Execution time: {state['query_result']['execution_time']:.2f}s
                Validation score: {state['validation_score']:.0%}
                
                Provide a JSON response with:
                {{
                    "summary": "one line summary of what was accomplished",
                    "insights": ["list of 2-3 key insights from the results"],
                    "quality_assessment": "brief assessment of result quality",
                    "next_steps": ["suggestions for follow-up queries if applicable"]
                }}
                """
                
                response = self.sql_generator.client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": "You are a data analyst providing query result summaries."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=400
                )
                
                response_text = response.choices[0].message.content.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                summary_data = json.loads(response_text.strip())
                
            except Exception as e:
                logger.error(f"Failed to generate summary: {e}")
                summary_data = {
                    'summary': state['explanation'],
                    'insights': [],
                    'quality_assessment': f"Validation score: {state['validation_score']:.0%}",
                    'next_steps': []
                }
        else:
            # Error case - create error summary
            summary_data = {
                'summary': f"Query failed: {state['query_result']['error']}",
                'insights': [],
                'quality_assessment': "Query execution failed",
                'next_steps': ["Check date format compatibility", "Verify column data types"]
            }
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(state)
        
        # Adjust confidence for errors
        if has_error:
            overall_confidence = 0.0
        
        # Handle datetime serialization
        data = state['query_result']['data']
        if data:
            serialized_data = []
            for row in data:
                serialized_row = {}
                for key, value in row.items():
                    if hasattr(value, 'isoformat'):
                        serialized_row[key] = value.isoformat()
                    elif isinstance(value, (datetime, date)):
                        serialized_row[key] = str(value)
                    else:
                        serialized_row[key] = value
                serialized_data.append(serialized_row)
        else:
            serialized_data = data
        
        # Build final response
        response = {
            'success': not has_error,  # Changed from checking error is None
            'data': serialized_data,
            'columns': state['query_result']['columns'],
            'row_count': state['query_result']['row_count'],
            'sql_query': state['sql_query'],
            'visualization': state['visualization'],
            'explanation': state['explanation'],
            'summary': summary_data.get('summary', state['explanation']),
            'insights': summary_data.get('insights', []),
            'quality_assessment': summary_data.get('quality_assessment', ''),
            'next_steps': summary_data.get('next_steps', []),
            'confidence': overall_confidence,
            'execution_time': state['query_result']['execution_time'],
            'tables_used': [t for t, _ in state['selected_tables'][:5]] if state.get('selected_tables') else [],
            'validation_details': state.get('validation_checks', {})
        }
        
        # Add error details if present
        if has_error:
            response['error'] = state['query_result']['error']
            response['warning'] = "Query execution failed. Please check the error details."
        elif overall_confidence < 0.7:
            response['warning'] = "Moderate confidence in results. Please review carefully."
        
        state['response'] = response
        state['overall_confidence'] = overall_confidence
        
        return state
    
    async def learn_from_query(self, state: AgentState) -> Dict[str, Any]:
        """Learn from the query execution"""
        logger.info("Learning from query...")

        self.persistence_manager.update_agent_state(
            self.target_db,
            total_queries=self.persistence_manager.get_agent_state(self.target_db)['total_queries'] + 1
        )

        # Create QueryMemory object
        query_memory = QueryMemory(
            id=f"query_{datetime.now().timestamp()}",
            user_prompt=state['user_prompt'],
            prompt_embedding=self.embedding_service.embed_text(state['user_prompt']).tolist(),
            intent_summary=state['intent'],
            sql_query=state['sql_query'],
            tables_used=[t for t, _ in state['selected_tables'][:5]] if state.get('selected_tables') else [],
            result_summary=state['explanation'],
            success_score=0.0, # Will be calculated
            timestamp=datetime.now(),
            execution_time=state['query_result']['execution_time'],
            retry_count=state['iteration'],
            validation_checks=state['validation_checks'],
            query_complexity=QueryComplexity(state['intent']['query_complexity'])
        )

        # Create query result for learning
        query_result = self._query_result_from_state(state['query_result'])
        query_result.validation_score = state['validation_score']
        query_result.validation_checks = state['validation_checks']
        query_result.used_sample_data = bool(state['sample_data'])

        # Process for learning
        self.learning_pipeline.process_query_result(
            query_memory,
            query_result,
            state['sample_data']
        )

        if query_memory.success_score > 0.8:
            current_state = self.persistence_manager.get_agent_state(self.target_db)
            self.persistence_manager.update_agent_state(
                self.target_db,
                successful_queries=current_state['successful_queries'] + 1
            )

        # Return only the fields this node updates
        return {}

    def _intent_from_state(self, intent_dict: Dict[str, Any]) -> Any:
        """Convert intent dictionary back to Intent object"""
        from models import Intent, QueryIntent, QueryComplexity

        return Intent(
            intent_type=QueryIntent(intent_dict['intent_type']),
            entities=intent_dict['entities'],
            metrics=intent_dict['metrics'],
            filters=intent_dict['filters'],
            time_range=intent_dict['time_range'],
            expected_result_type=intent_dict['expected_result_type'],
            expects_single_value=intent_dict['expects_single_value'],
            expects_list=intent_dict['expects_list'],
            expects_aggregation=intent_dict['expects_aggregation'],
            intent_confidence=intent_dict['intent_confidence'],
            query_complexity=QueryComplexity(intent_dict['query_complexity'])
        )

    def _query_result_from_state(self, result_dict: Dict[str, Any]) -> Any:
        """Convert result dictionary to QueryResult object"""
        from models import QueryResult

        return QueryResult(
            sql="", # Not needed for validation
            data=pd.DataFrame(result_dict['data']),
            execution_time=result_dict['execution_time'],
            error=result_dict['error']
        )

    def _get_query_examples(self, user_prompt: str,
                            query_history: List[QueryMemory]) -> List[Dict[str, Any]]:
        """Get similar successful queries from history"""
        prompt_embedding = self.embedding_service.embed_text(user_prompt)

        similar_queries = []
        for memory in query_history:
            if memory.success_score > 0.8: # Only use successful queries
                similarity = self.embedding_service.compute_similarity(
                    prompt_embedding,
                    np.array(memory.prompt_embedding)
                )

                similar_queries.append({
                    'prompt': memory.user_prompt,
                    'sql': memory.sql_query,
                    'tables': memory.tables_used,
                    'similarity': similarity,
                    'success_score': memory.success_score
                })

        # Sort by similarity * success_score
        similar_queries.sort(
            key=lambda x: x['similarity'] * x['success_score'],
            reverse=True
        )

        return similar_queries[:5] # Return top 5

    def _calculate_overall_confidence(self, state: AgentState) -> float:
        """Calculate overall response confidence"""
        query_confidence = state.get('query_confidence', 0.5)
        validation_score = state.get('validation_score', 0.5)
        retry_penalty = 1.0 if state['iteration'] == 0 else 0.5
        viz_score = state['visualization']['score'] if state.get('visualization') else 0.7

        overall_confidence = (
            0.3 * query_confidence +
            0.3 * validation_score +
            0.2 * viz_score +
            0.2 * retry_penalty
        )

        return overall_confidence

    async def process_query(self, user_prompt: str) -> Dict[str, Any]:
        """Main entry point for processing a query"""
        logger.info(f"Starting query processing for: {user_prompt[:50]}...")
        
        # Initialize state
        initial_state = {
            'user_prompt': user_prompt,
            'iteration': 0,
            'max_iterations': config.MAX_RETRIES
        }
        
        # Run the graph with recursion limit config
        try:
            # LangGraph uses runtime config, not compile-time config
            runtime_config = {"recursion_limit": 50}
            
            # Add timeout handling
            import asyncio
            logger.info("Invoking graph...")
            
            # Set a timeout for the entire graph execution
            final_state = await asyncio.wait_for(
                self.graph.ainvoke(initial_state, config=runtime_config),
                timeout=120.0  # 2 minutes timeout
            )
            
            logger.info("Graph execution completed successfully")
            return final_state['response']
            
        except asyncio.TimeoutError:
            logger.error("Query processing timed out after 120 seconds")
            return {
                'success': False,
                'error': 'Query processing timed out. Please try a simpler query.',
                'confidence': 0.0
            }
        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'confidence': 0.0
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        # Get cache stats from persistence manager
        cache_stats = {'entries': 0, 'total_hits': 0, 'avg_hits': 0}
        if hasattr(self, 'sample_cache') and self.sample_cache:
            cache_stats = self.sample_cache.get_stats()
        else:
            # Get cache stats directly from database
            try:
                conn = self.persistence_manager.vec_store._connection
                cursor = conn.execute("""
                    SELECT COUNT(*) as entries,
                    COALESCE(SUM(hits), 0) as total_hits,
                    COALESCE(AVG(hits), 0) as avg_hits
                    FROM sample_cache_meta
                    WHERE target_db = ?
                """, (self.target_db,))
                row = cursor.fetchone()
                if row:
                    cache_stats = {
                        'entries': row['entries'] or 0,
                        'total_hits': row['total_hits'] or 0,
                        'avg_hits': float(row['avg_hits'] or 0)
                    }
            except Exception as e:
                logger.error(f"Failed to get cache stats: {e}")
        # Count tables with samples
        tables_with_samples = 0
        if hasattr(self, 'metadata'):
            for table_name in self.metadata:
                if self.persistence_manager.get_sample_data(self.target_db, table_name):
                    tables_with_samples += 1
        # UPDATED: Count schema summaries from separate sources
        summary_stats = {
            'total_summaries': len(self.table_summaries) + len(self.column_summaries),
            'table_summaries': len(self.table_summaries),
            'column_summaries': len(self.column_summaries),
            'high_confidence_summaries': 0
        }
        # Count high confidence summaries
        for summary_data in self.table_summaries.values():
            if summary_data['confidence'] >= 0.8:
                summary_stats['high_confidence_summaries'] += 1
        for summary_data in self.column_summaries.values():
            if summary_data['confidence'] >= 0.8:
                summary_stats['high_confidence_summaries'] += 1
        # Get typed edge statistics
        typed_edge_stats = {}
        if hasattr(self, 'relationship_graph') and hasattr(self.relationship_graph, 'edges'):
            for edge in self.relationship_graph.edges.values():
                edge_type = edge.relation_type
                if edge_type not in typed_edge_stats:
                    typed_edge_stats[edge_type] = 0
                typed_edge_stats[edge_type] += 1
        # Build relationship graph stats
        graph_stats = self.relationship_graph.get_statistics() if hasattr(self, 'relationship_graph') else {}
        graph_stats['typed_edges'] = typed_edge_stats
        return {
            'metadata': {
                'total_tables': len(self.metadata) if hasattr(self, 'metadata') else 0,
                'tables_with_samples': tables_with_samples
            },
            'schema_summaries': summary_stats,
            'relationship_graph': graph_stats,
            'learning': self.learning_pipeline.get_learning_statistics() if hasattr(self, 'learning_pipeline') else {},
            'cache': cache_stats
        }

    async def ensure_summaries_updated(self, max_age_days: int = 30, max_concurrent: int = 5):
        """Ensure all summaries are up-to-date."""
        if not config.SUMMARIZATION_ENABLED:
            logger.info("Summarization is disabled in configuration")
            return
        logger.info("Checking schema summaries...")
        
        # Step 1: First check and generate COLUMN summaries
        columns_needing_summary = []
        for table_name, table_info in self.metadata.items():
            columns = table_info.columns if hasattr(table_info, 'columns') else table_info.get('columns', [])
            for col in columns:
                col_name = col.get('name') if isinstance(col, dict) else getattr(col, 'name', None)
                if col_name:
                    col_key = f"{table_name}.{col_name}"
                    col_summary = self.column_summaries.get(col_key)
                    
                    if not col_summary:
                        columns_needing_summary.append((table_name, col, table_info))
                    elif col_summary:
                        # Check staleness
                        last_updated = datetime.fromisoformat(col_summary['last_updated'])
                        if (datetime.now() - last_updated).days > max_age_days:
                            columns_needing_summary.append((table_name, col, table_info))
        
        # Generate missing column summaries FIRST and WAIT for completion
        if columns_needing_summary:
            await self._generate_column_summaries(columns_needing_summary, max_concurrent)
            # IMPORTANT: Reload column summaries after generation
            self.column_summaries = self.persistence_manager.get_column_summaries(self.target_db)
            logger.info(f"Column summaries updated. Total: {len(self.column_summaries)}")
        
        # Step 2: THEN check and generate TABLE summaries
        tables_needing_summary = []
        for table_name, table_info in self.metadata.items():
            # Check if we have column summaries for this table
            table_columns = [col.get('name') if isinstance(col, dict) else getattr(col, 'name', None) 
                            for col in (table_info.columns if hasattr(table_info, 'columns') else table_info.get('columns', []))]
            
            # Count how many column summaries we have for this table
            column_summary_count = sum(1 for col_name in table_columns 
                                    if col_name and f"{table_name}.{col_name}" in self.column_summaries)
            
            # Only generate table summary if we have column summaries
            if column_summary_count > 0:
                table_summary = self.table_summaries.get(table_name)
                
                if not table_summary:
                    tables_needing_summary.append((table_name, table_info))
                elif table_summary:
                    # Check staleness
                    last_updated = datetime.fromisoformat(table_summary['last_updated'])
                    if (datetime.now() - last_updated).days > max_age_days:
                        tables_needing_summary.append((table_name, table_info))
            else:
                logger.warning(f"Skipping table {table_name} - no column summaries available")
        
        # Generate table summaries AFTER columns are done
        if tables_needing_summary:
            await self._generate_table_summaries(tables_needing_summary, max_concurrent)
            # Reload table summaries
            self.table_summaries = self.persistence_manager.get_table_summaries(self.target_db)
        
        # Log final status
        logger.info(f"Summary update complete. Tables: {len(self.table_summaries)}, Columns: {len(self.column_summaries)}")

    async def _generate_column_summaries(self, columns_needing_summary: List[Tuple[str, Dict, Dict]], max_concurrent: int):
        """Generate column summaries in batches."""
        logger.info(f"Generating {len(columns_needing_summary)} column summaries...")
        
        async def generate_column_summary_task(table_name, col_info, table_info):
            max_retries = 3
            retry_delay = 2.0
            
            for attempt in range(max_retries):
                try:
                    # Get table DDL
                    table_ddl = table_info.ddl if hasattr(table_info, 'ddl') else None
                    
                    # Get sample data for the table
                    table_sample = await asyncio.to_thread(
                        self.connection.get_table_sample, table_name, 10
                    )
                    
                    # Get column stats
                    col_name = col_info.get('name') if isinstance(col_info, dict) else getattr(col_info, 'name')
                    stats = await asyncio.to_thread(
                        self.schema_analyzer.get_column_stats, table_name, col_name
                    )
                    
                    # Extract sample values for this column
                    col_samples = []
                    if table_sample and table_sample.get('rows'):
                        col_samples = [row.get(col_name) for row in table_sample['rows'] 
                                    if row.get(col_name) is not None]
                    
                    # Get all columns for peer analysis
                    all_columns = table_info.columns if hasattr(table_info, 'columns') else table_info.get('columns', [])
                    
                    # Generate summary with full context
                    result = await asyncio.to_thread(
                        self.schema_summarizer.summarize_column,
                        table_name, col_info, col_samples, stats, all_columns,
                        table_ddl, table_sample
                    )
                    
                    # Check if summary was generated successfully
                    if result[0] is not None and result[1] is not None:
                        logger.info(f"✓ Generated summary for column {table_name}.{col_name}")
                        return True
                    else:
                        logger.warning(f"Summary generation returned None for {table_name}.{col_name}")
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying in {retry_delay} seconds... (attempt {attempt + 2}/{max_retries})")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            logger.error(f"Failed to generate summary for {table_name}.{col_name} after {max_retries} attempts")
                            return False
                            
                except Exception as e:
                    logger.error(f"Error in attempt {attempt + 1} for {table_name}.{col_name}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.error(f"Failed to generate column summary after {max_retries} attempts: {e}")
                        return False
            
            return False
        
        # Process in batches
        successful = 0
        failed = 0
        
        for i in range(0, len(columns_needing_summary), max_concurrent):
            batch = columns_needing_summary[i:i + max_concurrent]
            tasks = [generate_column_summary_task(t[0], t[1], t[2]) for t in batch]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                    logger.error(f"Task exception: {result}")
                elif result:
                    successful += 1
                else:
                    failed += 1
        
        # Reload column summaries
        self.column_summaries = self.persistence_manager.get_column_summaries(self.target_db)
        
        logger.info(f"Column summary generation complete. Successful: {successful}, Failed: {failed}")

    async def _generate_table_summaries(self, tables_needing_summary: List[Tuple[str, Dict]], max_concurrent: int):
        """Generate table summaries in batches."""
        logger.info(f"Generating {len(tables_needing_summary)} table summaries...")
        
        async def generate_table_summary_task(table_name, table_info):
            max_retries = 3
            retry_delay = 2.0
            
            for attempt in range(max_retries):
                try:
                    # Check if we have column summaries for this table
                    columns = table_info.columns if hasattr(table_info, 'columns') else table_info.get('columns', [])
                    column_count = sum(1 for col in columns
                                    if f"{table_name}.{col.get('name', '')}" in self.column_summaries)
                    
                    if column_count == 0:
                        logger.warning(f"No column summaries found for {table_name}, skipping table summary")
                        return False
                    
                    # Get sample data
                    sample = await asyncio.to_thread(
                        self.connection.get_table_sample, table_name, 10
                    )
                    
                    # Generate table summary (will use existing column summaries)
                    result = await asyncio.to_thread(
                        self.schema_summarizer.summarize_table,
                        table_name, table_info, sample
                    )
                    
                    # Check if summary was generated successfully
                    if result[0] is not None and result[1] is not None:
                        logger.info(f"✓ Generated summary for table {table_name}")
                        return True
                    else:
                        logger.warning(f"Summary generation returned None for table {table_name}")
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying in {retry_delay} seconds... (attempt {attempt + 2}/{max_retries})")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            logger.error(f"Failed to generate summary for table {table_name} after {max_retries} attempts")
                            return False
                            
                except Exception as e:
                    logger.error(f"Error in attempt {attempt + 1} for table {table_name}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.error(f"Failed to generate table summary after {max_retries} attempts: {e}")
                        return False
            
            return False

        # Process in batches
        successful = 0
        failed = 0
        
        for i in range(0, len(tables_needing_summary), max_concurrent):
            batch = tables_needing_summary[i:i + max_concurrent]
            tasks = [generate_table_summary_task(t[0], t[1]) for t in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                    logger.error(f"Task exception: {result}")
                elif result:
                    successful += 1
                else:
                    failed += 1

        # Reload table summaries
        self.table_summaries = self.persistence_manager.get_table_summaries(self.target_db)
        logger.info(f"Table summary generation complete. Successful: {successful}, Failed: {failed}")
    
    async def ensure_relationships_updated(self, max_age_days: int = 30):
        """Build/update relationship graph if needed"""
        logger.info("Checking relationship graph...")
        
        # Check if graph needs update
        needs_update = False
        if self.relationship_graph.graph.number_of_edges() == 0:
            logger.info("Relationship graph is empty, needs building")
            needs_update = True
        else:
            # Check staleness - get the oldest edge timestamp
            oldest_edge_time = datetime.now()
            for edge in self.relationship_graph.edges.values():
                if edge.usage_contexts:
                    for ctx in edge.usage_contexts:
                        if 'timestamp' in ctx:
                            edge_time = ctx['timestamp']
                            if isinstance(edge_time, str):
                                edge_time = datetime.fromisoformat(edge_time)
                            if edge_time < oldest_edge_time:
                                oldest_edge_time = edge_time
            
            age_days = (datetime.now() - oldest_edge_time).days
            if age_days > max_age_days:
                logger.info(f"Relationship graph is {age_days} days old, needs update")
                needs_update = True
        
        if needs_update:
            logger.info("Building relationship graph...")
            await self._build_relationships_async()
            logger.info(f"Relationship graph updated with {self.relationship_graph.graph.number_of_edges()} edges")

    async def _build_relationships_async(self):
        """Build relationships using the new GraphBuilder"""
        try:
            # Use the graph builder with timeout
            stats = await asyncio.wait_for(
                self.graph_builder.build_graph(stages=[0, 1]),
                timeout=30.0
            )
            logger.info(f"Graph building completed: {stats}")
        except asyncio.TimeoutError:
            logger.error("Graph building timed out")
            # Ensure at least stage 0 is built
            await self.graph_builder._build_stage_0()
        except Exception as e:
            logger.error(f"Graph building failed: {e}")
            # Ensure basic functionality
            await self.graph_builder._build_stage_0()

    async def fetch_samples(self, state: AgentState) -> AgentState:
        """Fetch sample data and optionally generate summaries for selected tables."""
        logger.info("Fetching sample data...")
        
        sample_data = {}
        for table_name, _ in state['selected_tables'][:config.MAX_TABLES_TO_SAMPLE]:
            # Check persistent cache first
            cached_sample = self.sample_cache.get(table_name)
            if cached_sample:
                sample_data[table_name] = cached_sample
            else:
                # Fetch fresh sample
                sample = self.connection.get_table_sample(table_name, config.SAMPLE_SIZE)
                if sample['rows']:
                    # Save to persistent cache
                    self.sample_cache.put(table_name, sample)
                    sample_data[table_name] = sample
            
            # Only generate summaries if enabled and missing
            if config.SUMMARIZATION_ENABLED:
                # Check table summaries
                if table_name not in self.table_summaries:
                    logger.info(f"Generating summary for table {table_name}")
                    table_info = self.metadata.get(table_name)
                    if table_info:
                        # Generate table summary
                        table_summary, _ = self.schema_summarizer.summarize_table(
                            table_name, table_info, sample_data.get(table_name, {})
                        )
                        
                        # Generate column summaries for key columns
                        if sample_data.get(table_name):
                            sample_values = sample_data[table_name]['rows']
                            for col in table_info.columns[:10]:  # Limit to first 10 columns
                                col_name = col['name']
                                col_key = f"{table_name}.{col_name}"
                                
                                if col_key not in self.column_summaries:
                                    # Get column stats
                                    stats = self.schema_analyzer.get_column_stats(table_name, col_name)
                                    
                                    # Extract sample values for this column
                                    col_samples = [row.get(col_name) for row in sample_values 
                                                if row.get(col_name) is not None]
                                    
                                    # Generate summary
                                    col_summary, _ = self.schema_summarizer.summarize_column(
                                        table_name, col, col_samples, stats, table_info.columns
                                    )
        
        state['sample_data'] = sample_data
        logger.info(f"Fetched samples for {len(sample_data)} tables")
        
        # Reload summaries if they were generated
        if config.SUMMARIZATION_ENABLED:
            self.table_summaries = self.persistence_manager.get_table_summaries(self.target_db)
            self.column_summaries = self.persistence_manager.get_column_summaries(self.target_db)
        
        return state
    
    def get_graph_data(self) -> Dict[str, Any]:
        """Get complete graph data for visualization"""
        try:
            # Ensure nodes are loaded - use sync version
            if not hasattr(self, 'graph_builder') or not self.graph_builder.nodes:
                logger.warning("Graph nodes not loaded, loading from persistence...")
                # Load nodes from persistence if available
                if self.persistence_manager and self.target_db:
                    existing_nodes = self.persistence_manager.get_graph_nodes(self.target_db)
                    if existing_nodes:
                        self.graph_builder.nodes = existing_nodes
                        logger.info(f"Loaded {len(existing_nodes)} nodes from persistence")
                    else:
                        # Create nodes synchronously
                        logger.info("No nodes in persistence, creating new ones...")
                        self.graph_builder.nodes = {}
                        for table_name, table_info in self.metadata.items():
                            node = self.graph_builder._create_node(table_name, table_info)
                            self.graph_builder.nodes[table_name] = node
                        logger.info(f"Created {len(self.graph_builder.nodes)} nodes")
            
            # Get nodes with their metadata
            nodes = []
            for table_name, node in self.graph_builder.nodes.items():
                try:
                    # Get table summary data
                    table_summary = self.table_summaries.get(table_name, {}).get('summary', {})
                    
                    node_data = {
                        "id": table_name,
                        "label": table_name.split('.')[-1],  # Show only table name without schema
                        "full_name": table_name,
                        "type": node.table_type or table_summary.get('table_type', 'unknown'),
                        "purpose": node.purpose or table_summary.get('purpose'),
                        "grain": node.grain or table_summary.get('grain'),
                        "row_count": node.row_count,
                        "has_time": node.has_time,
                        "measures": node.measures or table_summary.get('measures', []),
                        "dimensions": node.dimensions or table_summary.get('dimensions', []),
                        "key_columns": node.key_columns or table_summary.get('key_columns', []),
                        "foreign_keys": node.foreign_keys or table_summary.get('foreign_keys', []),
                        "time_columns": node.time_columns,
                        "metadata": {
                            "table_summary": table_summary,
                            "confidence": self.table_summaries.get(table_name, {}).get('confidence', 0)
                        }
                    }
                    nodes.append(node_data)
                except Exception as e:
                    logger.error(f"Error processing node {table_name}: {e}")
            
            # Get edges with their metadata
            edges = []
            edge_id = 0
            
            for edge_key, edge in self.relationship_graph.edges.items():
                try:
                    # Handle different edge key formats
                    if isinstance(edge_key, tuple) and len(edge_key) >= 2:
                        from_table = edge_key[0]
                        to_table = edge_key[1]
                    else:
                        # Skip if we can't determine tables
                        continue
                    
                    edge_data = {
                        "id": edge_id,
                        "source": from_table,
                        "target": to_table,
                        "weight": edge.weight,
                        "type": edge.edge_type,
                        "features": edge.features,
                        "label": self._get_edge_label(edge),
                        "explanation": self.relationship_graph.get_edge_explanation(from_table, to_table)
                    }
                    edges.append(edge_data)
                    edge_id += 1
                except Exception as e:
                    logger.error(f"Error processing edge {edge_key}: {e}")
            
            # Get graph statistics
            stats = self.relationship_graph.get_statistics()
            
            # Get join paths for common table pairs with error handling
            try:
                common_paths = self._get_common_join_paths()
            except Exception as e:
                logger.error(f"Error getting common paths: {e}", exc_info=True)
                common_paths = []
            
            return {
                "nodes": nodes,
                "edges": edges,
                "statistics": stats,
                "common_paths": common_paths,
                "metadata": {
                    "total_tables": len(self.metadata),
                    "total_summaries": len(self.table_summaries) + len(self.column_summaries),
                    "graph_built_at": datetime.now().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error in get_graph_data: {e}", exc_info=True)
            raise
    
    def get_table_relationships(self, table_name: str) -> Dict[str, Any]:
        """Get all relationships for a specific table"""
        try:
            if table_name not in self.metadata:
                return {
                    "error": f"Table {table_name} not found",
                    "available_tables": list(self.metadata.keys())
                }
            
            # Get node information
            node = None
            if hasattr(self, 'graph_builder') and self.graph_builder.nodes:
                node = self.graph_builder.nodes.get(table_name)
            
            node_info = {
                "table_name": table_name,
                "type": node.table_type if node and hasattr(node, 'table_type') else "unknown",
                "purpose": node.purpose if node and hasattr(node, 'purpose') else None,
                "row_count": node.row_count if node and hasattr(node, 'row_count') else 0,
                "key_columns": node.key_columns if node and hasattr(node, 'key_columns') else [],
                "foreign_keys": node.foreign_keys if node and hasattr(node, 'foreign_keys') else []
            }
            
            # Get all edges involving this table
            outgoing_edges = []
            incoming_edges = []
            
            for edge_key, edge in self.relationship_graph.edges.items():
                if isinstance(edge_key, tuple) and len(edge_key) >= 2:
                    from_table = edge_key[0]
                    to_table = edge_key[1]
                    
                    if from_table == table_name:
                        outgoing_edges.append({
                            "to_table": to_table,
                            "weight": edge.weight,
                            "type": edge.edge_type,
                            "features": edge.features,
                            "explanation": self.relationship_graph.get_edge_explanation(from_table, to_table),
                            "join_template": self.relationship_graph.get_join_template(from_table, to_table)
                        })
                    elif to_table == table_name:
                        incoming_edges.append({
                            "from_table": from_table,
                            "weight": edge.weight,
                            "type": edge.edge_type,
                            "features": edge.features,
                            "explanation": self.relationship_graph.get_edge_explanation(from_table, to_table),
                            "join_template": self.relationship_graph.get_join_template(from_table, to_table)
                        })
            
            # Get neighbors from graph with error handling
            try:
                neighbors = self.relationship_graph.get_neighbors(table_name)
            except Exception as e:
                logger.error(f"Error getting neighbors: {e}")
                neighbors = {}
            
            # Get shortest paths to other important tables
            important_tables = self._get_important_tables()
            paths = {}
            
            for target in important_tables:
                if target != table_name:
                    try:
                        path = self.relationship_graph.find_shortest_reliable_path(
                            table_name, target, min_weight=0.6, max_hops=3
                        )
                        if path:
                            path_info = []
                            total_weight = 0
                            
                            for step in path:
                                # Handle both 2-tuple and 3-tuple formats safely
                                if isinstance(step, tuple):
                                    if len(step) == 3:
                                        from_t, to_t, weight = step
                                    elif len(step) == 2:
                                        from_t, to_t = step
                                        weight = 0.5  # Default weight
                                    else:
                                        logger.warning(f"Unexpected path step format: {step}")
                                        continue
                                else:
                                    logger.warning(f"Path step is not a tuple: {step}")
                                    continue
                                
                                path_info.append((from_t, to_t, weight))
                                total_weight += weight
                            
                            if path_info:
                                paths[target] = {
                                    "path": path_info,
                                    "total_weight": total_weight,
                                    "hops": len(path_info)
                                }
                    except Exception as e:
                        logger.debug(f"Could not find path from {table_name} to {target}: {e}")
            
            return {
                "node_info": node_info,
                "outgoing_edges": sorted(outgoing_edges, key=lambda x: x['weight'], reverse=True),
                "incoming_edges": sorted(incoming_edges, key=lambda x: x['weight'], reverse=True),
                "neighbors": {
                    table: {
                        "edges": [
                            {
                                "type": e.edge_type,
                                "weight": e.weight,
                                "direction": "outgoing" if e.from_table == table_name else "incoming"
                            }
                            for e in edges
                        ]
                    }
                    for table, edges in neighbors.items()
                },
                "paths_to_important_tables": paths,
                "summary": {
                    "total_connections": len(outgoing_edges) + len(incoming_edges),
                    "strong_connections": sum(1 for e in outgoing_edges + incoming_edges if e['weight'] > 0.8),
                    "is_hub": len(outgoing_edges) + len(incoming_edges) > 5,
                    "is_isolated": len(outgoing_edges) + len(incoming_edges) == 0
                }
            }
        except Exception as e:
            logger.error(f"Error in get_table_relationships: {e}", exc_info=True)
            raise

    def _get_edge_label(self, edge: RelationshipEdge) -> str:
        """Generate a concise label for an edge"""
        if edge.edge_type == 'explicit_fk':
            return f"FK: {edge.features.get('from_column', '')} → {edge.features.get('to_column', '')}"
        elif edge.edge_type == 'fk_like_inclusion':
            ratio = edge.features.get('inclusion_ratio', 0)
            return f"Inclusion ({ratio:.0%})"
        elif edge.edge_type == 'column_similarity':
            return f"Similar: {edge.features.get('semantic_role', 'column')}"
        elif edge.edge_type == 'via_junction':
            return f"Via: {edge.features.get('junction_table', '').split('.')[-1]}"
        elif edge.edge_type == 'table_similarity':
            return f"Similar tables ({edge.features.get('similarity_score', 0):.2f})"
        else:
            return edge.edge_type.replace('_', ' ').title()
    
    def _get_common_join_paths(self) -> List[Dict[str, Any]]:
        """Get commonly used join paths"""
        try:
            # Find fact and dimension tables
            fact_tables = []
            dimension_tables = []
            
            # Check if graph_builder has nodes
            if hasattr(self, 'graph_builder') and self.graph_builder.nodes:
                for name, node in self.graph_builder.nodes.items():
                    if hasattr(node, 'table_type'):
                        if node.table_type == 'fact':
                            fact_tables.append(name)
                        elif node.table_type == 'dimension':
                            dimension_tables.append(name)
            
            logger.info(f"Found {len(fact_tables)} fact tables and {len(dimension_tables)} dimension tables")
            
            paths = []
            
            # Find paths between facts and dimensions
            for fact in fact_tables[:5]:  # Limit to prevent too many
                for dim in dimension_tables[:10]:
                    try:
                        path = self.relationship_graph.find_shortest_reliable_path(
                            fact, dim, min_weight=0.6, max_hops=2
                        )
                        
                        if path:
                            logger.debug(f"Path from {fact} to {dim}: {path}")
                            path_steps = []
                            
                            for i, step in enumerate(path):
                                logger.debug(f"Processing step {i}: {step}, type: {type(step)}")
                                
                                # Handle different formats
                                if isinstance(step, tuple):
                                    if len(step) == 3:
                                        from_t, to_t, weight = step
                                    elif len(step) == 2:
                                        from_t, to_t = step
                                        weight = 0.5  # Default weight
                                    else:
                                        logger.warning(f"Unexpected tuple length {len(step)} in step: {step}")
                                        continue
                                else:
                                    logger.warning(f"Step is not a tuple: {step}, type: {type(step)}")
                                    continue
                                
                                path_steps.append({
                                    "from": from_t,
                                    "to": to_t,
                                    "weight": weight
                                })
                            
                            if path_steps:
                                paths.append({
                                    "from": fact,
                                    "to": dim,
                                    "path": path_steps,
                                    "total_weight": sum(s['weight'] for s in path_steps),
                                    "length": len(path_steps)
                                })
                    except Exception as e:
                        logger.error(f"Error finding path from {fact} to {dim}: {e}", exc_info=True)
            
            # Sort by total weight
            paths.sort(key=lambda x: x['total_weight'], reverse=True)
            return paths[:20]  # Return top 20 paths
            
        except Exception as e:
            logger.error(f"Error in _get_common_join_paths: {e}", exc_info=True)
            return []
        
    def _get_important_tables(self) -> List[str]:
        """Get list of important tables based on various criteria"""
        important = []
        # Ensure nodes are available
        if hasattr(self, 'graph_builder') and self.graph_builder.nodes:
            # Add fact tables
            for name, node in self.graph_builder.nodes.items():
                if node.table_type == 'fact':
                    important.append(name)
                # Add tables with many measures
                if len(node.measures) >= 3:
                    important.append(name)
        # Add high-degree tables (hubs)
        degree_threshold = 5
        for table in self.metadata:
            neighbors = self.relationship_graph.get_neighbors(table)
            degree = sum(len(edges) for edges in neighbors.values())
            if degree >= degree_threshold:
                important.append(table)
        # Remove duplicates and limit
        return list(set(important))[:10]
    
    async def check_for_schema_changes(self) -> Dict[str, Any]:
        """Check for new tables or columns and update summaries if needed"""
        logger.info("Checking for schema changes...")
        
        # Check if schema updates are enabled
        if not config.AUTO_SCHEMA_UPDATE:
            logger.info("Schema updates are disabled in configuration")
            return {
                'new_tables': [],
                'new_columns': [],
                'removed_tables': [],
                'removed_columns': [],
                'skipped': True,
                'reason': 'AUTO_SCHEMA_UPDATE is disabled'
            }
        
        changes = {
            'new_tables': [],
            'new_columns': [],
            'removed_tables': [],
            'removed_columns': []
        }
        
        try:
            # First, check if we should skip based on recent extraction
            agent_state = self.persistence_manager.get_agent_state(self.target_db)
            last_initialized = agent_state.get('last_initialized')
            
            if last_initialized:
                last_time = datetime.fromisoformat(last_initialized)
                age_minutes = (datetime.now() - last_time).total_seconds() / 60
                
                if age_minutes < 5:  # Skip if extracted within last 5 minutes
                    logger.info(f"Metadata was extracted {age_minutes:.1f} minutes ago, skipping re-extraction")
                    return changes
            
            # Perform quick check first to see if full extraction is needed
            with self.connection.get_connection() as conn:
                inspector = sa.inspect(conn)
                
                # Get current table names without full metadata extraction
                current_table_names = set()
                current_table_count = 0
                
                # Check all schemas
                for schema in inspector.get_schema_names():
                    if not self.metadata_extractor._is_system_schema(schema):
                        try:
                            schema_tables = inspector.get_table_names(schema=schema)
                            for table in schema_tables:
                                current_table_names.add(f"{schema}.{table}")
                                current_table_count += 1
                        except Exception as e:
                            logger.debug(f"Cannot access schema '{schema}': {e}")
                
                # Check default schema
                try:
                    default_tables = inspector.get_table_names()
                    for table in default_tables:
                        # Only add if not already present with schema prefix
                        if not any(name.endswith(f".{table}") for name in current_table_names):
                            current_table_names.add(table)
                            current_table_count += 1
                except Exception as e:
                    logger.debug(f"Cannot access default schema: {e}")
            
            # Compare with stored metadata
            stored_table_names = set(self.metadata.keys())
            stored_table_count = len(stored_table_names)
            
            logger.info(f"Quick check - Current: {current_table_count} tables, Stored: {stored_table_count} tables")
            
            # If table names exactly match, check for column changes on a few tables
            if current_table_names == stored_table_names:
                logger.info("Table names match, checking for column changes...")
                
                # Sample a few tables to check for column changes
                tables_to_check = list(stored_table_names)[:5]  # Check first 5 tables
                column_changes_found = False
                
                with self.connection.get_connection() as conn:
                    inspector = sa.inspect(conn)
                    
                    for table_name in tables_to_check:
                        # Parse schema and table name
                        if '.' in table_name:
                            schema, table_only = table_name.split('.', 1)
                        else:
                            schema = None
                            table_only = table_name
                        
                        try:
                            # Get current columns
                            current_columns = inspector.get_columns(table_only, schema=schema)
                            current_col_names = {col['name'].lower() for col in current_columns}
                            
                            # Get stored columns
                            stored_cols = {col['name'].lower() for col in self.metadata[table_name].columns}
                            
                            # Check for differences
                            if current_col_names != stored_cols:
                                column_changes_found = True
                                break
                        except Exception as e:
                            logger.debug(f"Error checking columns for {table_name}: {e}")
                
                if not column_changes_found:
                    logger.info("No schema changes detected (quick check passed)")
                    return changes
            
            # If we get here, we need full extraction
            logger.info("Schema changes detected or structure mismatch, performing full extraction...")
            
            # Get current metadata from database
            current_metadata = self.metadata_extractor.extract_all_metadata()
            
            # Update extraction timestamp
            self.persistence_manager.update_agent_state(
                self.target_db,
                last_initialized=datetime.now().isoformat()
            )
            
            # Normalize table names for comparison
            def normalize_table_name(name):
                return name.lower().strip()
            
            # Create normalized mappings
            stored_tables_normalized = {normalize_table_name(k): k for k in self.metadata.keys()}
            current_tables_normalized = {normalize_table_name(k): k for k in current_metadata.keys()}
            
            # Compare normalized names
            stored_keys = set(stored_tables_normalized.keys())
            current_keys = set(current_tables_normalized.keys())
            
            # Find new tables
            new_table_keys = current_keys - stored_keys
            if new_table_keys:
                logger.info(f"Found {len(new_table_keys)} new tables")
                for norm_key in new_table_keys:
                    actual_table_name = current_tables_normalized[norm_key]
                    changes['new_tables'].append(actual_table_name)
                    
                    # Add new table to metadata
                    table_info = current_metadata[actual_table_name]
                    self.metadata[actual_table_name] = table_info
                    
                    # Save to persistence
                    self.persistence_manager.save_table_metadata(
                        self.target_db, actual_table_name, table_info
                    )
            
            # Find removed tables
            removed_table_keys = stored_keys - current_keys
            if removed_table_keys:
                logger.info(f"Found {len(removed_table_keys)} removed tables")
                for norm_key in removed_table_keys:
                    actual_table_name = stored_tables_normalized[norm_key]
                    changes['removed_tables'].append(actual_table_name)
                    
                    # Remove from metadata
                    if actual_table_name in self.metadata:
                        del self.metadata[actual_table_name]
            
            # Check for column changes in existing tables
            common_keys = stored_keys & current_keys
            for norm_key in common_keys:
                stored_table_name = stored_tables_normalized[norm_key]
                current_table_name = current_tables_normalized[norm_key]
                
                # Get columns
                stored_cols = {col['name'].lower() for col in self.metadata[stored_table_name].columns}
                current_cols = {col['name'].lower() for col in current_metadata[current_table_name].columns}
                
                # New columns
                new_cols = current_cols - stored_cols
                if new_cols:
                    for col_name in new_cols:
                        # Find actual column name (with proper case)
                        for col in current_metadata[current_table_name].columns:
                            if col['name'].lower() == col_name:
                                changes['new_columns'].append(f"{current_table_name}.{col['name']}")
                                break
                    
                    # Update table metadata
                    self.metadata[stored_table_name] = current_metadata[current_table_name]
                    self.persistence_manager.save_table_metadata(
                        self.target_db, stored_table_name, current_metadata[current_table_name]
                    )
                
                # Removed columns
                removed_cols = stored_cols - current_cols
                if removed_cols:
                    for col_name in removed_cols:
                        # Find actual column name with proper case
                        for col in self.metadata[stored_table_name].columns:
                            if col['name'].lower() == col_name:
                                changes['removed_columns'].append(f"{stored_table_name}.{col['name']}")
                                break
            
            # Log summary
            logger.info(f"Schema comparison complete:")
            logger.info(f"  Stored tables: {len(self.metadata)}")
            logger.info(f"  Current tables: {len(current_metadata)}")
            logger.info(f"  New tables: {len(changes['new_tables'])}")
            logger.info(f"  Removed tables: {len(changes['removed_tables'])}")
            logger.info(f"  New columns: {len(changes['new_columns'])}")
            logger.info(f"  Removed columns: {len(changes['removed_columns'])}")
            
            # Generate summaries for new tables and columns
            if config.SUMMARIZATION_ENABLED and (changes['new_tables'] or changes['new_columns']):
                logger.info("Generating summaries for new schema elements...")
                
                # Prepare lists for summary generation
                tables_needing_summary = [
                    (table_name, current_metadata[table_name])
                    for table_name in changes['new_tables']
                    if table_name in current_metadata
                ]
                
                columns_needing_summary = []
                for col_key in changes['new_columns']:
                    table_name, col_name = col_key.split('.', 1)
                    if table_name in current_metadata:
                        table_info = current_metadata[table_name]
                        for col in table_info.columns:
                            if col['name'] == col_name:
                                columns_needing_summary.append((table_name, col, table_info))
                                break
                
                # Generate summaries
                if columns_needing_summary:
                    await self._generate_column_summaries(columns_needing_summary, max_concurrent=5)
                
                if tables_needing_summary:
                    await self._generate_table_summaries(tables_needing_summary, max_concurrent=5)
                
                # Reload summaries
                self.table_summaries = self.persistence_manager.get_table_summaries(self.target_db)
                self.column_summaries = self.persistence_manager.get_column_summaries(self.target_db)
                
                # Update components with new summaries
                self.table_selector_llm.table_summaries = self.table_summaries
                self.table_selector_llm.column_summaries = self.column_summaries
                self.sql_generator.table_summaries = self.table_summaries
                self.sql_generator.column_summaries = self.column_summaries
            


                # Update foreign keys if there are new tables
                if changes['new_tables']:
                    logger.info("Updating relationship graph with new foreign keys...")
                    foreign_keys = self.metadata_extractor.extract_foreign_keys()
                    
                    # Add only new foreign keys
                    for from_table, to_table in foreign_keys:
                        if from_table in changes['new_tables'] or to_table in changes['new_tables']:
                            self.relationship_graph.add_edge(
                                from_table, to_table, weight=1.0, edge_type='explicit_fk'
                            )
                
                logger.info(f"Schema check complete. Changes: {len(changes['new_tables'])} new tables, "
                        f"{len(changes['new_columns'])} new columns")
            
            if not config.SUMMARIZATION_ENABLED:
                logger.info("Summarization is disabled, skipping summary generation for new schema elements")
            
            return changes
            
        except Exception as e:
            logger.error(f"Error checking for schema changes: {e}", exc_info=True)
            return changes
    

    def _get_tables_with_samples(self, all_tables: Dict[str, Any]) -> Dict[str, Any]:
        """Filter tables to only include those with available samples"""
        tables_with_samples = {}
        
        for table_name, table_info in all_tables.items():
            # Check if sample exists in cache
            if self.sample_cache.get(table_name) is not None:
                tables_with_samples[table_name] = table_info
        
        return tables_with_samples
    
    async def ensure_critical_samples_available(self, min_tables: int = 10):
        """Ensure samples are available for most important tables"""
        logger.info(f"Checking sample availability for critical tables...")
        
        # Get tables without samples
        tables_without_samples = []
        for table_name in self.metadata:
            if self.sample_cache.get(table_name) is None:
                tables_without_samples.append(table_name)
        
        if not tables_without_samples:
            logger.info("All tables already have samples")
            return
        
        logger.info(f"Found {len(tables_without_samples)} tables without samples")
        
        # Prioritize tables to sample
        # Use graph importance and table type to prioritize
        priority_tables = []
        
        # First, add fact and dimension tables
        if hasattr(self, 'graph_builder') and self.graph_builder.nodes:
            for table_name, node in self.graph_builder.nodes.items():
                if table_name in tables_without_samples:
                    if node.table_type in ['fact', 'dimension']:
                        priority_tables.append((table_name, 1.0))
                    elif node.table_type == 'junction':
                        priority_tables.append((table_name, 0.8))
                    else:
                        priority_tables.append((table_name, 0.5))
        
        # Add remaining tables with lower priority
        for table_name in tables_without_samples:
            if not any(t[0] == table_name for t in priority_tables):
                priority_tables.append((table_name, 0.3))
        
        # Sort by priority
        priority_tables.sort(key=lambda x: x[1], reverse=True)
        
        # Fetch samples for top N tables
        tables_to_sample = [t[0] for t in priority_tables[:min_tables]]
        logger.info(f"Pre-loading samples for {len(tables_to_sample)} priority tables")
        
        for table_name in tables_to_sample:
            try:
                sample = await asyncio.to_thread(
                    self.connection.get_table_sample,
                    table_name,
                    config.SAMPLE_SIZE
                )
                if sample['rows']:
                    self.sample_cache.put(table_name, sample)
                    logger.debug(f"Pre-loaded sample for {table_name}")
            except Exception as e:
                logger.error(f"Failed to pre-load sample for {table_name}: {e}")

    async def fetch_table_head(self, table_name: str, sample_size: int = 1000, 
                            force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Universal function to fetch table head with caching"""
        if not force_refresh:
            cached_sample = self.sample_cache.get(table_name)
            if cached_sample:
                logger.debug(f"Using cached head for {table_name}")
                return cached_sample
        
        logger.info(f"Fetching fresh head for {table_name} (size: {sample_size})")
        try:
            sample = await asyncio.to_thread(
                self.connection.get_table_sample, 
                table_name, 
                sample_size
            )
            
            if sample and sample.get('rows'):
                self.sample_cache.put(table_name, sample)
                logger.info(f"Fetched and cached {len(sample['rows'])} rows for {table_name}")
                return sample
            else:
                logger.warning(f"No data retrieved for {table_name}")
                return None
        
        except Exception as e:
            logger.error(f"Failed to fetch head for {table_name}: {e}")
            return None
        
    async def prefetch_all_table_heads(self, sample_size: int = 1000, 
                                    max_concurrent: int = 5) -> Dict[str, Any]:
        """Fetch heads for all tables concurrently"""
        logger.info(f"Starting to fetch heads for {len(self.metadata)} tables...")
        
        results = {
            'success': [],
            'failed': [],
            'total': len(self.metadata)
        }
        
        table_names = list(self.metadata.keys())
        
        for i in range(0, len(table_names), max_concurrent):
            batch = table_names[i:i + max_concurrent]
            
            tasks = []
            for table_name in batch:
                task = self.fetch_table_head(table_name, sample_size, force_refresh=True)
                tasks.append((table_name, task))
            
            for table_name, task in tasks:
                try:
                    result = await task
                    if result:
                        results['success'].append(table_name)
                    else:
                        results['failed'].append(table_name)
                except Exception as e:
                    logger.error(f"Failed to fetch head for {table_name}: {e}")
                    results['failed'].append(table_name)
            
            processed = len(results['success']) + len(results['failed'])
            logger.info(f"Progress: {processed}/{results['total']} tables processed")
        
        logger.info(f"Completed fetching heads: {len(results['success'])} success, "
                    f"{len(results['failed'])} failed")
        
        return results
    

    def _assess_table_quality_from_sample(self, table_name: str, sample_data: Dict[str, Any], 
                                        table_type: Optional[str] = None) -> Dict[str, Any]:
        """Assess table quality from cached sample data"""
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
        
        # Calculate fill rates and identify empty columns
        column_fill_rates = {}
        empty_columns = []
        low_diversity_columns = []
        
        for col in columns:
            non_null_count = sum(1 for row in rows if row.get(col) is not None and str(row.get(col)).strip())
            fill_rate = non_null_count / actual_row_count if actual_row_count > 0 else 0
            column_fill_rates[col] = fill_rate
            
            if fill_rate == 0:
                empty_columns.append(col)
            
            # Check diversity
            unique_values = set(row.get(col) for row in rows if row.get(col) is not None)
            if len(unique_values) < 3 and fill_rate > 0:
                low_diversity_columns.append(col)
        
        # Overall fill rate
        overall_fill_rate = sum(column_fill_rates.values()) / len(columns) if columns else 0
        
        # Check if has meaningful data
        has_meaningful_data = any(rate > 0.5 for rate in column_fill_rates.values())
        
        # Calculate quality score based on table type
        quality_score = self._calculate_type_aware_quality_score(
            table_type=table_type,
            row_count=actual_row_count,
            row_category=row_category,
            fill_rate=overall_fill_rate,
            empty_column_ratio=len(empty_columns) / len(columns) if columns else 1,
            has_meaningful_data=has_meaningful_data,
            diversity_ratio=1 - (len(low_diversity_columns) / len(columns)) if columns else 0
        )
        
        return {
            'quality_score': quality_score,
            'row_count_estimate': row_count_estimate,
            'is_exact_count': is_exact_count,
            'row_category': row_category,
            'fill_rate': overall_fill_rate,
            'column_fill_rates': column_fill_rates,
            'empty_columns': empty_columns,
            'low_diversity_columns': low_diversity_columns,
            'has_meaningful_data': has_meaningful_data,
            'quality_category': self._categorize_quality(quality_score)
        }

    def _calculate_type_aware_quality_score(self, table_type: Optional[str], row_count: int,
                                        row_category: str, fill_rate: float,
                                        empty_column_ratio: float, has_meaningful_data: bool,
                                        diversity_ratio: float) -> float:
        """Calculate quality score based on table type"""
        if not table_type:
            table_type = 'unknown'
        
        # Type-specific scoring
        if table_type in ['catalog', 'reference', 'configuration']:
            # Catalog tables: row count doesn't matter, fill rate is critical
            quality_score = (
                0.5 * fill_rate +
                0.3 * (1.0 if has_meaningful_data else 0.0) +
                0.2 * diversity_ratio
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
                0.2 * diversity_ratio +
                0.2 * (1.0 if has_meaningful_data else 0.0)
            )
        elif table_type == 'junction':
            # Junction tables: fill rate matters most
            quality_score = (
                0.6 * fill_rate +
                0.2 * (1.0 if has_meaningful_data else 0.0) +
                0.2 * (1.0 if row_count > 0 else 0.0)
            )
        else:
            # Unknown type: balanced scoring
            row_score = min(1.0, row_count / 100) if row_count < 1000 else 1.0
            quality_score = (
                0.25 * row_score +
                0.25 * fill_rate +
                0.25 * (1.0 if has_meaningful_data else 0.0) +
                0.25 * diversity_ratio
            )
        
        return quality_score

    def _categorize_quality(self, score: float) -> str:
        """Categorize quality score"""
        if score >= 0.8:
            return 'high'
        elif score >= 0.5:
            return 'medium'
        elif score >= 0.3:
            return 'low'
        else:
            return 'very_low'

    def _get_table_type_from_summary(self, table_name: str) -> Optional[str]:
        """Get table type from summary if available"""
        if table_name in self.table_summaries:
            return self.table_summaries[table_name]['summary'].get('table_type', 'unknown')
        return None