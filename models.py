from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

class QueryIntent(Enum):
    AGGREGATION = "aggregation"
    COMPARISON = "comparison"
    TREND = "trend"
    DISTRIBUTION = "distribution"
    LOOKUP = "lookup"

class QueryComplexity(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"

@dataclass
class Intent:
    intent_type: QueryIntent
    entities: List[str]
    metrics: List[str]
    filters: Dict[str, Any]
    time_range: Optional[Dict[str, datetime]]
    expected_result_type: str
    expects_single_value: bool
    expects_list: bool
    expects_aggregation: bool
    intent_confidence: float
    query_complexity: QueryComplexity

@dataclass
class TableInfo:
    name: str
    schema: str
    columns: List[Dict[str, Any]]
    ddl: str
    sample_data: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    last_analyzed: Optional[datetime] = None

@dataclass
class QueryResult:
    sql: str
    data: Any
    execution_time: float
    error: Optional[str] = None
    retry_count: int = 0
    validation_score: float = 0.0
    validation_checks: Dict[str, bool] = field(default_factory=dict)
    used_sample_data: bool = False

@dataclass
class QueryMemory:
    id: str
    user_prompt: str
    prompt_embedding: List[float]
    intent_summary: Dict[str, Any]
    sql_query: str
    tables_used: List[str]
    result_summary: str
    success_score: float
    timestamp: datetime
    execution_time: float
    retry_count: int
    validation_checks: Dict[str, bool]
    query_complexity: QueryComplexity

@dataclass
class RelationshipEdge:
    from_table: str
    to_table: str
    weight: float
    edge_type: str  # 'explicit_fk', 'fk_like_inclusion', 'key_equivalent', 'joinable_via', 'topic_similarity'
    relation_type: str  # Same as edge_type for compatibility
    features: Dict[str, Any] = field(default_factory=dict)  # {inclusion_ratio, jaccard, type_compat, examples}
    usage_count: int = 0
    usage_contexts: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        # Ensure relation_type matches edge_type
        self.relation_type = self.edge_type

@dataclass
class GraphNode:
    """Enhanced node representation for relationship graph"""
    table_name: str
    purpose: Optional[str] = None
    table_type: Optional[str] = None  # fact|dimension|junction|transaction|reference
    grain: Optional[str] = None
    key_columns: List[str] = field(default_factory=list)
    foreign_keys: List[str] = field(default_factory=list)
    measures: List[str] = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    time_columns: List[str] = field(default_factory=list)
    row_count: Optional[int] = None
    has_time: bool = False
    default_joins: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
@dataclass
class JoinPath:
    """Represents a join path between tables"""
    tables: List[str]
    edges: List[RelationshipEdge]
    total_weight: float
    join_clauses: List[str]

@dataclass
class InsightHistory:
    table_name: str
    column_name: Optional[str]
    insight: str
    confidence_score: float
    timestamp: datetime
    query_context: str
    success_flag: bool