from dataclasses import dataclass

@dataclass
class Config:
    # Database
    DATABASE_URI: str = ""
    
    # Schema Management
    AUTO_SCHEMA_UPDATE: bool = False  # Master switch for automatic schema operations
    SCHEMA_UPDATE_ON_STARTUP: bool = False  # Check schema changes on startup
    SCHEMA_UPDATE_INTERVAL_HOURS: int = 24  # How often to check for updates
    SUMMARIZATION_ENABLED: bool = False  # Enable/disable summary generation
    SUMMARIZATION_ON_STARTUP: bool = False  # Generate summaries on startup
    SUMMARIZATION_MAX_AGE_DAYS: int = 30  # Max age before regenerating summaries

    # Adaptive thresholds
    INITIAL_MAX_TABLES: int = 10
    MIN_TABLE_RELEVANCE: float = 0.3
    SEMANTIC_SIMILARITY_THRESHOLD: float = 0.5
    RELATIONSHIP_UPDATE_THRESHOLD: float = 0.8
    METADATA_UPDATE_THRESHOLD: float = 0.9
    DIVERSITY_THRESHOLD: float = 0.6
    
    # Sample data parameters
    SAMPLE_SIZE: int = 3
    SAMPLE_CACHE_TTL_HOURS: int = 24
    SAMPLE_CACHE_MAX_SIZE_MB: int = 100
    SAMPLE_FETCH_TIMEOUT: int = 2
    MAX_TABLES_TO_SAMPLE: int = 10
    
    # Learning parameters
    SIMPLE_QUERY_SUCCESS_REQUIREMENT: int = 2
    COMPLEX_QUERY_SUCCESS_REQUIREMENT: int = 3
    CONFIDENCE_DECAY_RATE: float = 0.95
    DECAY_PERIOD_DAYS: int = 30
    
    # Performance parameters
    QUERY_TIMEOUT_SIMPLE: int = 15
    QUERY_TIMEOUT_COMPLEX: int = 30
    MAX_RETRIES: int = 3
    CACHE_TTL_HOURS: int = 24
    
    # LLM settings
    LLM_MODEL: str = "anthropic.claude-sonnet-4-20250514-v1:0"
    LLM_TEMPERATURE: float = 0.1
    
    # Embedding settings
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSION: int = 3072

        # Sample requirements
    REQUIRE_SAMPLES_FOR_SELECTION: bool = True
    PRELOAD_SAMPLES_ON_STARTUP: bool = False
    PRELOAD_SAMPLE_COUNT: int = 20

config = Config()