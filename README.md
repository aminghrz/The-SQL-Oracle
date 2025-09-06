# SQL Oracle Agent

A sophisticated text-to-SQL agent built with LangGraph that intelligently understands database schemas and generates accurate SQL queries from natural language prompts.

## 🚀 Key Capabilities

### Intelligent Schema Understanding
- **Automated Schema Analysis**: Extracts and analyzes database metadata across multiple schemas
- **Semantic Schema Summarization**: Uses LLM to understand table purposes, column meanings, and relationships
- **Dynamic Schema Evolution**: Automatically detects and adapts to schema changes
- **Multi-dialect Support**: Works with Oracle, PostgreSQL, MySQL, SQLite, and SQL Server

### Advanced Query Generation
- **Context-Aware SQL Generation**: Leverages table summaries and sample data for accurate queries
- **Relationship Discovery**: Automatically infers table relationships through multiple methods
- **Retry Logic with Learning**: Intelligently retries failed queries with alternative approaches
- **Dialect-Specific Optimization**: Generates database-specific SQL syntax and patterns

### Semantic Search & Selection
- **Embedding-Based Table Selection**: Uses vector embeddings to find relevant tables
- **Column-Level Semantic Matching**: Matches user intent to specific columns and their meanings
- **Quality-Aware Filtering**: Prioritizes tables based on data quality and completeness
- **Intent-Driven Selection**: Adapts table selection based on query type (aggregation, lookup, trend, etc.)

### Persistent Learning System
- **Query Memory**: Learns from successful and failed queries
- **Relationship Graph**: Builds and maintains a comprehensive table relationship graph
- **Performance Optimization**: Caches samples and summaries for faster response times
- **Continuous Improvement**: Adapts selection strategies based on historical performance

### Enterprise Features
- **Multi-Language Support**: Handles queries in multiple languages including Persian/Farsi
- **Data Visualization**: Automatically generates appropriate charts and visualizations
- **Comprehensive Validation**: Multi-layer result validation with confidence scoring
- **RESTful API**: Production-ready FastAPI interface with health checks and monitoring

## 🏗️ Architecture

The system employs a sophisticated multi-stage approach:

1. **Intent Analysis**: Understands user query intent and expected result types
2. **Semantic Table Selection**: Uses embeddings and summaries to find relevant tables
3. **Context-Aware SQL Generation**: Creates optimized queries with proper joins and filters
4. **Intelligent Validation**: Validates results and retries with alternative strategies
5. **Learning Integration**: Continuously improves through query success/failure analysis

## 🔧 Technical Stack

- **LangGraph**: Orchestrates the multi-step reasoning workflow
- **OpenAI GPT-4**: Powers natural language understanding and SQL generation
- **SQLAlchemy**: Provides database abstraction and metadata extraction
- **Vector Embeddings**: Enables semantic similarity search for tables and columns
- **FastAPI**: Delivers high-performance REST API
- **SQLite**: Manages persistent storage for learning and caching

## 🎯 Use Cases

- **Business Intelligence**: Convert natural language questions into complex analytical queries
- **Data Exploration**: Help users discover and query unfamiliar databases
- **Report Automation**: Generate recurring reports from natural language specifications
- **Cross-Database Analytics**: Query multiple database systems with consistent interface
- **Self-Service Analytics**: Enable non-technical users to access data independently

## 🔒 Enterprise Ready

- Persistent learning across sessions
- Comprehensive error handling and recovery
- Performance monitoring and statistics
- Configurable quality thresholds
- Production-grade logging and debugging
- Scalable caching mechanisms

---
