import asyncio
import logging
from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from sql_agent import SQLAgent
from config import config
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
# At the top of main.py
import sys

# Set UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Configure logging with UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger('table_selector').setLevel(logging.DEBUG)

import json
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    global agent
    database_uri = os.getenv("DATABASE_URI", config.DATABASE_URI)
    if not database_uri:
        raise ValueError("DATABASE_URI not configured")
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    openai_base_url = os.environ.get("OPENAI_BASE_URL")
    if not openai_base_url:
        raise ValueError("OPENAI_BASE_URL not set")
    
    logger.info("Initializing SQL Agent...")
    logger.info(f"Schema update settings - Auto: {config.AUTO_SCHEMA_UPDATE}, "
                f"On startup: {config.SCHEMA_UPDATE_ON_STARTUP}")
    logger.info(f"Summarization settings - Enabled: {config.SUMMARIZATION_ENABLED}, "
                f"On startup: {config.SUMMARIZATION_ON_STARTUP}")
    
    agent = SQLAgent(database_uri)
    logger.info("SQL Agent initialized successfully")

    if config.PRELOAD_SAMPLES_ON_STARTUP:
        logger.info("Pre-loading samples for critical tables...")
        await agent.ensure_critical_samples_available(min_tables=config.PRELOAD_SAMPLE_COUNT)
        logger.info("Sample pre-loading completed")
    
    # Check if this is a fresh initialization or existing one
    agent_state = agent.persistence_manager.get_agent_state(agent.target_db)
    
    # Only check for schema changes if enabled and we have metadata extracted
    if config.AUTO_SCHEMA_UPDATE and config.SCHEMA_UPDATE_ON_STARTUP:
        if agent_state.get('metadata_extracted', False):
            logger.info("Checking for schema changes...")
            changes = await agent.check_for_schema_changes()
            
            if changes['new_tables']:
                logger.info(f"New tables detected: {', '.join(changes['new_tables'][:5])}...")
            if changes['new_columns']:
                logger.info(f"New columns detected: {', '.join(changes['new_columns'][:5])}...")
        else:
            logger.info("First time initialization - skipping schema change check")
    else:
        logger.info("Schema update checking is disabled")
    
    # Ensure summaries are up to date if enabled
    if config.SUMMARIZATION_ENABLED and config.SUMMARIZATION_ON_STARTUP:
        await agent.ensure_summaries_updated(max_age_days=config.SUMMARIZATION_MAX_AGE_DAYS)
        logger.info("Schema summaries checked and updated")
    else:
        logger.info("Summarization is disabled or skipped on startup")
    
    # Start periodic tasks only if enabled
    if config.AUTO_SCHEMA_UPDATE:
        asyncio.create_task(periodic_schema_check(agent, interval_hours=config.SCHEMA_UPDATE_INTERVAL_HOURS))
        asyncio.create_task(periodic_summary_update(agent, interval_hours=24))
    else:
        logger.info("Periodic schema updates are disabled")
    
    # Always update relationships (lightweight operation)
    await agent.ensure_relationships_updated(max_age_days=30)
    logger.info("Relationship graph checked and updated")
    
    yield  # The application runs here
    
    # Code to run on shutdown
    logger.info("Shutting down...")

async def periodic_schema_check(agent: SQLAgent, interval_hours: int = 6):
    """Periodically check for schema changes."""
    if not config.AUTO_SCHEMA_UPDATE:
        logger.info("Periodic schema check disabled")
        return
        
    while True:
        try:
            await asyncio.sleep(interval_hours * 3600)  # Wait for interval
            
            if config.AUTO_SCHEMA_UPDATE:  # Double-check in case config changed
                logger.info("Running periodic schema change check...")
                changes = await agent.check_for_schema_changes()
                
                if changes['new_tables'] or changes['new_columns']:
                    logger.info(f"Schema changes detected: {len(changes['new_tables'])} new tables, "
                               f"{len(changes['new_columns'])} new columns")
            else:
                logger.info("Schema updates disabled, skipping periodic check")
                
        except Exception as e:
            logger.error(f"Periodic schema check failed: {e}")

async def periodic_summary_update(agent: SQLAgent, interval_hours: int = 24):
    """Periodically check and update summaries."""
    if not config.SUMMARIZATION_ENABLED:
        logger.info("Periodic summary update disabled")
        return
        
    while True:
        try:
            await asyncio.sleep(interval_hours * 3600)  # Wait for interval
            
            if config.SUMMARIZATION_ENABLED:  # Double-check in case config changed
                logger.info("Running periodic summary update...")
                await agent.ensure_summaries_updated(max_age_days=config.SUMMARIZATION_MAX_AGE_DAYS)
            else:
                logger.info("Summarization disabled, skipping periodic update")
                
        except Exception as e:
            logger.error(f"Periodic summary update failed: {e}")


# Initialize FastAPI app
app = FastAPI(title="SQL Chatbot Agent", version="1.0.0", lifespan=lifespan)

# Global agent instance
agent = None

class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    success: bool
    data: list = []
    columns: list = []
    row_count: int = 0
    sql_query: str = ""
    visualization: Optional[Dict[str, Any]] = None
    explanation: str = ""
    confidence: float = 0.0
    execution_time: float = 0.0
    tables_used: list = []
    warning: Optional[str] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SQL Chatbot Agent API",
        "status": "running",
        "endpoints": {
            "query": "/query",
            "statistics": "/statistics",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    return {
        "status": "healthy",
        "agent_initialized": agent is not None
    }

class CustomJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            cls=DateTimeEncoder
        ).encode("utf-8")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a natural language query"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        user_prompt = request.prompt

        logger.info(f"Processing query: {request.prompt[:50]}...")
        
        # Add overall timeout for the endpoint
        import asyncio
        result = await asyncio.wait_for(
            agent.process_query(user_prompt),
            timeout=90.0  # 90 seconds timeout
        )
        
        # Handle datetime serialization in data
        if result.get('data'):
            # Convert datetime objects to strings
            for row in result['data']:
                for key, value in row.items():
                    if isinstance(value, datetime):
                        row[key] = value.isoformat()
        
        # Convert to response model
        response = QueryResponse(**result)
        
        logger.info(f"Query processed successfully. Confidence: {response.confidence:.2f}")
        return response
        
    except asyncio.TimeoutError:
        logger.error("Query timeout at endpoint level")
        return QueryResponse(
            success=False,
            error="Query processing timed out. The query might be too complex.",
            confidence=0.0
        )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Query processing failed: {error_trace}")
        
        return QueryResponse(
            success=False,
            error=str(e),
            confidence=0.0
        )

@app.get("/statistics")
async def get_statistics():
    """Get agent statistics"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        stats = agent.get_statistics()
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/graph")
async def get_graph():
    """Get the complete relationship graph"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Get graph data
        graph_data = agent.get_graph_data()
        
        return {
            "success": True,
            "graph": graph_data
        }
    except Exception as e:
        logger.error(f"Failed to get graph data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph/table/{table_name}")
async def get_table_relationships(table_name: str):
    """Get relationships for a specific table"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Get relationships for the table
        relationships = agent.get_table_relationships(table_name)
        
        return {
            "success": True,
            "table": table_name,
            "relationships": relationships
        }
    except Exception as e:
        logger.error(f"Failed to get relationships for {table_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(query_id: str, feedback: str):
    """Submit feedback for a query"""
    # This endpoint can be implemented to handle user feedback
    # and improve the learning pipeline
    return {
        "success": True,
        "message": "Feedback received"
    }
def run_server():
    """Run the FastAPI server"""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=9995,
        reload=True,
        reload_excludes=[
            "*.sqlite3",
            "*.sqlite3-journal",
            "*.sqlite3-wal", 
            "*.db",
            "*.log",
            "*.pyc",
            "__pycache__/*",
            ".git/*",
            "*.tmp",
            "*.temp",
            "*.cache"
        ],
        log_level="info"
    )


@app.post("/admin/update-summaries")
async def update_summaries(max_age_days: int = 30):
    """Manually trigger summary update check."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        await agent.ensure_summaries_updated(max_age_days=max_age_days)
        
        # Get summary statistics
        total_summaries = len(agent.schema_summaries)
        table_summaries = len([k for k in agent.schema_summaries.keys() if '.' not in k])
        column_summaries = len([k for k in agent.schema_summaries.keys() if '.' in k])
        
        return {
            "success": True,
            "message": "Summary update completed",
            "statistics": {
                "total_summaries": total_summaries,
                "table_summaries": table_summaries,
                "column_summaries": column_summaries
            }
        }
    except Exception as e:
        logger.error(f"Failed to update summaries: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/admin/check-schema-changes")
async def check_schema_changes():
    """Manually check for schema changes and update summaries."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    try:
        changes = await agent.check_for_schema_changes()
        return {
            "success": True,
            "message": "Schema check completed",
            "changes": {
                "new_tables": changes['new_tables'],
                "new_columns": changes['new_columns'],
                "removed_tables": changes['removed_tables'],
                "removed_columns": changes['removed_columns'],
                "summary": f"{len(changes['new_tables'])} new tables, {len(changes['new_columns'])} new columns"
            }
        }
    except Exception as e:
        logger.error(f"Failed to check schema changes: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/admin/cache/samples")
async def clear_sample_cache(table_name: Optional[str] = None):
    """Clear cached samples for a specific table or all tables"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        deleted_count = agent.sample_cache.clear_cache(table_name)
        
        return {
            "success": True,
            "message": f"Cleared cache for {'table ' + table_name if table_name else 'all tables'}",
            "deleted_count": deleted_count
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/cache/samples")
async def get_cached_samples():
    """Get list of tables with cached samples"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        cached_tables = agent.sample_cache.get_cached_tables()
        
        return {
            "success": True,
            "cached_tables": cached_tables,
            "count": len(cached_tables),
            "total_tables": len(agent.metadata)
        }
    except Exception as e:
        logger.error(f"Failed to get cached tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/prefetch-heads")
async def prefetch_table_heads(
    sample_size: int = 1000,
    max_concurrent: int = 5
):
    """Prefetch heads for all tables"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        results = await agent.prefetch_all_table_heads(
            sample_size=sample_size,
            max_concurrent=max_concurrent
        )
        
        return {
            "success": True,
            "message": "Table heads prefetched",
            "results": {
                "successful": len(results['success']),
                "failed": len(results['failed']),
                "total": results['total'],
                "failed_tables": results['failed']
            }
        }
    except Exception as e:
        logger.error(f"Failed to prefetch heads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/fetch-head/{table_name}")
async def fetch_single_table_head(
    table_name: str,
    sample_size: int = 1000,
    force_refresh: bool = False
):
    """Fetch head for a specific table"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        if table_name not in agent.metadata:
            raise HTTPException(status_code=404, detail=f"Table {table_name} not found")
        
        sample = await agent.fetch_table_head(
            table_name,
            sample_size=sample_size,
            force_refresh=force_refresh
        )
        
        if sample:
            return {
                "success": True,
                "table_name": table_name,
                "row_count": len(sample.get('rows', [])),
                "columns": sample.get('columns', []),
                "sample_rows": sample.get('rows', [])[:5]
            }
        else:
            return {
                "success": False,
                "table_name": table_name,
                "error": "Failed to fetch sample data"
            }
    
    except Exception as e:
        logger.error(f"Failed to fetch head for {table_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    # For development - in production use: uvicorn main:app
    run_server()