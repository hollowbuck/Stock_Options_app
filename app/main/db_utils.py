"""
Enhanced Database Utilities for SQL-First Architecture
Replaces all Excel I/O with SQLite operations
Maintains full compatibility with existing workflows

Author: Professional Refactored Version
Version: 3.1 - Performance Optimizations (Writer Queue + Bulk Writes)
"""

import os
import sqlite3
import sys
import time
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import atexit


# ============================================================================
# CONNECTION POOL (PER-DATABASE, CRITICAL FIX)
# ============================================================================

_POOL: Dict[str, list[sqlite3.Connection]] = {}  # Changed to dict keyed by db_path
_POOL_LOCK = None

# ============================================================================
# SERIALIZED DB WRITER THREAD (Performance Optimization)
# ============================================================================

_WRITE_QUEUE = None
_WRITER_THREAD = None
_WRITER_LOCK = None
_ACTIVE_WRITERS = {}  # Track active writers per DB path


def _get_lock():
    global _POOL_LOCK
    if _POOL_LOCK is None:
        import threading
        _POOL_LOCK = threading.Lock()
    return _POOL_LOCK


def _prep_conn(conn: sqlite3.Connection) -> None:
    """Prepare connection with performance optimizations"""
    try:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA cache_size=-64000')
        conn.execute('PRAGMA temp_store=MEMORY')
        conn.execute('PRAGMA foreign_keys=ON')
    except Exception:
        pass


# ============================================================================
# DATABASE CONNECTION MANAGEMENT
# ============================================================================

@contextmanager
def sqlite_connection(db_path: str, timeout: float = 30.0):
    """
    Context manager for SQLite connections with auto-commit/rollback
    Supports connection pooling for performance
    
    Args:
        db_path: Path to SQLite database file
        timeout: Database lock timeout in seconds
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    lock = _get_lock()
    
    # Try to get from pool FOR THIS SPECIFIC DATABASE
    conn = None
    with lock:
        db_pool = _POOL.get(db_path, [])
        if db_pool:
            conn = db_pool.pop()
            # Verify connection is still valid and points to correct database
            try:
                conn.execute("SELECT 1")
                # Verify we're connected to the right database
                cursor = conn.execute("PRAGMA database_list")
                actual_db = cursor.fetchone()[2]  # file path
                if actual_db and os.path.normpath(actual_db) != os.path.normpath(db_path):
                    # Wrong database! Close and create new connection
                    conn.close()
                    conn = None
            except:
                try:
                    conn.close()
                except:
                    pass
                conn = None
    
    # Create new connection if needed
    if conn is None:
        conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)
        _prep_conn(conn)
    
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        # Return to pool FOR THIS SPECIFIC DATABASE (max 5 connections per DB)
        with lock:
            if db_path not in _POOL:
                _POOL[db_path] = []
            if len(_POOL[db_path]) < 5:
                # Verify connection is still valid
                try:
                    conn.execute("SELECT 1")
                    _POOL[db_path].append(conn)
                except:
                    try:
                        conn.close()
                    except:
                        pass
            else:
                try:
                    conn.close()
                except:
                    pass


# ============================================================================
# SERIALIZED WRITER THREAD FUNCTIONS
# ============================================================================

def _get_writer_lock():
    global _WRITER_LOCK
    if _WRITER_LOCK is None:
        _WRITER_LOCK = threading.Lock()
    return _WRITER_LOCK


def _db_writer_worker(db_path: str, write_queue: queue.Queue):
    """
    Background worker that consumes write tasks and executes them in batches
    Reduces lock contention and groups writes into fewer transactions
    """
    conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
    _prep_conn(conn)
    
    try:
        while True:
            try:
                task = write_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                task_type, data = task
                
                if task_type == 'batch':
                    # data is list of (table_name, dataframe) tuples
                    conn.execute('BEGIN TRANSACTION')
                    try:
                        for table_name, df in data:
                            # Use bulk insert with method='multi' for speed
                            df.to_sql(
                                table_name, 
                                conn, 
                                if_exists='replace', 
                                index=False, 
                                method='multi',
                                chunksize=2000
                            )
                        conn.commit()
                    except Exception as e:
                        conn.rollback()
                        print(f"⚠️ Writer thread batch error: {e}")
                
                elif task_type == 'single':
                    # data is (table_name, dataframe)
                    table_name, df = data
                    try:
                        df.to_sql(
                            table_name, 
                            conn, 
                            if_exists='replace', 
                            index=False, 
                            method='multi',
                            chunksize=2000
                        )
                        conn.commit()
                    except Exception as e:
                        conn.rollback()
                        print(f"⚠️ Writer thread single error: {e}")
                
                write_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Writer thread unexpected error: {e}")
                
    finally:
        try:
            conn.close()
        except:
            pass


def get_writer_queue(db_path: str, max_queue_size: int = 256) -> queue.Queue:
    """
    Get or create a write queue for a specific database path
    
    Args:
        db_path: Path to the database
        max_queue_size: Maximum queue size (default: 256)
    
    Returns:
        Queue for submitting write tasks
    """
    global _ACTIVE_WRITERS
    
    lock = _get_writer_lock()
    
    with lock:
        if db_path not in _ACTIVE_WRITERS:
            write_queue = queue.Queue(maxsize=max_queue_size)
            writer_thread = threading.Thread(
                target=_db_writer_worker,
                args=(db_path, write_queue),
                daemon=True,
                name=f"DBWriter-{Path(db_path).stem}"
            )
            writer_thread.start()
            _ACTIVE_WRITERS[db_path] = (write_queue, writer_thread)
            
            # Register cleanup on exit
            def cleanup():
                try:
                    write_queue.put(None, timeout=1.0)
                    writer_thread.join(timeout=2.0)
                except:
                    pass
            atexit.register(cleanup)
        
        return _ACTIVE_WRITERS[db_path][0]


def submit_write_task(db_path: str, table_name: str, df: pd.DataFrame, use_queue: bool = True):
    """
    Submit a write task to the writer queue (non-blocking)
    
    Args:
        db_path: Database path
        table_name: Table name
        df: DataFrame to write
        use_queue: If True, use writer queue; if False, write directly (blocking)
    """
    if use_queue:
        write_queue = get_writer_queue(db_path)
        try:
            write_queue.put(('single', (table_name, df)), timeout=5.0)
        except queue.Full:
            # Fallback to direct write if queue is full
            print(f"⚠️ Write queue full, falling back to direct write for {table_name}")
            with sqlite_connection(db_path) as conn:
                df.to_sql(table_name, conn, if_exists='replace', index=False, method='multi', chunksize=2000)
    else:
        with sqlite_connection(db_path) as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False, method='multi', chunksize=2000)


def submit_batch_write_task(db_path: str, batch: List[Tuple[str, pd.DataFrame]], use_queue: bool = True):
    """
    Submit a batch of write tasks to the writer queue (most efficient)
    
    Args:
        db_path: Database path
        batch: List of (table_name, dataframe) tuples
        use_queue: If True, use writer queue; if False, write directly (blocking)
    """
    if use_queue:
        write_queue = get_writer_queue(db_path)
        try:
            write_queue.put(('batch', batch), timeout=5.0)
        except queue.Full:
            # Fallback to direct write if queue is full
            print(f"⚠️ Write queue full, falling back to direct batch write")
            with sqlite_connection(db_path) as conn:
                conn.execute('BEGIN TRANSACTION')
                try:
                    for table_name, df in batch:
                        df.to_sql(table_name, conn, if_exists='replace', index=False, method='multi', chunksize=2000)
                    conn.commit()
                except:
                    conn.rollback()
                    raise
    else:
        with sqlite_connection(db_path) as conn:
            conn.execute('BEGIN TRANSACTION')
            try:
                for table_name, df in batch:
                    df.to_sql(table_name, conn, if_exists='replace', index=False, method='multi', chunksize=2000)
                conn.commit()
            except:
                conn.rollback()
                raise


def wait_for_writes(db_path: str, timeout: float = 30.0):
    """
    Wait for all pending writes to complete
    
    Args:
        db_path: Database path
        timeout: Maximum time to wait (seconds)
    """
    if db_path in _ACTIVE_WRITERS:
        write_queue, _ = _ACTIVE_WRITERS[db_path]
        try:
            write_queue.join()  # Wait for all tasks to complete
        except:
            pass


def shutdown_writers():
    """Shutdown all active writer threads (cleanup)"""
    global _ACTIVE_WRITERS
    
    lock = _get_writer_lock()
    with lock:
        for db_path, (write_queue, writer_thread) in list(_ACTIVE_WRITERS.items()):
            try:
                write_queue.put(None, timeout=1.0)  # Shutdown signal
                writer_thread.join(timeout=2.0)
            except:
                pass
        _ACTIVE_WRITERS.clear()


atexit.register(shutdown_writers)


# ============================================================================
# TABLE MANAGEMENT
# ============================================================================

def list_tables(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    List all tables in the database with row counts
    
    Returns:
        DataFrame with table names and row counts
    """
    query = """
    SELECT name, sql 
    FROM sqlite_master 
    WHERE type='table' AND name NOT LIKE 'sqlite_%'
    ORDER BY name
    """
    tables_df = pd.read_sql_query(query, conn)
    
    # Add row counts
    if not tables_df.empty:
        row_counts = []
        for table_name in tables_df['name']:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM [{table_name}]").fetchone()[0]
                row_counts.append(count)
            except:
                row_counts.append(0)
        tables_df['row_count'] = row_counts
    
    return tables_df


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if table exists in database"""
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    result = conn.execute(query, (table_name,)).fetchone()
    return result is not None


def drop_table_if_exists(conn: sqlite3.Connection, table_name: str) -> None:
    """Drop table if it exists"""
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS [{table_name}]")
    cur.close()


def get_table_schema(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    """Get schema information for a table"""
    query = f"PRAGMA table_info([{table_name}])"
    return pd.read_sql_query(query, conn)


# ============================================================================
# DATA READING OPERATIONS
# ============================================================================

def read_table(db_path: str, table_name: str, columns: Optional[List[str]] = None,
               where_clause: Optional[str] = None, order_by: Optional[str] = None,
               limit: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Read data from a single table with optional filtering
    
    Args:
        db_path: Path to database file
        table_name: Name of table to read
        columns: List of columns to select (None = all)
        where_clause: SQL WHERE clause (without WHERE keyword)
        order_by: SQL ORDER BY clause (without ORDER BY keyword)
        limit: Maximum number of rows to return
    
    Returns:
        DataFrame with table data or None if error
    """
    try:
        with sqlite_connection(db_path) as conn:
            cols = ", ".join([f"[{c}]" for c in columns]) if columns else "*"
            query = f"SELECT {cols} FROM [{table_name}]"
            
            if where_clause:
                query += f" WHERE {where_clause}"
            if order_by:
                query += f" ORDER BY {order_by}"
            if limit:
                query += f" LIMIT {limit}"
            
            return pd.read_sql_query(query, conn)
    except Exception:
        return None


def read_all_tables(db_path: str) -> Dict[str, pd.DataFrame]:
    """
    Read all tables from database into dictionary of DataFrames
    
    Args:
        db_path: Path to database file
    
    Returns:
        Dictionary mapping table names to DataFrames
    """
    # Clear any cached connections to this database to ensure fresh read
    _clear_pool_for_db(db_path)
    
    frames = {}
    with sqlite_connection(db_path) as conn:
        tables = list_tables(conn)
        for table_name in tables['name'].tolist():
            if table_name not in {'sqlite_sequence'}:
                try:
                    frames[table_name] = pd.read_sql_query(f"SELECT * FROM [{table_name}]", conn)
                except Exception as e:
                    print(f"Warning: Could not read table {table_name}: {e}")
    return frames


def _clear_pool_for_db(db_path: str):
    """Clear cached connections for a specific database"""
    lock = _get_lock()
    with lock:
        if db_path in _POOL:
            conns = _POOL[db_path]
            for conn in conns:
                try:
                    conn.close()
                except:
                    pass
            _POOL[db_path] = []


# ============================================================================
# DATA WRITING OPERATIONS
# ============================================================================

def _normalize_table_name(name: str) -> str:
    """Normalize table name for SQLite compatibility"""
    import re
    norm = re.sub(r'[^A-Za-z0-9_]', '_', str(name))
    return (norm or 'table')[:63]


def write_dataframe(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame,
                    if_exists: str = 'replace', create_index_on: Optional[List[str]] = None) -> None:
    """
    Write DataFrame to database table with optional indexing (OPTIMIZED with bulk writes)
    
    Args:
        conn: Database connection
        table_name: Name of table to write
        df: DataFrame to write
        if_exists: How to behave if table exists ('fail', 'replace', 'append')
        create_index_on: List of columns to create indices on
    """
    safe_name = _normalize_table_name(table_name)
    
    # Write DataFrame with bulk insert optimization
    df.to_sql(safe_name, conn, if_exists=if_exists, index=False, method='multi', chunksize=2000)
    
    # Create indices if specified
    if create_index_on:
        for col in create_index_on:
            if col in df.columns:
                index_name = f"idx_{safe_name}_{col}"[:63]
                try:
                    conn.execute(f"CREATE INDEX IF NOT EXISTS [{index_name}] ON [{safe_name}]([{col}])")
                except Exception as e:
                    print(f"Warning: Could not create index on {col}: {e}")


def write_frames_as_tables(db_path: str, frames: Dict[str, pd.DataFrame],
                           if_exists: str = 'replace', indices: Optional[Dict[str, List[str]]] = None) -> None:
    """
    Write multiple DataFrames as tables in database
    
    Args:
        db_path: Path to database file
        frames: Dictionary mapping table names to DataFrames
        if_exists: How to behave if tables exist
        indices: Optional dict mapping table names to lists of columns to index
    """
    # Ensure directory exists
    db_path_obj = Path(db_path)
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # FIXED: Explicitly close connection after use to avoid Windows file locking
    conn = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
        _prep_conn(conn)
        
        conn.execute('BEGIN TRANSACTION')
        try:
            for name, df in frames.items():
                index_cols = indices.get(name) if indices else None
                write_dataframe(conn, name, df, if_exists=if_exists, create_index_on=index_cols)
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
    finally:
        # FIXED: Explicitly close connection to release file lock
        if conn:
            try:
                conn.close()
                # Small delay on Windows to ensure lock is released
                if sys.platform == 'win32':
                    time.sleep(0.05)
            except:
                pass


# ============================================================================
# SYMBOL MANAGEMENT (backward compatible)
# ============================================================================

def upsert_symbols_list(db_path: str, symbols: pd.DataFrame, table_name: str = "Symbols") -> None:
    """
    Update or insert symbols into symbols table
    Backward compatible: accepts DataFrame
    """
    with sqlite_connection(db_path) as conn:
        # Create table if not exists
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS [{table_name}] (
                symbol TEXT PRIMARY KEY,
                added_date TEXT,
                last_updated TEXT
            )
        """)
        
        # If DataFrame provided, convert to list
        if isinstance(symbols, pd.DataFrame):
            if 'Symbol' in symbols.columns:
                symbol_list = symbols['Symbol'].tolist()
            else:
                symbol_list = symbols.iloc[:, 0].tolist()
        else:
            symbol_list = symbols
        
        # Upsert symbols
        now = datetime.now().isoformat()
        for symbol in symbol_list:
            symbol_str = str(symbol).strip().upper()
            conn.execute(f"""
                INSERT INTO [{table_name}] (symbol, added_date, last_updated)
                VALUES (?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET last_updated = ?
            """, (symbol_str, now, now, now))


def read_symbols_list(db_path: str, table_name: str = "Symbols") -> Optional[pd.DataFrame]:
    """Get list of symbols from database"""
    return read_table(db_path, table_name)


# ============================================================================
# METADATA AND AUDIT TRACKING
# ============================================================================

def create_metadata_table(conn: sqlite3.Connection) -> None:
    """Create metadata table for tracking processing history"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processing_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name TEXT NOT NULL,
            operation_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            row_count INTEGER,
            processing_time_seconds REAL,
            parameters TEXT,
            status TEXT,
            error_message TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_metadata_table 
        ON processing_metadata(table_name)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_metadata_timestamp 
        ON processing_metadata(timestamp)
    """)


def log_processing_event(conn: sqlite3.Connection, table_name: str, operation_type: str,
                         row_count: int, processing_time: float, parameters: Optional[Dict] = None,
                         status: str = "success", error_message: Optional[str] = None) -> None:
    """Log a processing event to metadata table"""
    create_metadata_table(conn)
    
    params_json = json.dumps(parameters) if parameters else None
    timestamp = datetime.now().isoformat()
    
    conn.execute("""
        INSERT INTO processing_metadata 
        (table_name, operation_type, timestamp, row_count, processing_time_seconds, 
         parameters, status, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (table_name, operation_type, timestamp, row_count, processing_time, 
          params_json, status, error_message))


def get_processing_history(conn: sqlite3.Connection, table_name: Optional[str] = None,
                          limit: int = 100) -> pd.DataFrame:
    """Get processing history from metadata"""
    query = """
        SELECT * FROM processing_metadata
        WHERE 1=1
    """
    
    if table_name:
        query += f" AND table_name = '{table_name}'"
    
    query += f" ORDER BY timestamp DESC LIMIT {limit}"
    
    return pd.read_sql_query(query, conn)


# ============================================================================
# UTILITIES
# ============================================================================

def create_database_backup(db_path: str, backup_dir: Optional[str] = None,
                          timestamp: bool = True) -> str:
    """Create a backup copy of database file"""
    db_path_obj = Path(db_path)
    
    if backup_dir:
        backup_path_obj = Path(backup_dir)
        backup_path_obj.mkdir(parents=True, exist_ok=True)
    else:
        backup_path_obj = db_path_obj.parent
    
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{db_path_obj.stem}_{timestamp_str}{db_path_obj.suffix}"
    else:
        backup_name = f"{db_path_obj.stem}_backup{db_path_obj.suffix}"
    
    backup_path = backup_path_obj / backup_name
    
    # Use SQLite backup API for safe copying
    with sqlite_connection(str(db_path)) as source_conn:
        with sqlite_connection(str(backup_path)) as backup_conn:
            source_conn.backup(backup_conn)
    
    return str(backup_path)


def vacuum_database(db_path: str) -> None:
    """Vacuum database to reclaim space and optimize"""
    with sqlite_connection(db_path) as conn:
        conn.execute("VACUUM")


def get_database_size(db_path: str) -> Dict[str, Any]:
    """Get database size information"""
    if not os.path.exists(db_path):
        return {"exists": False}
    
    file_size = os.path.getsize(db_path)
    
    with sqlite_connection(db_path) as conn:
        tables = list_tables(conn)
        
        return {
            "exists": True,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "table_count": len(tables),
            "total_rows": tables['row_count'].sum() if not tables.empty else 0
        }


def export_table_to_csv(db_path: str, table_name: str, output_path: str) -> None:
    """Export table to CSV file"""
    with sqlite_connection(db_path) as conn:
        df = read_table(db_path, table_name)
        if df is not None:
            df.to_csv(output_path, index=False)


def close_all_connections():
    """Close all pooled connections (cleanup)"""
    global _POOL
    lock = _get_lock()
    with lock:
        for db_path, conns in _POOL.items():
            for conn in conns:
                try:
                    conn.close()
                except:
                    pass
        _POOL.clear()
