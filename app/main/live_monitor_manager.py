"""
Per-User Live Monitor Manager - OPTIMIZED & FIXED
Parallel processing with proper error handling and thread safety
"""

import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib


class LiveMonitorSession:
    """
    Per-user monitoring session with PARALLEL processing
    Optimized for speed with proper thread safety
    """
    
    def __init__(self, user_id: int, runtime_context, symbols: List[str], 
                 expiry_date: str, distance_percentage: float):
        self.user_id = user_id
        self.runtime_context = runtime_context
        self.symbols = symbols
        self.expiry_date = expiry_date
        self.distance_percentage = distance_percentage
        
        # State management
        self.is_running = False
        self.worker_thread = None
        self.stop_event = threading.Event()
        
        # Data storage (in-memory only)
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.calculated_data: Dict[str, pd.DataFrame] = {}
        self.filtered_data: Dict[str, List[Dict]] = {}
        self.last_update = None
        
        # Change tracking
        self.previous_filtered = {}
        self.changes_queue = queue.Queue(maxsize=100)
        
        # Performance tracking
        self.stats = {
            'fetch_count': 0,
            'last_fetch_time': 0,
            'avg_fetch_time': 0,
            'error_count': 0,
            'last_error': None
        }
        
        # Lock for thread-safe access
        self.data_lock = threading.Lock()
        
        # Thread pool for parallel processing
        self.executor = None
        self._create_executor()
    
    def _create_executor(self):
        """Create or recreate the thread pool executor"""
        if self.executor:
            try:
                self.executor.shutdown(wait=True, cancel_futures=True)
            except:
                pass
        
        self.executor = ThreadPoolExecutor(
            max_workers=20, 
            thread_name_prefix=f"LiveMon-U{self.user_id}"
        )
    
    def start(self, refresh_interval: int = 10):
        """Start the monitoring loop"""
        if self.is_running:
            return False
        
        self.is_running = True
        self.stop_event.clear()
        
        # Recreate executor if needed
        self._create_executor()
        
        self.worker_thread = threading.Thread(
            target=self._monitor_loop,
            args=(refresh_interval,),
            daemon=True,
            name=f"LiveMonitor-User{self.user_id}"
        )
        self.worker_thread.start()
        return True
    
    def stop(self):
        """Stop the monitoring loop"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        # Shutdown executor gracefully
        if self.executor:
            try:
                self.executor.shutdown(wait=True, cancel_futures=True)
            except:
                pass
        
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
    
    def _monitor_loop(self, refresh_interval: int):
        """Main monitoring loop - runs in background thread"""
        print(f"[LiveMonitor] Starting OPTIMIZED monitor loop for user {self.user_id}")
        
        while not self.stop_event.is_set():
            try:
                start_time = time.time()
                
                print(f"[LiveMonitor] User {self.user_id}: Fetching data (PARALLEL MODE)...")
                
                # Fetch and process data (PARALLEL)
                self._fetch_and_process_parallel()
                
                # Update stats
                elapsed = time.time() - start_time
                with self.data_lock:
                    self.stats['fetch_count'] += 1
                    self.stats['last_fetch_time'] = elapsed
                    
                    count = self.stats['fetch_count']
                    self.stats['avg_fetch_time'] = (
                        (self.stats['avg_fetch_time'] * (count - 1) + elapsed) / count
                    )
                
                print(f"[LiveMonitor] User {self.user_id}: Fetch completed in {elapsed:.2f}s (PARALLEL)")
                
                # Wait for next iteration
                self.stop_event.wait(timeout=refresh_interval)
                
            except Exception as e:
                import traceback
                print(f"[LiveMonitor] ERROR in monitor loop for user {self.user_id}:")
                print(traceback.format_exc())
                
                with self.data_lock:
                    self.stats['error_count'] += 1
                    self.stats['last_error'] = str(e)
                
                self.stop_event.wait(timeout=5)
    
    def _fetch_and_process_parallel(self):
        """
        OPTIMIZED: Parallel fetch and process
        Reduces 328s to ~20-30s for 154 symbols
        """
        print(f"[LiveMonitor] User {self.user_id}: Starting PARALLEL fetch_and_process")
        print(f"[LiveMonitor] Symbols: {len(self.symbols)} total")
        print(f"[LiveMonitor] Expiry: {self.expiry_date}")
        print(f"[LiveMonitor] Distance: {self.distance_percentage}%")
        
        # Import functions using importlib (avoids 'yield' keyword issues)
        try:
            yield_module = importlib.import_module('app.main.yield')
            fetch_cmp_zerodha = yield_module.fetch_cmp_zerodha
            fetch_option_chain_zerodha_live_monitor = yield_module.fetch_option_chain_zerodha_live_monitor
            print("[LiveMonitor] Successfully imported yield functions")
        except Exception as e:
            print(f"[LiveMonitor] CRITICAL: Cannot import yield functions: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # OPTIMIZATION 1: Batch fetch ALL CMPs in parallel (much faster)
        print(f"[LiveMonitor] Step 1: Batch fetching CMPs for {len(self.symbols)} symbols...")
        cmp_start = time.time()
        
        cmp_map = self._batch_fetch_cmps(fetch_cmp_zerodha)
        
        cmp_elapsed = time.time() - cmp_start
        print(f"[LiveMonitor] CMPs fetched in {cmp_elapsed:.2f}s (got {len(cmp_map)} valid)")
        
        # OPTIMIZATION 2: Process symbols in PARALLEL
        print(f"[LiveMonitor] Step 2: Processing option chains in PARALLEL...")
        chains_start = time.time()
        
        new_raw_data = {}
        new_calculated_data = {}
        
        # Submit all symbol processing tasks to thread pool
        future_to_symbol = {}
        for symbol in self.symbols:
            if symbol not in cmp_map:
                print(f"[LiveMonitor] Skipping {symbol} - no CMP available")
                continue
            
            future = self.executor.submit(
                self._process_single_symbol_safe,
                symbol,
                cmp_map[symbol],
                fetch_option_chain_zerodha_live_monitor
            )
            future_to_symbol[future] = symbol
        
        # Collect results as they complete
        completed = 0
        failed = 0
        
        for future in as_completed(future_to_symbol, timeout=60):
            symbol = future_to_symbol[future]
            try:
                result = future.result(timeout=15)  # 15s timeout per symbol
                if result:
                    df, calculated_df = result
                    new_raw_data[symbol] = df
                    new_calculated_data[symbol] = calculated_df
                    completed += 1
                else:
                    failed += 1
                    
            except TimeoutError:
                print(f"[LiveMonitor] TIMEOUT processing {symbol}")
                failed += 1
            except Exception as e:
                print(f"[LiveMonitor] ERROR processing {symbol}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
        
        chains_elapsed = time.time() - chains_start
        print(f"[LiveMonitor] Processed {completed} symbols successfully, {failed} failed in {chains_elapsed:.2f}s (PARALLEL)")
        
        # OPTIMIZATION 3: Apply filtering (fast, in-memory)
        print("[LiveMonitor] Step 3: Applying filters...")
        filter_start = time.time()
        
        new_filtered = self._apply_filtering(new_calculated_data)
        
        filter_elapsed = time.time() - filter_start
        print(f"[LiveMonitor] Filtering completed in {filter_elapsed:.2f}s")
        
        print(f"[LiveMonitor] Filtered results:")
        print(f"  - Filtered contracts: {len(new_filtered.get('filtered', []))}")
        print(f"  - High yield contracts: {len(new_filtered.get('high_yield', []))}")
        
        # Compute diffs
        diffs = self._compute_diffs(new_filtered)
        
        # Update shared state (thread-safe)
        with self.data_lock:
            self.raw_data = new_raw_data
            self.calculated_data = new_calculated_data
            self.filtered_data = new_filtered
            self.last_update = datetime.now()
            
            if diffs:
                try:
                    self.changes_queue.put_nowait(diffs)
                    print(f"[LiveMonitor] Queued diffs: {len(diffs.get('filtered_added', []))} filtered added, {len(diffs.get('high_yield_added', []))} high yield added")
                except queue.Full:
                    try:
                        self.changes_queue.get_nowait()
                        self.changes_queue.put_nowait(diffs)
                    except:
                        pass
    
    def _batch_fetch_cmps(self, fetch_cmp_func) -> Dict[str, float]:
        """
        OPTIMIZATION: Fetch CMPs for all symbols in parallel
        Uses ThreadPoolExecutor to fetch 20 symbols concurrently
        """
        cmp_map = {}
        
        # Submit all CMP fetch tasks
        future_to_symbol = {
            self.executor.submit(self._fetch_single_cmp_safe, fetch_cmp_func, symbol): symbol 
            for symbol in self.symbols
        }
        
        # Collect results with timeout
        for future in as_completed(future_to_symbol, timeout=30):
            symbol = future_to_symbol[future]
            try:
                cmp = future.result(timeout=5)
                if cmp:
                    cmp_map[symbol] = cmp
            except Exception as e:
                print(f"[LiveMonitor] CMP fetch failed for {symbol}: {e}")
        
        return cmp_map
    
    def _fetch_single_cmp_safe(self, fetch_cmp_func, symbol: str) -> Optional[float]:
        """Thread-safe wrapper for fetching single CMP"""
        try:
            return fetch_cmp_func(symbol)
        except Exception as e:
            print(f"[LiveMonitor] Exception fetching CMP for {symbol}: {e}")
            return None
    
    def _process_single_symbol_safe(self, symbol: str, cmp: float, 
                                    fetch_chain_func) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Thread-safe wrapper for processing a single symbol
        Returns: (raw_df, calculated_df) or None
        """
        try:
            # Fetch option chain
            contracts = fetch_chain_func(
                symbol, cmp, self.distance_percentage, self.expiry_date
            )
            
            if not contracts:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(contracts)
            
            if df.empty:
                return None
            
            # Apply calculations
            calculated_df = self._calculate_columns(df)
            
            return (df, calculated_df)
            
        except Exception as e:
            print(f"[LiveMonitor] Exception in _process_single_symbol_safe for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply calculated columns (same logic as filter.py)
        Fully vectorized for performance
        """
        result_df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['Margin_Required', 'BidPrice', 'Lot_Size', 'CMP', 
                        'StrikePrice', 'Days']
        if not all(col in result_df.columns for col in required_cols):
            print(f"[LiveMonitor] Missing required columns: {[c for c in required_cols if c not in result_df.columns]}")
            return result_df
        
        try:
            # Convert to numeric and handle zeros
            margin = pd.to_numeric(result_df['Margin_Required'], errors='coerce').replace(0, np.nan)
            bid_price = pd.to_numeric(result_df['BidPrice'], errors='coerce')
            lot_size = pd.to_numeric(result_df['Lot_Size'], errors='coerce')
            cmp = pd.to_numeric(result_df['CMP'], errors='coerce')
            strike = pd.to_numeric(result_df['StrikePrice'], errors='coerce')
            days = pd.to_numeric(result_df['Days'], errors='coerce').clip(lower=1)
            
            # Vectorized calculations
            result_df['Lots (Ill get)'] = np.floor(8_000_000 / margin).fillna(0).astype(np.int32)
            result_df['Premium/Lot'] = (lot_size * bid_price).fillna(0)
            result_df['Cost'] = 12.0
            result_df['Net Prem'] = result_df['Premium/Lot'] - 12.0
            result_df['Total Prem'] = result_df['Net Prem'] * result_df['Lots (Ill get)']
            result_df['Yield'] = ((result_df['Total Prem'] / days) * 30).fillna(0)
            result_df['Distance'] = (cmp - strike).fillna(0)
            result_df['Dist %'] = (result_df['Distance'] / cmp).fillna(0)
            result_df['Days Left'] = days.fillna(0).round(0).astype(np.int32)
            result_df['Prem/Day'] = (result_df['Total Prem'] / days).fillna(0).round(0).astype(np.int32)
            result_df['Prem Until Expiry'] = result_df['Total Prem'].fillna(0).round(0).astype(np.int32)
            
        except Exception as e:
            print(f"[LiveMonitor] Error in _calculate_columns: {e}")
            import traceback
            traceback.print_exc()
        
        return result_df
    
    def _apply_filtering(self, calculated_data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
        """
        Apply filtering criteria (same as filter.py)
        Returns dict with 'filtered' and 'high_yield' keys
        """
        filtered_rows = []
        high_yield_rows = []
        
        # Output columns
        out_cols = ['Symbol', 'CMP', 'OptionType', 'StrikePrice', 'BidPrice',
                   'Days Left', 'Prem/Day', 'Prem Until Expiry', 'Yield', 'Dist %']
        
        for symbol, df in calculated_data.items():
            if df.empty:
                continue
            
            if not set(['Yield', 'Dist %']).issubset(df.columns):
                continue
            
            try:
                # Filter 1: Yield > 50000 AND |Dist %| > 15%
                dist_pct_percentage = df['Dist %'] * 100
                mask_filtered = (df['Yield'] > 50000) & (abs(dist_pct_percentage) > 15)
                filtered_df = df[mask_filtered].copy()
                
                if len(filtered_df) > 0:
                    filtered_df['Dist %'] = (filtered_df['Dist %'] * 100).abs()
                    filtered_df['Distance'] = filtered_df['Distance'].abs()
                    
                    available_cols = [col for col in out_cols if col in filtered_df.columns]
                    filtered_subset = filtered_df[available_cols]
                    filtered_rows.extend(filtered_subset.to_dict('records'))
                
                # Filter 2: Yield > 80000
                mask_high = (df['Yield'] > 80000)
                high_yield_df = df[mask_high].copy()
                
                if len(high_yield_df) > 0:
                    high_yield_df['Dist %'] = high_yield_df['Dist %'] * 100
                    
                    available_cols = [col for col in out_cols if col in high_yield_df.columns]
                    high_yield_subset = high_yield_df[available_cols]
                    high_yield_rows.extend(high_yield_subset.to_dict('records'))
                    
            except Exception as e:
                print(f"[LiveMonitor] Error filtering {symbol}: {e}")
                continue
        
        # Sort results
        if filtered_rows:
            filtered_rows = sorted(filtered_rows, key=lambda x: x.get('Dist %', 0), reverse=True)
        
        if high_yield_rows:
            high_yield_rows = sorted(high_yield_rows, key=lambda x: x.get('Yield', 0), reverse=True)
        
        return {
            'filtered': filtered_rows,
            'high_yield': high_yield_rows
        }
    
    def _compute_diffs(self, new_filtered: Dict[str, List[Dict]]) -> Dict:
        """Compute diffs between previous and new filtered data"""
        diffs = {
            'timestamp': datetime.now().isoformat(),
            'filtered_added': [],
            'filtered_updated': [],
            'filtered_removed': [],
            'high_yield_added': [],
            'high_yield_updated': [],
            'high_yield_removed': []
        }
        
        def row_key(row):
            return f"{row.get('Symbol', '')}_{row.get('OptionType', '')}_{row.get('StrikePrice', 0)}"
        
        # Compare filtered
        old_filtered = {row_key(r): r for r in self.previous_filtered.get('filtered', [])}
        new_filtered_dict = {row_key(r): r for r in new_filtered.get('filtered', [])}
        
        for key, row in new_filtered_dict.items():
            if key not in old_filtered:
                diffs['filtered_added'].append(row)
            elif row != old_filtered[key]:
                diffs['filtered_updated'].append(row)
        
        for key in old_filtered:
            if key not in new_filtered_dict:
                diffs['filtered_removed'].append(old_filtered[key])
        
        # Compare high_yield
        old_high = {row_key(r): r for r in self.previous_filtered.get('high_yield', [])}
        new_high_dict = {row_key(r): r for r in new_filtered.get('high_yield', [])}
        
        for key, row in new_high_dict.items():
            if key not in old_high:
                diffs['high_yield_added'].append(row)
            elif row != old_high[key]:
                diffs['high_yield_updated'].append(row)
        
        for key in old_high:
            if key not in new_high_dict:
                diffs['high_yield_removed'].append(old_high[key])
        
        self.previous_filtered = new_filtered
        
        return diffs
    
    def get_current_data(self) -> Dict:
        """Get current filtered data (thread-safe)"""
        with self.data_lock:
            return {
                'filtered_data': self.filtered_data.copy(),
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'stats': self.stats.copy(),
                'symbols_count': len(self.symbols),
                'is_running': self.is_running
            }
    
    def get_changes(self, timeout: float = 1.0) -> Optional[Dict]:
        """Get latest changes (blocking with timeout)"""
        try:
            return self.changes_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class LiveMonitorManager:
    """Global manager for all user monitoring sessions"""
    
    def __init__(self):
        self.sessions: Dict[int, LiveMonitorSession] = {}
        self.lock = threading.Lock()
    
    def create_session(self, user_id: int, runtime_context, symbols: List[str],
                      expiry_date: str, distance_percentage: float) -> LiveMonitorSession:
        """Create or replace monitoring session for user"""
        with self.lock:
            if user_id in self.sessions:
                self.sessions[user_id].stop()
            
            session = LiveMonitorSession(
                user_id, runtime_context, symbols, expiry_date, distance_percentage
            )
            self.sessions[user_id] = session
            
            return session
    
    def get_session(self, user_id: int) -> Optional[LiveMonitorSession]:
        """Get existing session for user"""
        with self.lock:
            return self.sessions.get(user_id)
    
    def stop_session(self, user_id: int):
        """Stop and remove session for user"""
        with self.lock:
            if user_id in self.sessions:
                self.sessions[user_id].stop()
                del self.sessions[user_id]
    
    def stop_all(self):
        """Stop all active sessions"""
        with self.lock:
            for session in self.sessions.values():
                session.stop()
            self.sessions.clear()
    
    def get_active_sessions(self) -> List[int]:
        """Get list of active user IDs"""
        with self.lock:
            return [uid for uid, session in self.sessions.items() if session.is_running]
    
    def cleanup_stale_sessions(self, max_idle_minutes: int = 30):
        """Remove sessions that haven't updated recently"""
        cutoff = datetime.now().timestamp() - (max_idle_minutes * 60)
        
        with self.lock:
            stale_users = []
            for user_id, session in self.sessions.items():
                if session.last_update:
                    if session.last_update.timestamp() < cutoff:
                        stale_users.append(user_id)
            
            for user_id in stale_users:
                self.sessions[user_id].stop()
                del self.sessions[user_id]
            
            return len(stale_users)


# Global manager instance
_monitor_manager = LiveMonitorManager()


def get_monitor_manager() -> LiveMonitorManager:
    """Get global monitor manager instance"""
    return _monitor_manager