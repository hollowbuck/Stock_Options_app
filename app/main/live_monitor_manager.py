"""
Per-User Live Monitor Manager - FIXED VERSION
Addresses display corruption issues after first cycle
"""

import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import importlib
import copy

class LiveMonitorSession:
    """
    Per-user monitoring session with FIXED data consistency
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
        
        # Change tracking - FIXED: Deep copy to avoid reference issues
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
        
        # Thread pool
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
            max_workers=15, 
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
        print(f"[LiveMonitor] Starting FIXED monitor loop for user {self.user_id}")
        
        while not self.stop_event.is_set():
            try:
                start_time = time.time()
                
                print(f"[LiveMonitor] User {self.user_id}: Fetching data (FIXED MODE)...")
                
                # Fetch and process data
                self._fetch_and_process_ultra_fast()
                
                # Update stats
                elapsed = time.time() - start_time
                with self.data_lock:
                    self.stats['fetch_count'] += 1
                    self.stats['last_fetch_time'] = elapsed
                    
                    count = self.stats['fetch_count']
                    if count > 1:
                        self.stats['avg_fetch_time'] = (
                            (self.stats['avg_fetch_time'] * (count - 1) + elapsed) / count
                        )
                    else:
                        self.stats['avg_fetch_time'] = elapsed
                
                print(f"[LiveMonitor] User {self.user_id}: ✅ Fetch completed in {elapsed:.2f}s (FIXED MODE)")
                
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
    
    def _fetch_and_process_ultra_fast(self):
        """
        FIXED: Ultra-fast processing with consistent data structure
        """
        print(f"[LiveMonitor] Starting FIXED fetch_and_process")
        print(f"[LiveMonitor] Symbols: {len(self.symbols)} | Expiry: {self.expiry_date} | Distance: {self.distance_percentage}%")
        
        # Import functions
        try:
            yield_module = importlib.import_module('app.main.yield')
            fetch_cmp_zerodha = yield_module.fetch_cmp_zerodha
            fetch_option_chain_zerodha_live_monitor = yield_module.fetch_option_chain_zerodha_live_monitor
        except Exception as e:
            print(f"[LiveMonitor] CRITICAL: Cannot import yield functions: {e}")
            raise
        
        # STEP 1: Parallel CMP fetching
        print(f"[LiveMonitor] Step 1: Fetching CMPs...")
        cmp_start = time.time()
        
        cmp_map = self._ultra_fast_cmp_fetch(fetch_cmp_zerodha)
        
        cmp_elapsed = time.time() - cmp_start
        print(f"[LiveMonitor] ✅ CMPs: {len(cmp_map)}/{len(self.symbols)} in {cmp_elapsed:.2f}s")
        
        # STEP 2: Parallel option chain processing
        print(f"[LiveMonitor] Step 2: Processing option chains...")
        chains_start = time.time()
        
        new_raw_data, new_calculated_data = self._ultra_fast_chain_processing(
            cmp_map, fetch_option_chain_zerodha_live_monitor
        )
        
        chains_elapsed = time.time() - chains_start
        print(f"[LiveMonitor] ✅ Chains: {len(new_calculated_data)} processed in {chains_elapsed:.2f}s")
        
        # STEP 3: Apply filtering with consistent data structure
        filter_start = time.time()
        new_filtered = self._apply_filtering_fixed(new_calculated_data)
        filter_elapsed = time.time() - filter_start
        
        print(f"[LiveMonitor] ✅ Filtered: {len(new_filtered.get('filtered', []))} contracts, {len(new_filtered.get('high_yield', []))} high-yield in {filter_elapsed:.2f}s")
        
        # FIXED: Compute diffs with proper deep copying
        diffs = self._compute_diffs_fixed(new_filtered)
        
        # Update shared state
        with self.data_lock:
            self.raw_data = new_raw_data
            self.calculated_data = new_calculated_data
            self.filtered_data = copy.deepcopy(new_filtered)  # FIXED: Deep copy
            self.last_update = datetime.now()
            
            # Queue diffs if any changes detected
            if diffs and self._has_meaningful_changes(diffs):
                try:
                    self.changes_queue.put_nowait(diffs)
                    print(f"[LiveMonitor] Queued meaningful diffs")
                except queue.Full:
                    try:
                        self.changes_queue.get_nowait()
                        self.changes_queue.put_nowait(diffs)
                    except:
                        pass
    
    def _has_meaningful_changes(self, diffs: Dict) -> bool:
        """Check if diffs contain meaningful changes"""
        return (
            len(diffs.get('filtered_added', [])) > 0 or
            len(diffs.get('filtered_removed', [])) > 0 or
            len(diffs.get('high_yield_added', [])) > 0 or
            len(diffs.get('high_yield_removed', [])) > 0
        )
    
    def _ultra_fast_cmp_fetch(self, fetch_cmp_func) -> Dict[str, float]:
        """Fast CMP fetching with error handling"""
        cmp_map = {}
        chunk_size = 25
        
        for i in range(0, len(self.symbols), chunk_size):
            chunk = self.symbols[i:i+chunk_size]
            
            # Submit chunk
            future_to_symbol = {
                self.executor.submit(fetch_cmp_func, symbol): symbol 
                for symbol in chunk
            }
            
            # Collect with timeout
            try:
                for future in as_completed(future_to_symbol, timeout=12):
                    symbol = future_to_symbol[future]
                    try:
                        cmp = future.result(timeout=2)
                        if cmp and cmp > 0:  # FIXED: Validate CMP value
                            cmp_map[symbol] = float(cmp)
                    except Exception:
                        pass
            except TimeoutError:
                # Collect completed futures
                for future in future_to_symbol:
                    if future.done():
                        try:
                            cmp = future.result(timeout=0)
                            if cmp and cmp > 0:
                                cmp_map[future_to_symbol[future]] = float(cmp)
                        except:
                            pass
        
        return cmp_map
    
    def _ultra_fast_chain_processing(self, cmp_map: Dict[str, float], 
                                     fetch_chain_func) -> Tuple[Dict, Dict]:
        """Process option chains with consistent data structure"""
        new_raw_data = {}
        new_calculated_data = {}
        
        valid_symbols = [s for s in self.symbols if s in cmp_map]
        chunk_size = 12
        
        for i in range(0, len(valid_symbols), chunk_size):
            chunk = valid_symbols[i:i+chunk_size]
            
            # Submit chunk
            future_to_symbol = {}
            for symbol in chunk:
                future = self.executor.submit(
                    self._process_single_symbol_safe,
                    symbol,
                    cmp_map[symbol],
                    fetch_chain_func
                )
                future_to_symbol[future] = symbol
            
            # Collect with timeout
            try:
                for future in as_completed(future_to_symbol, timeout=25):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result(timeout=8)
                        if result:
                            df, calculated_df = result
                            new_raw_data[symbol] = df
                            new_calculated_data[symbol] = calculated_df
                    except:
                        pass
            except TimeoutError:
                # Collect completed
                for future in future_to_symbol:
                    if future.done():
                        try:
                            result = future.result(timeout=0)
                            if result:
                                df, calculated_df = result
                                new_raw_data[future_to_symbol[future]] = df
                                new_calculated_data[future_to_symbol[future]] = calculated_df
                        except:
                            pass
        
        return new_raw_data, new_calculated_data
    
    def _process_single_symbol_safe(self, symbol: str, cmp: float, 
                                    fetch_chain_func) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Process a single symbol with error handling"""
        try:
            contracts = fetch_chain_func(symbol, cmp, self.distance_percentage, self.expiry_date)
            
            if not contracts:
                return None
            
            df = pd.DataFrame(contracts)
            if df.empty:
                return None
            
            calculated_df = self._calculate_columns_fixed(df)
            return (df, calculated_df)
            
        except:
            return None
    
    def _calculate_columns_fixed(self, df: pd.DataFrame) -> pd.DataFrame:
        """FIXED: Calculate columns with consistent data types"""
        result_df = df.copy()
        
        required_cols = ['Margin_Required', 'BidPrice', 'Lot_Size', 'CMP', 'StrikePrice', 'Days']
        if not all(col in result_df.columns for col in required_cols):
            return result_df
        
        try:
            # FIXED: Ensure consistent numeric conversion
            margin = pd.to_numeric(result_df['Margin_Required'], errors='coerce').replace(0, np.nan)
            bid_price = pd.to_numeric(result_df['BidPrice'], errors='coerce')
            lot_size = pd.to_numeric(result_df['Lot_Size'], errors='coerce')
            cmp = pd.to_numeric(result_df['CMP'], errors='coerce')
            strike = pd.to_numeric(result_df['StrikePrice'], errors='coerce')
            days = pd.to_numeric(result_df['Days'], errors='coerce').clip(lower=1)
            
            # FIXED: Consistent calculations with proper rounding
            result_df['Lots (Ill get)'] = np.floor(8_000_000 / margin).fillna(0).astype(np.int32)
            result_df['Premium/Lot'] = (lot_size * bid_price).fillna(0).round(2)
            result_df['Cost'] = 12.0
            result_df['Net Prem'] = (result_df['Premium/Lot'] - 12.0).round(2)
            result_df['Total Prem'] = (result_df['Net Prem'] * result_df['Lots (Ill get)']).round(2)
            result_df['Yield'] = ((result_df['Total Prem'] / days) * 30).fillna(0).round(2)
            result_df['Distance'] = (cmp - strike).fillna(0).round(2)
            result_df['Dist %'] = (result_df['Distance'] / cmp * 100).fillna(0).round(2)
            result_df['Days Left'] = days.fillna(0).round(0).astype(np.int32)
            result_df['Prem/Day'] = (result_df['Total Prem'] / days).fillna(0).round(0).astype(np.int32)
            result_df['Prem Until Expiry'] = result_df['Total Prem'].fillna(0).round(0).astype(np.int32)
            
        except Exception as e:
            print(f"[LiveMonitor] Error in calculate_columns_fixed: {e}")
            pass
        
        return result_df
    
    def _apply_filtering_fixed(self, calculated_data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict]]:
        """FIXED: Apply filtering with consistent data structure"""
        filtered_rows = []
        high_yield_rows = []
        
        # FIXED: Ensure consistent column names and data types
        out_cols = ['Symbol', 'CMP', 'OptionType', 'StrikePrice', 'BidPrice',
                   'Days Left', 'Prem/Day', 'Prem Until Expiry', 'Yield', 'Dist %']
        
        for symbol, df in calculated_data.items():
            if df.empty or not set(['Yield', 'Dist %']).issubset(df.columns):
                continue
            
            try:
                # FIXED: Consistent filtering logic
                mask_filtered = (df['Yield'] > 50000) & (abs(df['Dist %']) > 15)
                filtered_df = df[mask_filtered].copy()
                
                if len(filtered_df) > 0:
                    # FIXED: Ensure consistent data formatting
                    for row_dict in filtered_df.to_dict('records'):
                        # Ensure all required fields are present with correct types
                        clean_row = {}
                        for col in out_cols:
                            if col in row_dict:
                                val = row_dict[col]
                                if pd.isna(val):
                                    clean_row[col] = 0 if col in ['Yield', 'Prem/Day', 'Prem Until Expiry'] else ''
                                else:
                                    clean_row[col] = val
                            else:
                                clean_row[col] = 0 if col in ['Yield', 'Prem/Day', 'Prem Until Expiry'] else ''
                        
                        # Ensure Dist % is absolute
                        if 'Dist %' in clean_row:
                            clean_row['Dist %'] = abs(float(clean_row['Dist %']))
                        
                        filtered_rows.append(clean_row)
                
                # High yield filter
                mask_high = (df['Yield'] > 80000)
                high_yield_df = df[mask_high].copy()
                
                if len(high_yield_df) > 0:
                    for row_dict in high_yield_df.to_dict('records'):
                        # Ensure all required fields are present
                        clean_row = {}
                        for col in out_cols:
                            if col in row_dict:
                                val = row_dict[col]
                                if pd.isna(val):
                                    clean_row[col] = 0 if col in ['Yield', 'Prem/Day', 'Prem Until Expiry'] else ''
                                else:
                                    clean_row[col] = val
                            else:
                                clean_row[col] = 0 if col in ['Yield', 'Prem/Day', 'Prem Until Expiry'] else ''
                        
                        high_yield_rows.append(clean_row)
                        
            except Exception as e:
                print(f"[LiveMonitor] Error filtering {symbol}: {e}")
                continue
        
        # FIXED: Consistent sorting
        if filtered_rows:
            filtered_rows = sorted(filtered_rows, key=lambda x: float(x.get('Dist %', 0)), reverse=True)
        if high_yield_rows:
            high_yield_rows = sorted(high_yield_rows, key=lambda x: float(x.get('Yield', 0)), reverse=True)
        
        return {'filtered': filtered_rows, 'high_yield': high_yield_rows}
    
    def _compute_diffs_fixed(self, new_filtered: Dict[str, List[Dict]]) -> Dict:
        """FIXED: Compute diffs with proper deep copying"""
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
            # FIXED: More robust key generation
            symbol = str(row.get('Symbol', ''))
            option_type = str(row.get('OptionType', ''))
            strike = str(row.get('StrikePrice', 0))
            return f"{symbol}_{option_type}_{strike}"
        
        # FIXED: Use deep copy of previous data
        old_filtered = {row_key(r): r for r in self.previous_filtered.get('filtered', [])}
        new_filtered_dict = {row_key(r): r for r in new_filtered.get('filtered', [])}
        
        # Find added and updated in filtered
        for key, row in new_filtered_dict.items():
            if key not in old_filtered:
                diffs['filtered_added'].append(row)
            else:
                # FIXED: Better comparison ignoring small float differences
                old_row = old_filtered[key]
                if self._rows_different(old_row, row):
                    diffs['filtered_updated'].append(row)
        
        # Find removed from filtered
        for key, row in old_filtered.items():
            if key not in new_filtered_dict:
                diffs['filtered_removed'].append(row)
        
        # Same for high_yield
        old_high = {row_key(r): r for r in self.previous_filtered.get('high_yield', [])}
        new_high_dict = {row_key(r): r for r in new_filtered.get('high_yield', [])}
        
        for key, row in new_high_dict.items():
            if key not in old_high:
                diffs['high_yield_added'].append(row)
            else:
                old_row = old_high[key]
                if self._rows_different(old_row, row):
                    diffs['high_yield_updated'].append(row)
        
        for key, row in old_high.items():
            if key not in new_high_dict:
                diffs['high_yield_removed'].append(row)
        
        # FIXED: Update previous_filtered with deep copy
        self.previous_filtered = copy.deepcopy(new_filtered)
        
        return diffs
    
    def _rows_different(self, row1: Dict, row2: Dict, threshold: float = 0.01) -> bool:
        """FIXED: Compare rows with tolerance for float differences"""
        for key in row1.keys():
            if key not in row2:
                return True
            
            val1, val2 = row1[key], row2[key]
            
            # For numeric values, use threshold comparison
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(float(val1) - float(val2)) > threshold:
                    return True
            else:
                if str(val1) != str(val2):
                    return True
        
        return False
    
    def get_current_data(self) -> Dict:
        """Get current filtered data (thread-safe)"""
        with self.data_lock:
            return {
                'filtered_data': copy.deepcopy(self.filtered_data),  # FIXED: Deep copy
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'stats': self.stats.copy(),
                'symbols_count': len(self.symbols),
                'is_running': self.is_running
            }
    
    def get_changes(self, timeout: float = 1.0) -> Optional[Dict]:
        """Get latest changes"""
        try:
            return self.changes_queue.get(timeout=timeout)
        except queue.Empty:
            return None

class LiveMonitorManager:
    """Global manager"""
    
    def __init__(self):
        self.sessions: Dict[int, LiveMonitorSession] = {}
        self.lock = threading.Lock()
    
    def create_session(self, user_id: int, runtime_context, symbols: List[str],
                      expiry_date: str, distance_percentage: float) -> LiveMonitorSession:
        with self.lock:
            if user_id in self.sessions:
                self.sessions[user_id].stop()
            session = LiveMonitorSession(user_id, runtime_context, symbols, expiry_date, distance_percentage)
            self.sessions[user_id] = session
            return session
    
    def get_session(self, user_id: int) -> Optional[LiveMonitorSession]:
        with self.lock:
            return self.sessions.get(user_id)
    
    def stop_session(self, user_id: int):
        with self.lock:
            if user_id in self.sessions:
                self.sessions[user_id].stop()
                del self.sessions[user_id]
    
    def stop_all(self):
        with self.lock:
            for session in self.sessions.values():
                session.stop()
            self.sessions.clear()
    
    def get_active_sessions(self) -> List[int]:
        with self.lock:
            return [uid for uid, session in self.sessions.items() if session.is_running]

_monitor_manager = LiveMonitorManager()

def get_monitor_manager() -> LiveMonitorManager:
    return _monitor_manager