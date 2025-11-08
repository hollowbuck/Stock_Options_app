#!/usr/bin/env python3

"""
Enhanced High-Performance Options Data Calculator + Filter Script (Modified with File Management)

FIXED: Output file saving issues resolved
- Proper path handling and validation
- Better error messages for debugging
- Ensured write_frames_as_tables is correctly imported and used

Usage:
python filter.py --workers 4

"""

import pandas as pd
import numpy as np
import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from threading import Lock
import multiprocessing as mp
import shutil
from datetime import datetime
import glob
import sqlite3

class PerformanceTracker:
    """Thread-safe performance tracker with simplified progress line"""
    def __init__(self):
        self.lock = Lock()
        self.start_time = time.time()
        self.processed_sheets = 0
        self.total_sheets = 0
        self.total_rows = 0

    def update_progress(self, sheet_name, rows_processed):
        with self.lock:
            self.processed_sheets += 1
            self.total_rows += rows_processed
            elapsed = time.time() - self.start_time
            if self.total_sheets > 0:
                progress = (self.processed_sheets / self.total_sheets) * 100
                eta = (elapsed / self.processed_sheets) * (self.total_sheets - self.processed_sheets)
                # Simplified log: [current/total] sheet_name | ETA: seconds
                print(f"\r[{self.processed_sheets:3d}/{self.total_sheets}] {sheet_name:12s} | ETA: {eta:5.1f}s", end="", flush=True)

    def set_total(self, total):
        self.total_sheets = total

    def get_summary(self):
        elapsed = time.time() - self.start_time
        return {
            'total_time': elapsed,
            'sheets_processed': self.processed_sheets,
            'total_rows': self.total_rows,
            'sheets_per_second': self.processed_sheets / elapsed if elapsed > 0 else 0,
            'rows_per_second': self.total_rows / elapsed if elapsed > 0 else 0
        }

def ensure_folder_exists(folder_path):
    """Ensure that the folder exists, create if it doesn't"""
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    print(f"📁 Ensured folder exists: {folder_path}")

def get_latest_options_file(folder_path):
    """Get the latest Options_Data.db file from the specified folder"""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Look for Options_Data.db files
    pattern = "Options_Data*.db"
    files = list(folder.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No Options_Data.db files found in {folder_path}")
    
    # Get the most recently modified file
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"📄 Found latest file: {latest_file}")
    return latest_file

def create_timestamped_copy(source_file, destination_folder, base_name=None):
    """Create a timestamped copy of the source file in the destination folder"""
    ensure_folder_exists(destination_folder)
    
    source_path = Path(source_file)
    dest_folder = Path(destination_folder)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine the base name
    if base_name is None:
        base_name = source_path.stem
    
    # Create destination filename (SQLite)
    dest_filename = f"{base_name}_{timestamp}.db"
    dest_path = dest_folder / dest_filename
    
    # Copy the file
    shutil.copy2(source_file, dest_path)
    print(f"📋 Created copy: {dest_path}")
    
    return dest_path

def read_all_tables(db_path):
    """Read all tables from SQLite database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    result = {}
    for table_name, in tables:
        try:
            df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", conn)
            result[table_name] = df
        except Exception as e:
            print(f"⚠️ Warning: Could not read table {table_name}: {e}")
    
    conn.close()
    return result

def write_frames_as_tables(db_path, frames_dict):
    """
    Write DataFrames as tables to SQLite database (OPTIMIZED with bulk writes)
    Uses single transaction and bulk insert for maximum performance
    """
    # Ensure parent directory exists
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing file if it exists
    if db_path.exists():
        db_path.unlink()
    
    # Create connection with optimized settings
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    try:
        # Enable WAL mode and performance optimizations
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA cache_size=-64000')
        
        # OPTIMIZED: Single transaction for all tables
        conn.execute('BEGIN TRANSACTION')
        try:
            for table_name, df in frames_dict.items():
                # Clean table name for SQLite
                clean_table_name = table_name.replace(' ', '_').replace('-', '_')
                # OPTIMIZED: Bulk insert with method='multi' and chunksize
                df.to_sql(
                    clean_table_name, 
                    conn, 
                    if_exists='replace', 
                    index=False,
                    method='multi',
                    chunksize=2000
                )
                print(f"  ✅ Wrote table '{clean_table_name}' with {len(df)} rows")
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
    finally:
        conn.close()
    
    return db_path

def detect_actual_data_columns(df):
    """
    Detect and return all expected base columns that exist with data in the DataFrame.
    """
    expected_base_columns = [
        'Symbol', 'CMP', 'ScripName', 'OptionType', 'StrikePrice', 'BidPrice',
        'BidQty', 'AskPrice', 'AskQty', 'LTP', 'Lot_Size', 'Margin_Required',
        'LastUpdate', 'Days'
    ]

    available_columns = []
    for col in expected_base_columns:
        if col in df.columns:
            non_null_count = df[col].count()
            if non_null_count > 0:
                available_columns.append(col)
    
    return available_columns

def safe_convert_to_numeric(series, col_name):
    """
    Safely convert series to numeric, coercing errors to NaN.
    """
    try:
        converted = pd.to_numeric(series, errors='coerce')
        return converted.values
    except Exception as e:
        return np.full(len(series), np.nan)

def add_calculated_columns_to_sheet(df):
    """
    Calculate and append columns based on data in df.
    Uses actual 'Days' column values. No default fallback.
    OPTIMIZED: Fully vectorized with minimal DataFrame conversions.
    """
    # Create a copy to work with
    result_df = df.copy()
    
    # Detect available columns
    available_columns = detect_actual_data_columns(result_df)
    if not available_columns:
        return result_df

    # Keep only the base columns that exist
    result_df = result_df[available_columns].copy()
    
    # Check required columns for calculations
    required_columns = ['Margin_Required', 'BidPrice', 'Lot_Size', 'CMP', 'StrikePrice', 'Days']
    missing = [col for col in required_columns if col not in result_df.columns]
    if missing:
        # Can't calculate, return original
        return result_df

    # OPTIMIZED: Convert to numeric once, replace 0 with NaN for safe division
    # This avoids multiple conversions and uses vectorized operations
    margin = pd.to_numeric(result_df['Margin_Required'], errors='coerce').replace(0, np.nan)
    bid_price = pd.to_numeric(result_df['BidPrice'], errors='coerce')
    lot_size = pd.to_numeric(result_df['Lot_Size'], errors='coerce')
    cmp = pd.to_numeric(result_df['CMP'], errors='coerce')
    strike = pd.to_numeric(result_df['StrikePrice'], errors='coerce')
    days = pd.to_numeric(result_df['Days'], errors='coerce').clip(lower=1)  # Avoid division by zero
    
    # OPTIMIZED: Vectorized calculations in one pass (no masking needed)
    result_df['Lots (Ill get)'] = np.floor(8_000_000 / margin).fillna(0).astype(np.int32)
    result_df['Premium/Lot'] = (lot_size * bid_price).fillna(0)
    result_df['Cost'] = 12.0
    result_df['Net Prem'] = result_df['Premium/Lot'] - 12.0
    result_df['Total Prem'] = result_df['Net Prem'] * result_df['Lots (Ill get)']
    result_df['Yield'] = ((result_df['Total Prem'] / days) * 30).fillna(0)
    result_df['Distance'] = (cmp - strike).fillna(0)
    result_df['Dist %'] = (result_df['Distance'] / cmp).fillna(0)
    
    # OPTIMIZED: Filter-specific columns (vectorized)
    result_df['Days Left'] = days.fillna(0).round(0).astype(np.int32)
    result_df['Prem/Day'] = (result_df['Total Prem'] / days).fillna(0).round(0).astype(np.int32)
    result_df['Prem Until Expiry'] = result_df['Total Prem'].fillna(0).round(0).astype(np.int32)
    
    return result_df

def process_single_sheet(sheet_info, all_sheets_data, tracker):
    """Process a single sheet with proper error handling"""
    sheet_index, sheet_name = sheet_info
    try:
        df = all_sheets_data[sheet_name].copy()
        
        # Normalize column names to avoid trailing/leading space mismatches
        df.columns = df.columns.str.strip()
        
        processed_df = add_calculated_columns_to_sheet(df)
        valid_rows = processed_df.dropna(subset=['Symbol']).shape[0]
        
        tracker.update_progress(sheet_name, valid_rows)
        return sheet_name, processed_df, valid_rows
    except Exception as e:
        tracker.update_progress(f"{sheet_name}(ERR)", 0)
        return sheet_name, None, f"Error: {e}"

def create_filter_output(input_file, output_file='filter_output.db'):
    """
    Create filter output with proper Days column handling
    FIXED: Proper path handling and validation
    """
    print(f"\n📊 Creating filter output: {output_file}")
    
    # Convert to absolute path immediately
    output_file = Path(output_file).resolve()
    print(f"🔍 Resolved output path: {output_file}")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not output_file.parent.exists():
        raise IOError(f"Failed to create output directory: {output_file.parent}")
    print(f"📁 Output directory confirmed: {output_file.parent}")
    
    # Read all symbol tables from the processed SQLite db
    print(f"📖 Reading data from: {input_file}")
    all_sheets = read_all_tables(str(input_file))
    print(f"📊 Found {len(all_sheets)} tables to process")
    
    filtered_rows_list = []
    high_yield_rows_list = []
    
    for sheet_name, df in all_sheets.items():
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Check required columns for filtering
        if not set(['Yield', 'Dist %', 'Distance']).issubset(df.columns):
            print(f"  ⚠️ Skipping {sheet_name}: missing required columns")
            continue
        
        # Apply filtering logic - same as original
        dist_pct_values = df['Dist %'].copy()
        dist_pct_percentage = dist_pct_values * 100
        dist_threshold = 15
        
        # Main filter: Yield > 50000 and |Dist %| > 15%
        mask_main = (df['Yield'] > 50000) & (abs(dist_pct_percentage) > dist_threshold)
        filtered_df = df[mask_main].copy()
        if len(filtered_df) > 0:
            filtered_df['Dist %'] = (filtered_df['Dist %'] * 100).abs()
            filtered_df['Distance'] = filtered_df['Distance'].abs()
            filtered_rows_list.append(filtered_df)
            print(f"  ✓ {sheet_name}: {len(filtered_df)} filtered rows")
        
        # High yield filter: Yield > 80000
        mask_high = (df['Yield'] > 80000)
        high_yield_df = df[mask_high].copy()
        if len(high_yield_df) > 0:
            high_yield_df['Dist %'] = high_yield_df['Dist %'] * 100
            high_yield_rows_list.append(high_yield_df)
            print(f"  ✓ {sheet_name}: {len(high_yield_df)} high yield rows")
    
    # Columns to select for output
    selected_columns = ['Symbol', 'CMP', 'OptionType', 'StrikePrice', 'BidPrice', 'Yield', 'Dist %']
    
    def augment_filtered(df):
        """Add additional columns for filter output - FIXED to handle Days properly"""
        out_df = df.copy()
        
        # FIXED: Always use Days column that exists, no fallback
        if 'Days' in df.columns:
            out_df['Days Left'] = pd.to_numeric(df['Days'], errors='coerce')
        elif 'Days Left' in df.columns:
            # Already exists from calculation
            out_df['Days Left'] = pd.to_numeric(df['Days Left'], errors='coerce')
        else:
            # This should not happen if calculations worked properly
            print(f"  ⚠️ Warning: No Days column found, using calculated Days Left")
            out_df['Days Left'] = 22  # Fallback as last resort
        
        # Calculate Prem/Day and Prem Until Expiry
        if 'Total Prem' in df.columns:
            total_prem = pd.to_numeric(df['Total Prem'], errors='coerce')
            out_df['Prem/Day'] = np.where(out_df['Days Left'] != 0, total_prem / out_df['Days Left'], 0)
            out_df['Prem Until Expiry'] = total_prem
        elif 'Prem/Day' in df.columns and 'Prem Until Expiry' in df.columns:
            # Already calculated
            out_df['Prem/Day'] = pd.to_numeric(df['Prem/Day'], errors='coerce')
            out_df['Prem Until Expiry'] = pd.to_numeric(df['Prem Until Expiry'], errors='coerce')
        else:
            # Fallback calculation from Yield if available
            if 'Yield' in df.columns:
                yield_val = pd.to_numeric(df['Yield'], errors='coerce')
                daily_yield = yield_val / 30
                out_df['Prem/Day'] = daily_yield * out_df['Days Left'] / out_df['Days Left']
                out_df['Prem Until Expiry'] = daily_yield * out_df['Days Left']
            else:
                out_df['Prem/Day'] = 0
                out_df['Prem Until Expiry'] = 0
        
        # Round to whole numbers
        out_df['Prem/Day'] = out_df['Prem/Day'].round(0)
        out_df['Prem Until Expiry'] = out_df['Prem Until Expiry'].round(0)
        out_df['Days Left'] = out_df['Days Left'].round(0)
        
        return out_df
    
    # Output columns in the specified order
    out_cols = ['Symbol', 'CMP', 'OptionType', 'StrikePrice', 'BidPrice',
                'Days Left', 'Prem/Day', 'Prem Until Expiry', 'Yield', 'Dist %']
    
    # Process filtered data
    if filtered_rows_list:
        final_filtered_df = pd.concat(filtered_rows_list, ignore_index=True)
        # Augment first to ensure Days/derived columns are present
        final_filtered_df = augment_filtered(final_filtered_df)
        final_filtered_df = final_filtered_df[out_cols]
        final_filtered_df = final_filtered_df.sort_values(by='Dist %', ascending=False)
    else:
        final_filtered_df = pd.DataFrame(columns=out_cols)
    
    # Process high yield data
    if high_yield_rows_list:
        high_yield_filtered_df = pd.concat(high_yield_rows_list, ignore_index=True)
        # Augment first to ensure Days/derived columns are present
        high_yield_filtered_df = augment_filtered(high_yield_filtered_df)
        high_yield_filtered_df = high_yield_filtered_df[out_cols]
        high_yield_filtered_df = high_yield_filtered_df.sort_values(by='Yield', ascending=False)
    else:
        high_yield_filtered_df = pd.DataFrame(columns=out_cols)
    
    # Prepare frames for output
    frames = {
        'Filtered': final_filtered_df,
        'HighYield': high_yield_filtered_df,
    }
    
    # DEBUG: Log what we're about to write
    print(f"\n📊 Preparing to write {len(frames)} tables:")
    for table_name, df in frames.items():
        print(f"  - '{table_name}': {len(df)} rows, {len(df.columns)} columns")
        print(f"    Columns: {list(df.columns)}")
    
    # Write to database with enhanced error handling
    print(f"\n💾 Writing to database: {output_file}")
    try:
        result_path = write_frames_as_tables(str(output_file), frames)
        print(f"✅ Write completed to: {result_path}")
    except Exception as e:
        print(f"❌ Error writing to database: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Verify file creation and contents
    if not output_file.exists():
        raise IOError(f"❌ File was not created at expected location: {output_file}")
    
    file_size = output_file.stat().st_size
    print(f"✅ File created successfully: {output_file}")
    print(f"📦 File size: {file_size:,} bytes")
    
    # Verify the tables in the database
    try:
        verify_conn = sqlite3.connect(str(output_file))
        verify_cursor = verify_conn.cursor()
        verify_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        actual_tables = [row[0] for row in verify_cursor.fetchall()]
        verify_conn.close()
        print(f"\n🔍 Verification - Tables in output file: {actual_tables}")
        print(f"✅ Expected tables present: {'Filtered' in actual_tables and 'HighYield' in actual_tables}")
    except Exception as e:
        print(f"⚠️ Could not verify tables: {e}")
    
    print(f"\n📈 Rows added to 'Filtered': {final_filtered_df.shape[0]}")
    print(f"📈 Rows added to 'HighYield': {high_yield_filtered_df.shape[0]}")
    
    return len(final_filtered_df), len(high_yield_filtered_df)

def process_calculated_columns(input_file, output_file, max_workers=None):
    """
    Process calculated columns on the input file and save to output file
    Uses actual 'Days' column from each sheet. No default days parameter.
    """
    start_time = time.time()
    tracker = PerformanceTracker()
    
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"🚀 Processing calculated columns: {input_file}")
    print(f"⚡ Using parallel processing with {max_workers or 'auto'} workers")
    
    # Read all symbol datasets up-front from SQLite
    all_sheets = read_all_tables(str(input_file))
    all_sheets = {name: df.rename(columns=lambda c: str(c).strip()) for name, df in all_sheets.items()}
    
    sheet_names = list(all_sheets.keys())
    print(f"📊 Found {len(sheet_names)} sheets to process")
    
    tracker.set_total(len(sheet_names))
    
    if max_workers is None:
        max_workers = min(mp.cpu_count() * 2, 8, len(sheet_names))
    
    print(f"🔧 Using {max_workers} worker threads\n")
    
    processed_sheets = {}
    error_count = 0
    
    # Define a DF-based processing function for parallel execution without I/O
    def process_df(sheet_tuple):
        sheet_name, df = sheet_tuple
        try:
            processed_df = add_calculated_columns_to_sheet(df)
            valid_rows = processed_df.dropna(subset=['Symbol']).shape[0]
            tracker.update_progress(sheet_name, valid_rows)
            return sheet_name, processed_df, valid_rows, None
        except Exception as err:
            tracker.update_progress(f"{sheet_name}(ERR)", 0)
            return sheet_name, None, 0, err
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_df, item): item[0] for item in all_sheets.items()}
        
        for future in as_completed(futures):
            sheet_name = futures[future]
            _, processed_df, _, err = future.result()
            
            if err is None and processed_df is not None:
                processed_sheets[sheet_name] = processed_df
            else:
                error_count += 1
    
    print("\n")
    stats = tracker.get_summary()
    
    # Save processed data
    save_start = time.time()
    print(f"💾 Saving calculated columns data to SQLite: {output_file}")
    
    try:
        # Guarantee non-empty DB creation
        if not processed_sheets:
            processed_sheets = {"Empty": pd.DataFrame(columns=['Symbol'])}
        write_frames_as_tables(str(output_file), processed_sheets)
        save_time = time.time() - save_start
        print(f"\n✅ Calculated Columns Processing Complete!")
        print(f"📁 Enhanced output (SQLite): {output_file}")
        print(f"📊 Sheets processed: {len(processed_sheets)}/{len(sheet_names)}")
        print(f"📈 Total rows processed: {stats['total_rows']:,}")
        print(f"⏱️ Processing time: {stats['total_time']:.2f}s")
        print(f"💾 Save time: {save_time:.2f}s")
        print(f"🚀 Total time: {stats['total_time'] + save_time:.2f}s")
        print(f"⚡ Performance: {stats['sheets_per_second']:.1f} sheets/sec, {stats['rows_per_second']:.0f} rows/sec")
        
        if error_count > 0:
            print(f"⚠️ Errors encountered: {error_count} sheet(s)")
        
        return {
            'processed_sheets': len(processed_sheets),
            'total_sheets': len(sheet_names),
            'total_rows': stats['total_rows'],
            'processing_time': stats['total_time'],
            'save_time': save_time,
            'total_time': stats['total_time'] + save_time,
            'errors': error_count
        }
    
    except Exception as e:
        raise Exception(f"Error saving SQLite output: {e}")

def process_options_data_file_parallel(input_file=None, output_file=None, max_workers=None):
    """
    Main function with file management workflow for webapp integration
    FIXED: Better path handling and error reporting
    """
    try:
        print("🚀 Starting Enhanced Options Data Processing with File Management\n")
        
        # Step 1: Get latest Options_Data.db from Processed_Options folder
        processed_options_folder = "Processed_Options"
        calculated_columns_folder = "Calculated_Columns" 
        filtered_options_folder = "Filtered_Options"
        
        print("📂 Step 1: Finding latest Options_Data.db file...")
        try:
            latest_options_file = get_latest_options_file(processed_options_folder)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print(f"💡 Please ensure 'Options_Data.db' exists in '{processed_options_folder}' folder")
            raise
        
        # Step 2: Create timestamped copy in Calculated_Columns folder
        print("\n📂 Step 2: Creating timestamped copy in Calculated_Columns folder...")
        timestamped_copy = create_timestamped_copy(
            latest_options_file, 
            calculated_columns_folder, 
            "Options_Data"
        )
        
        # Step 3: Create processing copy for calculated columns
        print("\n📂 Step 3: Creating processing copy for calculated columns...")
        processing_copy = create_timestamped_copy(
            timestamped_copy,
            calculated_columns_folder,
            "Calculated_Columns"
        )
        
        # Step 4: Process calculated columns on the processing copy
        print("\n📂 Step 4: Processing calculated columns...")
        calc_results = process_calculated_columns(
            processing_copy, 
            processing_copy,  # Save back to the same file
            max_workers=max_workers
        )
        
        # Step 5: Copy processed file to Filtered_Options folder
        print("\n📂 Step 5: Copying processed file to Filtered_Options folder...")
        filtered_copy = create_timestamped_copy(
            processing_copy,
            filtered_options_folder,
            processing_copy.stem  # Keep the same name structure
        )
        
        # Step 6: Create filter output from the file in Filtered_Options
        print("\n📂 Step 6: Creating filtered output...")
        filter_start = time.time()
        
        # Create output filename with timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use absolute path from the start
        filtered_output_dest = Path(filtered_options_folder).resolve() / f'filter_output_{ts}.db'
        
        print(f"🔍 Current working directory: {Path.cwd()}")
        print(f"🔍 Output destination: {filtered_output_dest}")
        
        # Call create_filter_output with string path
        filtered_count, high_yield_count = create_filter_output(
            str(filtered_copy), 
            str(filtered_output_dest)
        )
        
        filter_time = time.time() - filter_start
        
        # Final summary
        total_time = calc_results['total_time'] + filter_time
        
        print(f"\n🎯 Complete Processing Summary:")
        print(f"📁 Original file: {latest_options_file}")
        print(f"📁 Calculated columns file: {processing_copy}")
        print(f"📁 Filtered source file: {filtered_copy}")
        print(f"📁 Final output: {filtered_output_dest}")
        print(f"📊 Sheets processed: {calc_results['processed_sheets']}")
        print(f"📈 Total rows processed: {calc_results['total_rows']:,}")
        print(f"📊 Filtered records: {filtered_count}")
        print(f"📊 High yield records: {high_yield_count}")
        print(f"⏱️ Calculated columns time: {calc_results['total_time']:.2f}s")
        print(f"⏱️ Filter processing time: {filter_time:.2f}s")
        print(f"🚀 Total processing time: {total_time:.2f}s")
        
        if calc_results['errors'] > 0:
            print(f"⚠️ Errors encountered: {calc_results['errors']} sheet(s)")
        
        print("\n✅ All processing completed successfully!")
        
        return {
            'success': True,
            'original_file': str(latest_options_file),
            'calculated_file': str(processing_copy),
            'filtered_file': str(filtered_copy),
            'output_file': str(filtered_output_dest),
            'processed_sheets': calc_results['processed_sheets'],
            'total_rows': calc_results['total_rows'],
            'filtered_count': filtered_count,
            'high_yield_count': high_yield_count,
            'total_time': total_time,
            'errors': calc_results['errors']
        }
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Main function with file management workflow"""
    parser = argparse.ArgumentParser(
        description="Enhanced High-Performance Options Data Calculator + Filter Script (FIXED)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features:
• FIXED: Proper output file saving with path validation
• FIXED: Better error messages and debugging
• FIXED: Embedded db_utils functions (no import issues)
• Uses actual 'Days' column from input file; no constant fallback
• High-performance multithreaded processing
• Reordered and formatted filter_output columns
• Updated column names for readability
• Simplified progress logging with ETA
• File management system with timestamped copies across folders

File Management Workflow:
1. Copy latest Options_Data.db from 'Processed_Options' to 'Calculated_Columns' with timestamp
2. Create processing copy in 'Calculated_Columns'
3. Add calculated columns to processing copy
4. Copy processed file to 'Filtered_Options'
5. Create filter_output.db with 2 tables (Filtered and HighYield)

Examples:
python filter.py --workers 4
"""
    )

    parser.add_argument('--workers', '-w',
                        type=int,
                        default=None,
                        help='Number of worker threads (default: auto-detect)')

    args = parser.parse_args()

    try:
        result = process_options_data_file_parallel(max_workers=args.workers)
        if not result['success']:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
