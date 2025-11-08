"""
Per-User Workspace Management
Provides isolated file storage for each user
"""

import os
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

# Root directory for all user workspaces
WORKSPACE_ROOT = Path('data/users')


class UserWorkspace:
    """
    Manages file workspace for a single user
    Provides isolated directories for data, outputs, and temp files
    """
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.root = WORKSPACE_ROOT / str(user_id)
        
        # Subdirectories
        self.processed_options = self.root / 'processed_options'
        self.calculated_columns = self.root / 'calculated_columns'
        self.filtered_options = self.root / 'filtered_options'
        self.exports = self.root / 'exports'
        self.temp = self.root / 'temp'
        self.logs = self.root / 'logs'
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create all required directories"""
        for directory in [
            self.root,
            self.processed_options,
            self.calculated_columns,
            self.filtered_options,
            self.exports,
            self.temp,
            self.logs
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_options_data_path(self) -> Path:
        """Get path to Options_Data.db"""
        return self.processed_options / 'Options_Data.db'
    
    def get_main_list_path(self) -> Path:
        """Get path to Main_List.db"""
        return self.root / 'Main_List.db'
    
    def get_filter_output_path(self, timestamp: Optional[str] = None) -> Path:
        """
        Get path to filter output file
        
        Args:
            timestamp: Optional timestamp string (YYYYMMDD_HHMMSS)
        
        Returns:
            Path to filter_output_*.db
        """
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.filtered_options / f'filter_output_{timestamp}.db'
    
    def get_latest_filter_output(self) -> Optional[Path]:
        """Get the latest filter output file"""
        files = sorted(self.filtered_options.glob('filter_output_*.db'), 
                      key=lambda f: f.stat().st_mtime, 
                      reverse=True)
        return files[0] if files else None
    
    def get_calculated_columns_path(self, timestamp: Optional[str] = None) -> Path:
        """Get path to calculated columns file"""
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.calculated_columns / f'Calculated_Columns_{timestamp}.db'
    
    def get_export_path(self, filename: str) -> Path:
        """Get path for export files"""
        return self.exports / filename
    
    def get_temp_path(self, filename: str) -> Path:
        """Get path for temporary files"""
        return self.temp / filename
    
    def get_log_path(self, filename: str) -> Path:
        """Get path for log files"""
        return self.logs / filename
    
    def get_disk_usage(self) -> dict:
        """
        Get disk usage statistics for user workspace
        
        Returns:
            Dict with size information
        """
        def get_dir_size(path: Path) -> int:
            """Get total size of directory in bytes"""
            total = 0
            try:
                for entry in path.rglob('*'):
                    if entry.is_file():
                        total += entry.stat().st_size
            except:
                pass
            return total
        
        return {
            'total_bytes': get_dir_size(self.root),
            'processed_options_bytes': get_dir_size(self.processed_options),
            'calculated_columns_bytes': get_dir_size(self.calculated_columns),
            'filtered_options_bytes': get_dir_size(self.filtered_options),
            'exports_bytes': get_dir_size(self.exports),
            'temp_bytes': get_dir_size(self.temp)
        }
    
    def get_disk_usage_mb(self) -> dict:
        """Get disk usage in MB"""
        usage = self.get_disk_usage()
        return {k: v / (1024 * 1024) for k, v in usage.items()}
    
    def cleanup_old_files(self, days_old: int = 7):
        """
        Clean up files older than specified days
        
        Args:
            days_old: Delete files older than this many days
        """
        cutoff = datetime.now() - timedelta(days=days_old)
        cutoff_timestamp = cutoff.timestamp()
        
        deleted_count = 0
        deleted_size = 0
        
        # Cleanup temp files
        for file in self.temp.glob('*'):
            try:
                if file.is_file() and file.stat().st_mtime < cutoff_timestamp:
                    size = file.stat().st_size
                    file.unlink()
                    deleted_count += 1
                    deleted_size += size
            except:
                pass
        
        # Cleanup old calculated columns (keep last 10)
        calc_files = sorted(self.calculated_columns.glob('Calculated_Columns_*.db'),
                           key=lambda f: f.stat().st_mtime,
                           reverse=True)
        for file in calc_files[10:]:
            try:
                size = file.stat().st_size
                file.unlink()
                deleted_count += 1
                deleted_size += size
            except:
                pass
        
        # Cleanup old filter outputs (keep last 10)
        filter_files = sorted(self.filtered_options.glob('filter_output_*.db'),
                            key=lambda f: f.stat().st_mtime,
                            reverse=True)
        for file in filter_files[10:]:
            try:
                size = file.stat().st_size
                file.unlink()
                deleted_count += 1
                deleted_size += size
            except:
                pass
        
        return {
            'deleted_count': deleted_count,
            'deleted_size_mb': deleted_size / (1024 * 1024)
        }
    
    def clear_temp(self):
        """Clear all temporary files"""
        try:
            shutil.rmtree(self.temp)
            self.temp.mkdir(parents=True, exist_ok=True)
        except:
            pass
    
    def delete_workspace(self):
        """Delete entire user workspace (use with caution!)"""
        try:
            shutil.rmtree(self.root)
        except:
            pass
    
    def list_files(self, directory: str = 'all') -> list:
        """
        List files in workspace
        
        Args:
            directory: Which directory to list ('all', 'processed', 'filtered', 'exports', etc.)
        
        Returns:
            List of file info dicts
        """
        files = []
        
        if directory == 'all':
            search_dirs = [
                ('processed_options', self.processed_options),
                ('calculated_columns', self.calculated_columns),
                ('filtered_options', self.filtered_options),
                ('exports', self.exports)
            ]
        else:
            dir_map = {
                'processed': self.processed_options,
                'calculated': self.calculated_columns,
                'filtered': self.filtered_options,
                'exports': self.exports,
                'temp': self.temp
            }
            search_dirs = [(directory, dir_map.get(directory, self.root))]
        
        for dir_name, dir_path in search_dirs:
            try:
                for file in dir_path.glob('*'):
                    if file.is_file():
                        stat = file.stat()
                        files.append({
                            'directory': dir_name,
                            'filename': file.name,
                            'path': str(file),
                            'size_bytes': stat.st_size,
                            'size_mb': stat.st_size / (1024 * 1024),
                            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
            except:
                pass
        
        return sorted(files, key=lambda x: x['modified'], reverse=True)


def get_user_workspace(user_id: int) -> UserWorkspace:
    """
    Get workspace instance for a user
    
    Args:
        user_id: User ID
    
    Returns:
        UserWorkspace instance
    """
    return UserWorkspace(user_id)


def cleanup_all_workspaces(days_old: int = 7):
    """
    Cleanup old files from all user workspaces
    
    Args:
        days_old: Delete files older than this many days
    """
    total_deleted = 0
    total_size = 0
    
    for user_dir in WORKSPACE_ROOT.iterdir():
        if user_dir.is_dir() and user_dir.name.isdigit():
            try:
                user_id = int(user_dir.name)
                workspace = UserWorkspace(user_id)
                result = workspace.cleanup_old_files(days_old)
                total_deleted += result['deleted_count']
                total_size += result['deleted_size_mb']
            except:
                pass
    
    return {
        'total_deleted_count': total_deleted,
        'total_deleted_size_mb': total_size
    }

