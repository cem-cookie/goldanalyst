import os
import zipfile
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict


class AutoArchiver:
    """Handle data archival with maximum compression"""
    
    def __init__(self, archive_dir: str = "data/archive"):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.archive_dir / "archive_index.json"
        self._ensure_index()
    
    def _ensure_index(self):
        """Ensure archive index exists"""
        if not self.index_file.exists():
            self._save_index({})
    
    def _load_index(self) -> dict:
        """Load archive index"""
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_index(self, index: dict):
        """Save archive index"""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    
    def get_quarter(self, date: datetime = None) -> str:
        """Get quarter string (e.g., '2025_Q1')"""
        if date is None:
            date = datetime.now()
        quarter = (date.month - 1) // 3 + 1
        return f"{date.year}_Q{quarter}"
    
    def get_month(self, date: datetime = None) -> str:
        """Get month string (e.g., '2025_04')"""
        if date is None:
            date = datetime.now()
        return f"{date.year}_{date.month:02d}"
    
    def compress_file(self, file_path: str, archive_name: str) -> Optional[str]:
        """Compress a single file with maximum compression"""
        if not os.path.exists(file_path):
            return None
        
        archive_path = self.archive_dir / f"{archive_name}.zip"
        
        try:
            with zipfile.ZipFile(archive_path, 'w', 
                               compression=zipfile.ZIP_DEFLATED,
                               compresslevel=9) as zf:
                zf.write(file_path, arcname=os.path.basename(file_path))
            
            original_size = os.path.getsize(file_path)
            compressed_size = os.path.getsize(archive_path)
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            print(f"[ARCHIVE] Compressed {archive_name}: {original_size/1024:.1f}KB -> {compressed_size/1024:.1f}KB ({ratio:.1f}% saved)")
            return str(archive_path)
        except Exception as e:
            print(f"[ERROR] Failed to compress {file_path}: {e}")
            return None
    
    def compress_directory(self, dir_path: str, archive_name: str) -> Optional[str]:
        """Compress a directory with maximum compression"""
        if not os.path.exists(dir_path):
            return None
        
        archive_path = self.archive_dir / f"{archive_name}.zip"
        
        try:
            with zipfile.ZipFile(archive_path, 'w',
                               compression=zipfile.ZIP_DEFLATED,
                               compresslevel=9) as zf:
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(dir_path))
                        zf.write(file_path, arcname)
            
            original_size = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(dir_path)
                for f in files
            )
            compressed_size = os.path.getsize(archive_path)
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            print(f"[ARCHIVE] Compressed {archive_name}: {original_size/1024/1024:.1f}MB -> {compressed_size/1024/1024:.1f}MB ({ratio:.1f}% saved)")
            return str(archive_path)
        except Exception as e:
            print(f"[ERROR] Failed to compress directory {dir_path}: {e}")
            return None
    
    def archive_gold_history(self) -> Optional[str]:
        """Archive gold history CSV"""
        csv_path = "data/gold_history.csv"
        if not os.path.exists(csv_path):
            return None
        
        quarter = self.get_quarter()
        archive_name = f"gold_history_{quarter}"
        return self.compress_file(csv_path, archive_name)
    
    def archive_news(self) -> Optional[str]:
        """Archive news folder"""
        news_path = "data/news"
        if not os.path.exists(news_path):
            return None
        
        quarter = self.get_quarter()
        archive_name = f"news_{quarter}"
        return self.compress_directory(news_path, archive_name)
    
    def archive_trading_decisions(self) -> Optional[str]:
        """Archive trading decision files"""
        decisions = []
        for filename in ["trading_decision.json", "market_analysis.json"]:
            if os.path.exists(f"data/{filename}"):
                decisions.append(f"data/{filename}")
        
        if not decisions:
            return None
        
        quarter = self.get_quarter()
        archive_path = self.archive_dir / f"decisions_{quarter}.zip"
        
        try:
            with zipfile.ZipFile(archive_path, 'w',
                               compression=zipfile.ZIP_DEFLATED,
                               compresslevel=9) as zf:
                for f in decisions:
                    zf.write(f, arcname=os.path.basename(f))
            
            return str(archive_path)
        except Exception as e:
            print(f"[ERROR] Failed to archive decisions: {e}")
            return None
    
    def archive_risk_reports(self) -> Optional[str]:
        """Archive risk report files"""
        risk_path = "data/risk_report.json"
        if not os.path.exists(risk_path):
            return None
        
        quarter = self.get_quarter()
        return self.compress_file(risk_path, f"risk_report_{quarter}")
    
    def archive_logs(self) -> Optional[str]:
        """Archive log files"""
        log_path = "data/logs"
        if not os.path.exists(log_path):
            return None
        
        month = self.get_month()
        archive_name = f"logs_{month}"
        return self.compress_directory(log_path, archive_name)
    
    def archive_all(self) -> Dict[str, Optional[str]]:
        """Run full archival process"""
        results = {
            "gold_history": self.archive_gold_history(),
            "news": self.archive_news(),
            "trading_decisions": self.archive_trading_decisions(),
            "risk_reports": self.archive_risk_reports(),
            "logs": self.archive_logs()
        }
        
        # Update index
        index = self._load_index()
        for key, path in results.items():
            if path:
                quarter = self.get_quarter()
                if quarter not in index:
                    index[quarter] = {}
                index[quarter][key] = {
                    "path": path,
                    "archived_at": datetime.now().isoformat(),
                    "size_kb": os.path.getsize(path) / 1024
                }
        self._save_index(index)
        
        return results
    
    def get_archive_info(self) -> List[Dict]:
        """Get information about all archives"""
        index = self._load_index()
        archives = []
        
        for quarter, items in index.items():
            for key, info in items.items():
                archives.append({
                    "quarter": quarter,
                    "type": key,
                    "path": info.get("path"),
                    "size_kb": info.get("size_kb", 0),
                    "archived_at": info.get("archived_at")
                })
        
        return sorted(archives, key=lambda x: x["archived_at"], reverse=True)
    
    def get_total_size(self) -> float:
        """Get total size of all archives in MB"""
        total = 0
        for root, _, files in os.walk(self.archive_dir):
            for f in files:
                if f.endswith('.zip'):
                    total += os.path.getsize(os.path.join(root, f))
        return total / (1024 * 1024)


# Global archiver instance
_archiver: Optional[AutoArchiver] = None


def get_archiver(archive_dir: str = "data/archive") -> AutoArchiver:
    """Get or create global archiver instance"""
    global _archiver
    if _archiver is None:
        _archiver = AutoArchiver(archive_dir)
    return _archiver