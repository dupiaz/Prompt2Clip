import json
import hashlib
from pathlib import Path
from typing import Any, Optional
import numpy as np
from src.config.settings import AppSettings

class CacheManager:
    """
    Centralized cache manager that replaces scattered cache logic.
    Supports both JSON and numpy (.npz) cache formats.
    """
    
    def __init__(self, namespace: str):
        settings = AppSettings()
        # Ensure we're at the project root for .cache
        self.cache_dir = settings.project_root / settings.cache_dir / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, key: str, prefix: str = "", ext: str = "json") -> Path:
        """Generate cache file path from namespace and key."""
        cache_hash = hashlib.md5(key.encode()).hexdigest()
        filename = f"{prefix}_{cache_hash}.{ext}" if prefix else f"{cache_hash}.{ext}"
        return self.cache_dir / filename
    
    def is_cached(self, key: str, prefix: str = "", ext: str = "json") -> bool:
        """Check if cache exists for given key."""
        return self.get_cache_path(key, prefix, ext).exists()
    
    def load_json(self, key: str, prefix: str = "") -> Optional[Any]:
        """Load JSON cache. Returns None if not found."""
        path = self.get_cache_path(key, prefix, "json")
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def save_json(self, key: str, data: Any, prefix: str = "") -> Path:
        """Save data as JSON cache. Returns cache path."""
        path = self.get_cache_path(key, prefix, "json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path
    
    def load_numpy(self, key: str, prefix: str = "") -> Optional[dict]:
        """Load numpy cache. Returns dict of arrays or None."""
        path = self.get_cache_path(key, prefix, "npz")
        if path.exists():
            return dict(np.load(path))
        return None
    
    def save_numpy(self, key: str, prefix: str = "", **arrays) -> Path:
        """Save numpy arrays to cache. Returns cache path."""
        path = self.get_cache_path(key, prefix, "npz")
        np.savez_compressed(path, **arrays)
        return path
