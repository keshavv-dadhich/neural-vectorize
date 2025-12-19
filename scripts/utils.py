"""
Utility functions used across the pipeline.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import hashlib

def generate_svg_id(svg_path: Path, prefix: str = "icon") -> str:
    """
    Generate a unique, deterministic ID for an SVG file.
    
    Args:
        svg_path: Path to SVG file
        prefix: Prefix for the ID (default: "icon")
    
    Returns:
        Unique ID string like "icon_000123"
    """
    # Use file content hash for determinism
    content = svg_path.read_bytes()
    hash_digest = hashlib.md5(content).hexdigest()[:8]
    # Convert to integer for sequential numbering
    numeric_id = int(hash_digest, 16) % 1000000
    return f"{prefix}_{numeric_id:06d}"


def save_metadata(metadata: List[Dict[str, Any]], output_path: Path):
    """Save metadata as JSON."""
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to {output_path}")


def load_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    """Load metadata from JSON."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def save_id_list(ids: List[str], output_path: Path):
    """Save list of IDs to text file."""
    with open(output_path, 'w') as f:
        f.write('\n'.join(ids))
    print(f"✅ Saved {len(ids)} IDs to {output_path}")


def load_id_list(id_path: Path) -> List[str]:
    """Load list of IDs from text file."""
    return id_path.read_text().strip().split('\n')


class ProgressTracker:
    """Simple progress tracker for pipeline stages."""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.current = 0
        self.desc = desc
    
    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
        percentage = (self.current / self.total) * 100
        print(f"\r{self.desc}: {self.current}/{self.total} ({percentage:.1f}%)", end='', flush=True)
    
    def finish(self):
        """Mark as complete."""
        print(f"\n✅ {self.desc} complete!")


def validate_svg_file(svg_path: Path) -> bool:
    """
    Basic validation of SVG file.
    
    Returns:
        True if file appears to be a valid SVG
    """
    try:
        content = svg_path.read_text()
        # Basic checks
        if '<svg' not in content.lower():
            return False
        if '<path' not in content.lower():
            return False
        # Check for embedded raster images (we don't want these)
        if '<image' in content.lower():
            return False
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # Test ID generation
    test_path = Path("test.svg")
    test_path.write_text("<svg></svg>")
    print(f"Generated ID: {generate_svg_id(test_path)}")
    test_path.unlink()
