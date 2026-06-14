"""Centralized path resolution.

Resolves project root, data directory, and third-party submodule paths
via environment variables with sensible defaults.  All other modules
import from here instead of crawling ``os.path.dirname(__file__)``.
"""

import os
from pathlib import Path


def _find_project_root() -> Path:
    """Locate the project root directory.

    Priority:
      1. ``STEREO_HOI_ROOT`` environment variable
      2. Walk up from this file until a ``pyproject.toml`` is found
      3. Fall back to the parent of the ``src`` directory
    """
    if env := os.environ.get("STEREO_HOI_ROOT"):
        return Path(env)

    candidate = Path(__file__).resolve().parent
    for _ in range(6):
        if (candidate / "pyproject.toml").exists():
            return candidate
        candidate = candidate.parent

    # Fallback: two levels above this file (src/stereo_hoi/_pathresolver.py)
    return Path(__file__).resolve().parents[2]


class PathResolver:
    """Lazy resolver for project paths."""

    def __init__(self, project_root: str | Path | None = None):
        self._root = Path(project_root) if project_root else _find_project_root()

    # -- root ----------------------------------------------------------------

    @property
    def project_root(self) -> Path:
        """Repository root (contains pyproject.toml)."""
        return self._root

    # -- data ----------------------------------------------------------------

    @property
    def data_root(self) -> Path:
        """Top-level data directory."""
        if env := os.environ.get("STEREO_HOI_DATA"):
            return Path(env)
        return self._root / "data"

    def clip_dir(self, clip: str) -> Path:
        """``data/<clip>/`` directory."""
        return self.data_root / clip

    # -- submodules ----------------------------------------------------------

    @property
    def ffs_dir(self) -> Path:
        """Fast-FoundationStereo checkout."""
        if env := os.environ.get("STEREO_HOI_FFS_DIR"):
            return Path(env)
        return self._root.parent / "Fast-FoundationStereo"

    @property
    def foundationpose_dir(self) -> Path:
        """FoundationPose checkout."""
        if env := os.environ.get("STEREO_HOI_FP_DIR"):
            return Path(env)
        return self._root.parent / "FoundationPose"

    @property
    def wilor_dir(self) -> Path:
        """WiLoR checkout."""
        if env := os.environ.get("STEREO_HOI_WILOR_DIR"):
            return Path(env)
        return self._root.parent / "WiLoR"

    @property
    def web_demo_dir(self) -> Path:
        """web_demo static assets directory."""
        return self._root / "web_demo"


# Module-level singleton — import this throughout the package.
paths = PathResolver()
