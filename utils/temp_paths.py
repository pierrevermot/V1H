from __future__ import annotations

import os
from pathlib import Path


_FALLBACK_TEMP_ROOT = Path("/lustre/fsn1/projects/rech/nab/udl61tt")


def _resolve_temp_root(*, subdir: str = "V1H_tmp") -> Path:
	base = os.environ.get("SCRATCH")
	root = Path(base).expanduser().resolve() if base else _FALLBACK_TEMP_ROOT
	if subdir:
		root = root / subdir
	root.mkdir(parents=True, exist_ok=True)
	return root