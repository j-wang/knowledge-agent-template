#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "rank-bm25", "scikit-learn"]
# ///
"""Thin shim: run the kb_engine search CLI against THIS directory's knowledge base.

During template development the engine is vendored in ./engine and loaded via
sys.path. A deployed agent replaces this shim's dependency block with the
published `kb-engine` package (git URL) and drops the sys.path line."""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("KB_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from kb_engine.search import main

main()
