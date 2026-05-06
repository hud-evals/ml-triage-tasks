"""
nmoe.data - Data loading, preprocessing, and mixture utilities.

Primary exports:
- build_loader: Main entry point for creating data loaders
- DeterministicLoader: Production loader with SWRR mixing
- MixturePlan: Data structure for mixture configuration

Preprocessing:
- PrepPipeline: Local preprocessing pipeline
- ParallelPrepPipeline: Multiprocess preprocessing
- PrepConfig: Pipeline configuration
- HuggingFaceSource: Stream from HuggingFace Hub

Index/Shard utilities:
- IndexReader, IndexWriter: Document boundary indices
- ShardedWriter: Write .npy + .idx shards
- ManifestInfo: Dataset manifest
"""
from __future__ import annotations

import importlib
from typing import Any


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Loader
    "build_loader": (".loader", "build_loader"),
    "DeterministicLoader": (".loader", "DeterministicLoader"),
    # Mixture
    "MixturePlan": (".mixture", "MixturePlan"),
    "StagePlan": (".mixture", "StagePlan"),
    "SourcePlan": (".mixture", "SourcePlan"),
    "HFSource": (".mixture", "HFSource"),
    "resolve_plan": (".mixture", "resolve_plan"),
    "populate_paths": (".mixture", "populate_paths"),
    # Dataset
    "NumpyFSLDataset": (".dataset", "NumpyFSLDataset"),
    "Cursor": (".dataset", "Cursor"),
    # Index
    "IndexReader": (".index", "IndexReader"),
    "IndexWriter": (".index", "IndexWriter"),
    "regenerate_index_from_shard": (".index", "regenerate_index_from_shard"),
    # Sources
    "DataSource": (".sources", "DataSource"),
    "Document": (".sources", "Document"),
    "HuggingFaceSource": (".sources", "HuggingFaceSource"),
    "HfFileSystemSource": (".sources", "HfFileSystemSource"),
    "JSONLSource": (".sources", "JSONLSource"),
    "TextFileSource": (".sources", "TextFileSource"),
    "ArrowSource": (".sources", "ArrowSource"),
    "create_source": (".sources", "create_source"),
    # Sinks
    "ShardWriter": (".sinks", "ShardWriter"),
    "ShardedWriter": (".sinks", "ShardedWriter"),
    "ShardInfo": (".sinks", "ShardInfo"),
    "ManifestInfo": (".sinks", "ManifestInfo"),
    "load_manifest": (".sinks", "load_manifest"),
    "verify_manifest": (".sinks", "verify_manifest"),
    # Preprocessing
    "PrepConfig": (".prep", "PrepConfig"),
    "PrepPipeline": (".prep", "PrepPipeline"),
    "ParallelPrepPipeline": (".prep", "ParallelPrepPipeline"),
    # Transforms
    "normalize_text": (".transforms", "normalize_text"),
    "tokenize": (".transforms", "tokenize"),
    "pack_document": (".transforms", "pack_document"),
    "get_shard_id": (".transforms", "get_shard_id"),
    "make_file_name": (".transforms", "make_file_name"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    module = importlib.import_module(module_name, __name__)
    return getattr(module, attr)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_EXPORTS.keys()))

__all__ = [
    # Loader
    "build_loader",
    "DeterministicLoader",
    # Mixture
    "MixturePlan",
    "resolve_plan",
    "populate_paths",
    # Dataset
    "NumpyFSLDataset",
    "Cursor",
    # Index
    "IndexReader",
    "IndexWriter",
    "regenerate_index_from_shard",
    # Sources
    "DataSource",
    "Document",
    "HuggingFaceSource",
    "HfFileSystemSource",
    "JSONLSource",
    "TextFileSource",
    "ArrowSource",
    "create_source",
    # Sinks
    "ShardWriter",
    "ShardedWriter",
    "ShardInfo",
    "ManifestInfo",
    "load_manifest",
    "verify_manifest",
    # Preprocessing
    "PrepConfig",
    "PrepPipeline",
    "ParallelPrepPipeline",
    # Transforms
    "normalize_text",
    "tokenize",
    "pack_document",
    "get_shard_id",
    "make_file_name",
]
