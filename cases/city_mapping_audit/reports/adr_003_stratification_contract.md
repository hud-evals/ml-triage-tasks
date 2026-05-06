# ADR-003: Stratified-eval output contract

**Status:** proposed  ·  **Date:** 2025-11-05  ·  **Author:** Priya

## Context

PR #071 introduced a stratified-eval harness with one axis
(lexical_overlap). Follow-up axes (name_length, city_frequency) will
land in Q1. Before we write more axis code, we should lock the output
contract.

## Decision

Each stratified run emits one CSV under `runs/stratified/` with the
following columns in order:

    axis, bucket, method, n, top_1, top_2, top_3

Axes: {lexical_overlap, name_length, city_frequency}. Buckets are
axis-specific; `n` is the number of hotels in the bucket.

## Rationale

Flat schema, easy to pivot in pandas or a notebook. Avoids one-file-
per-bucket sprawl.

## Alternatives considered

- Nested JSON (one file per axis): rejected for pivot ergonomics.
- Parquet: rejected for simplicity; CSV is fine at this scale.
