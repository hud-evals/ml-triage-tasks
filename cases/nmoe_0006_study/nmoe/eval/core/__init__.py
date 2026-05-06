"""
nmoe.eval.core: Karpathy/DCLM-style CORE evaluation (bundle-based).

This is intentionally:
- deterministic (fixed shuffle + fewshot sampling),
- bundle-based (no HF drift),
- vocab-scalable (chunked scoring; no [T,V] materialization for o200k).
"""

