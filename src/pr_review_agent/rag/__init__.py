"""RAG retrieval: AST-aware chunking, local embeddings, hybrid search.

Lives alongside the deterministic AST resolver in ``context/``, which stays the
baseline. Nothing here imports the optional heavy deps (lancedb,
sentence-transformers) at module load; those are imported lazily where used so
``import pr_review_agent.rag`` works without the ``rag`` extra installed.
"""
