import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np

from chunker import enrich, iter_chunks, tok
from corpus_loader import load_corpus
from models import RetrievedContext

logger = logging.getLogger(__name__)

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
NEBIUS_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
NEBIUS_BASE_URL_DEFAULT = "https://api.studio.nebius.com/v1/"
MAX_BATCH_TOKENS = 250_000
RRF_K = 60            # standard RRF constant
GLOBAL_CANDIDATE_K = 50   # how many each of FAISS global and BM25 global contribute
ERA_CANDIDATE_K = 200     # how many year-era FAISS contributes (already filtered to era)


def _detect_embedding_provider() -> str:
    """Determine which embedding provider to use based on env vars.

    OpenRouter's `qwen/qwen3-embedding-8b` is routed to a Nebius backend
    so the vectors are identical to the baked index. When OPENROUTER_API_KEY
    is set we still report "nebius" so the index subdirectory lookup
    points at the same chunks/vectors; `_embed_query` will route the
    actual HTTP call through OpenRouter.
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    if provider == "nebius" and os.environ.get("NEBIUS_API_KEY"):
        return "nebius"
    if provider == "gemini" and os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if provider in ("openai", "anthropic"):
        return "openai"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "nebius"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("NEBIUS_API_KEY"):
        return "nebius"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return "openai"  # fallback


class FaissRetriever:
    def __init__(self, corpus_dir: Path, index_dir: Path, top_k: int = 5):
        self._top_k  = top_k
        self._index  = None
        self._chunks: list[dict] = []
        self._bm25   = None
        self._bm25_type = None  # "bm25s" or "rank_bm25"
        self._bm25_stemmer = None
        self._chunk_bulletin_year: list[int | None] = []
        self._year_content_index: dict[int, list[int]] = {}

        self._embedding_provider = _detect_embedding_provider()
        logger.info("Embedding provider: %s", self._embedding_provider)

        # Use provider-specific subdirectory for index files
        provider_dir = index_dir / self._embedding_provider
        provider_dir.mkdir(parents=True, exist_ok=True)

        index_path    = provider_dir / "index.faiss"
        meta_path     = provider_dir / "chunks.json"
        bm25_path     = provider_dir / "bm25.pkl"

        # Check for existing chunks from ANY provider or root dir (for BM25 reuse)
        any_meta = None
        # Check root index dir first
        if (index_dir / "chunks.json").exists():
            any_meta = index_dir / "chunks.json"
        else:
            for alt in index_dir.iterdir():
                if alt.is_dir() and (alt / "chunks.json").exists():
                    any_meta = alt / "chunks.json"
                    break

        if index_path.exists() and meta_path.exists():
            self._load(index_path, meta_path)
        elif meta_path.exists():
            # chunks.json exists but no FAISS index — BM25-only mode
            logger.info("No FAISS index, loading chunks from %s for BM25-only retrieval", meta_path)
            with meta_path.open("r", encoding="utf-8") as f:
                self._chunks = json.load(f)
            logger.info("Loaded %d chunks (BM25-only mode, no FAISS semantic search)", len(self._chunks))
        elif any_meta:
            # Reuse chunks from another provider's index for BM25
            logger.info("No FAISS index for %s, loading chunks from %s for BM25-only retrieval",
                        self._embedding_provider, any_meta)
            with any_meta.open("r", encoding="utf-8") as f:
                self._chunks = json.load(f)
            logger.info("Loaded %d chunks (BM25-only mode, no FAISS semantic search)", len(self._chunks))
        else:
            self._build(corpus_dir, provider_dir, index_path, meta_path)

        # Reuse BM25 from root dir if not in provider dir
        if not bm25_path.exists() and (index_dir / "bm25.pkl").exists():
            bm25_path = index_dir / "bm25.pkl"

        bm25s_dir = index_dir / "bm25s"
        self._load_bm25(bm25s_dir, bm25_path)

        year_idx_path = index_dir / "year_index.json"
        self._build_year_index(year_idx_path)

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _load_bm25(self, bm25s_dir: Path, bm25_pkl_path: Path) -> None:
        if bm25s_dir.exists():
            try:
                import bm25s
                import Stemmer
                self._bm25 = bm25s.BM25.load(str(bm25s_dir))
                self._bm25_stemmer = Stemmer.Stemmer("english")
                self._bm25_type = "bm25s"
                logger.info("Loaded bm25s index from %s", bm25s_dir)
                return
            except ImportError:
                logger.warning("bm25s/PyStemmer not installed; trying rank-bm25")
        if bm25_pkl_path.exists():
            try:
                from rank_bm25 import BM25Okapi
                import pickle
                with bm25_pkl_path.open("rb") as f:
                    self._bm25 = pickle.load(f)
                self._bm25_type = "rank_bm25"
                logger.info("Loaded rank-bm25 index from %s", bm25_pkl_path)
                return
            except ImportError:
                logger.warning("rank-bm25 not installed; BM25 disabled")
        if self._chunks:
            self._build_bm25_fallback(bm25_pkl_path)

    def _build_bm25_fallback(self, bm25_path: Path) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("No BM25 library available; BM25 disabled")
            return
        import pickle
        logger.info("Building BM25 index over %d chunks", len(self._chunks))
        corpus = [self._tokenize(c["content"]) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)
        self._bm25_type = "rank_bm25"
        with bm25_path.open("wb") as f:
            pickle.dump(self._bm25, f)
        logger.info("Built and saved BM25 index to %s", bm25_path)

    # ── load ──────────────────────────────────────────────────────────────────

    def _load(self, index_path: Path, meta_path: Path) -> None:
        import faiss

        logger.info("Loading FAISS index from %s", index_path)
        self._index = faiss.read_index(str(index_path))

        with meta_path.open("r", encoding="utf-8") as f:
            self._chunks = json.load(f)
        logger.info("Loaded FAISS index with %d chunks (dim=%d, provider=%s)",
                     len(self._chunks), self._index.d, self._embedding_provider)

    # ── build ─────────────────────────────────────────────────────────────────

    def _build(
        self,
        corpus_dir: Path,
        index_dir: Path,
        index_path: Path,
        meta_path: Path,
    ) -> None:
        import faiss

        logger.info("Building FAISS index from corpus at %s (provider=%s)", corpus_dir, self._embedding_provider)
        documents = load_corpus(corpus_dir)

        all_chunks: list[dict] = []
        for doc in documents:
            for chunk in iter_chunks(doc.content):
                all_chunks.append({
                    "source":        str(doc.path),
                    "content":       enrich(chunk),
                    "table_summary": chunk.get("table_summary", ""),
                    "type":          chunk.get("type", "text"),
                    "section":       chunk.get("section", ""),
                })

        logger.info("Chunked corpus into %d chunks, embedding with %s...", len(all_chunks), self._embedding_provider)

        embeddings: list[list[float]] = []
        batches = self._build_batches(all_chunks)

        logger.info("Split into %d token-safe batches", len(batches))

        for i, batch in enumerate(batches):
            self._embed_batch_with_retry(batch, embeddings)
            logger.info("Embedded batch %d / %d  (%d chunks so far)",
                        i + 1, len(batches),
                        sum(len(b) for b in batches[:i+1]))

        dim    = len(embeddings[0])
        matrix = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)

        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(all_chunks, f)

        self._index  = index
        self._chunks = all_chunks
        logger.info("Built and saved FAISS index with %d chunks (dim=%d)", len(all_chunks), dim)

    # ── batch builder ─────────────────────────────────────────────────────────

    def _build_batches(self, all_chunks: list[dict]) -> list[list[str]]:
        """Split chunks into token-safe batches for the embedding API.
        Respects both the 300k token limit and the 2048 input limit per request.
        """
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for chunk in all_chunks:
            content       = chunk["content"]
            chunk_tokens  = tok(content)

            # flush if adding this chunk would exceed either limit
            if current_batch and (
                current_tokens + chunk_tokens > MAX_BATCH_TOKENS
                or len(current_batch) >= 2048
            ):
                batches.append(current_batch)
                current_batch  = []
                current_tokens = 0

            current_batch.append(content)
            current_tokens += chunk_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    # ── embed helpers ─────────────────────────────────────────────────────────

    def _embed_batch_with_retry(
        self,
        batch: list[str],
        embeddings: list,
    ) -> None:
        for attempt in range(10):
            try:
                if self._embedding_provider == "gemini":
                    self._embed_batch_gemini(batch, embeddings)
                elif self._embedding_provider == "nebius":
                    self._embed_batch_nebius(batch, embeddings)
                else:
                    self._embed_batch_openai(batch, embeddings)
                return
            except Exception as e:
                err_str = str(e).lower()
                if "429" in str(e) or "rate_limit" in err_str or "rate" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                    wait = 2 ** attempt * 5
                    logger.warning(
                        "Rate limit hit, retrying in %ds (attempt %d)",
                        wait, attempt + 1,
                    )
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Embedding failed after 10 attempts")

    def _embed_batch_openai(self, batch: list[str], embeddings: list) -> None:
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=batch)
        embeddings.extend([e.embedding for e in resp.data])

    def _embed_batch_gemini(self, batch: list[str], embeddings: list) -> None:
        from google import genai
        client = genai.Client()
        # Gemini: max 100 items per batch request, minimal delay to smooth rate
        SUB_BATCH = 100
        for i in range(0, len(batch), SUB_BATCH):
            sub = batch[i:i+SUB_BATCH]
            resp = client.models.embed_content(model=GEMINI_EMBEDDING_MODEL, contents=sub)
            embeddings.extend([e.values for e in resp.embeddings])
            if i + SUB_BATCH < len(batch):
                time.sleep(0.3)

    def _embed_batch_nebius(self, batch: list[str], embeddings: list) -> None:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["NEBIUS_API_KEY"],
            base_url=os.environ.get("NEBIUS_BASE_URL", NEBIUS_BASE_URL_DEFAULT),
        )
        SUB_BATCH = 32
        for i in range(0, len(batch), SUB_BATCH):
            sub = batch[i:i+SUB_BATCH]
            resp = client.embeddings.create(model=NEBIUS_EMBEDDING_MODEL, input=sub)
            embeddings.extend([e.embedding for e in resp.data])

    def _embed_query(self, question: str) -> np.ndarray:
        import faiss

        if self._embedding_provider == "gemini":
            from google import genai
            client = genai.Client()
            resp = client.models.embed_content(model=GEMINI_EMBEDDING_MODEL, contents=question)
            vec = np.array([resp.embeddings[0].values], dtype=np.float32)
        elif self._embedding_provider == "nebius":
            # Prefer OpenRouter so a single key covers both chat and
            # embeddings. OpenRouter's qwen/qwen3-embedding-8b is served
            # by Nebius backend so the vectors are identical to the
            # baked index. Fall back to a direct Nebius call when only
            # NEBIUS_API_KEY is set.
            from openai import OpenAI
            if os.environ.get("OPENROUTER_API_KEY"):
                client = OpenAI(
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url=os.environ.get(
                        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
                    ),
                )
                resp = client.embeddings.create(
                    model="qwen/qwen3-embedding-8b",
                    input=[question],
                )
            else:
                client = OpenAI(
                    api_key=os.environ["NEBIUS_API_KEY"],
                    base_url=os.environ.get("NEBIUS_BASE_URL", NEBIUS_BASE_URL_DEFAULT),
                )
                resp = client.embeddings.create(
                    model=NEBIUS_EMBEDDING_MODEL, input=[question],
                )
            vec = np.array([resp.data[0].embedding], dtype=np.float32)
        else:
            from openai import OpenAI
            client = OpenAI()
            resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=[question])
            vec = np.array([resp.data[0].embedding], dtype=np.float32)

        faiss.normalize_L2(vec)
        return vec

    # ── year-aware helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract_years(question: str) -> list[int]:
        return [int(y) for y in re.findall(r"(?<!\d)(1[89]\d{2}|20\d{2})(?!\d)", question)]

    def _build_year_index(self, year_idx_path: Path) -> None:
        year_pat = re.compile(r"treasury_bulletin_(\d{4})")

        self._chunk_bulletin_year = []
        for chunk in self._chunks:
            m = year_pat.search(chunk.get("source", ""))
            self._chunk_bulletin_year.append(int(m.group(1)) if m else None)

        if year_idx_path.exists():
            logger.info("Loading year content index from %s", year_idx_path)
            with year_idx_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            self._year_content_index = {int(k): v for k, v in raw.items()}
            logger.info("Loaded year index: %d distinct years", len(self._year_content_index))
            return

        logger.info("Building year content index over %d chunks", len(self._chunks))
        content_year_pat = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
        self._year_content_index = {}
        for i, chunk in enumerate(self._chunks):
            for year_str in set(content_year_pat.findall(chunk["content"])):
                self._year_content_index.setdefault(int(year_str), []).append(i)

        try:
            with year_idx_path.open("w", encoding="utf-8") as f:
                json.dump(self._year_content_index, f)
            logger.info(
                "Built and saved year index: %d distinct years to %s",
                len(self._year_content_index), year_idx_path,
            )
        except OSError:
            logger.info("Built year index in memory (%d distinct years, read-only filesystem)", len(self._year_content_index))

    def _year_era_indices(self, years: list[int], before: int = 1, after: int = 8) -> list[int]:
        if not years:
            return []
        return [
            i for i, by in enumerate(self._chunk_bulletin_year)
            if by is not None and any(year - before <= by <= year + after for year in years)
        ]

    def _year_content_indices(self, years: list[int], era_set: set[int]) -> list[int]:
        result: set[int] = set()
        for year in years:
            result.update(i for i in self._year_content_index.get(year, []) if i in era_set)
        return list(result)

    # ── retrieve ──────────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        faiss_ranks: dict[int, int],
        bm25_ranks: dict[int, int],
        faiss_w: float = 0.5,
        bm25_w: float = 0.5,
    ) -> list[tuple[int, float]]:
        """Combine FAISS and BM25 ranks via weighted RRF. Returns (idx, score) sorted descending."""
        all_idx = set(faiss_ranks) | set(bm25_ranks)
        rrf_scores: dict[int, float] = {}
        for idx in all_idx:
            faiss_rrf = faiss_w / (RRF_K + faiss_ranks[idx] + 1) if idx in faiss_ranks else 0.0
            bm25_rrf  = bm25_w  / (RRF_K + bm25_ranks[idx] + 1) if idx in bm25_ranks else 0.0
            rrf_scores[idx] = faiss_rrf + bm25_rrf
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def _bm25_top_k(self, question: str, k: int) -> dict[int, int]:
        """Return {chunk_index: rank} for the top-k BM25 matches."""
        if self._bm25 is None:
            return {}
        if self._bm25_type == "bm25s":
            import bm25s
            query_tokens = bm25s.tokenize([question], stopwords="en", stemmer=self._bm25_stemmer, show_progress=False)
            results, _scores = self._bm25.retrieve(query_tokens, k=k)
            return {int(idx): rank for rank, idx in enumerate(results[0]) if idx >= 0}
        tokens = self._tokenize(question)
        scores = self._bm25.get_scores(tokens)
        top = np.argsort(scores)[::-1][:k]
        return {int(idx): rank for rank, idx in enumerate(top)}

    def _bm25_scores_all(self, question: str) -> np.ndarray | None:
        """Return per-chunk BM25 scores (for candidate filtering in retrieve_by_source_files)."""
        if self._bm25 is None:
            return None
        if self._bm25_type == "bm25s":
            import bm25s
            query_tokens = bm25s.tokenize([question], stopwords="en", stemmer=self._bm25_stemmer, show_progress=False)
            results, scores = self._bm25.retrieve(query_tokens, k=len(self._chunks))
            out = np.zeros(len(self._chunks))
            for idx, score in zip(results[0], scores[0]):
                if idx >= 0:
                    out[int(idx)] = score
            return out
        tokens = self._tokenize(question)
        return self._bm25.get_scores(tokens)

    def retrieve_by_source_files(
        self,
        source_files: list[str],
        question: str,
        query_vec: "np.ndarray | None" = None,
    ) -> list[RetrievedContext]:
        if not self._chunks:
            return []
        names = {Path(f).name for f in source_files}
        candidate_indices = [i for i, c in enumerate(self._chunks) if Path(c["source"]).name in names]
        if not candidate_indices:
            return []

        # FAISS ranks within candidate set (skip if no FAISS index)
        faiss_ranks: dict[int, int] = {}
        if self._index is not None:
            if query_vec is None:
                query_vec = self._embed_query(question)
            candidate_matrix = np.array(
                [self._index.reconstruct(i) for i in candidate_indices], dtype=np.float32
            )
            faiss_scores = (candidate_matrix @ query_vec.T).flatten()
            faiss_order = np.argsort(faiss_scores)[::-1][:GLOBAL_CANDIDATE_K]
            faiss_ranks = {candidate_indices[int(i)]: rank for rank, i in enumerate(faiss_order)}

        # BM25 ranks within candidate set
        bm25_ranks: dict[int, int] = {}
        all_bm25_scores = self._bm25_scores_all(question)
        if all_bm25_scores is not None:
            candidate_bm25 = sorted(
                ((i, all_bm25_scores[i]) for i in candidate_indices),
                key=lambda x: x[1],
                reverse=True,
            )
            bm25_ranks = {i: rank for rank, (i, _) in enumerate(candidate_bm25[:GLOBAL_CANDIDATE_K])}

        if faiss_ranks:
            top_rrf = self._rrf_fuse(faiss_ranks, bm25_ranks, 0.5, 0.5)[:self._top_k]
        else:
            # BM25-only mode
            top_rrf = sorted(bm25_ranks.items(), key=lambda x: x[1])[:self._top_k]
            top_rrf = [(idx, 1.0 / (RRF_K + rank + 1)) for idx, rank in top_rrf]
        return [self._make_result(self._chunks[idx], score) for idx, score in top_rrf]

    def retrieve(
        self,
        question: str,
        query_vec: "np.ndarray | None" = None,
    ) -> list[RetrievedContext]:
        if not self._chunks:
            return []

        # FAISS global (semantic) — top GLOBAL_CANDIDATE_K
        faiss_ranks: dict[int, int] = {}
        if self._index is not None:
            if query_vec is None:
                query_vec = self._embed_query(question)
            _, faiss_indices = self._index.search(query_vec, GLOBAL_CANDIDATE_K)
            faiss_ranks = {
                int(idx): rank
                for rank, idx in enumerate(faiss_indices[0])
                if idx >= 0
            }

        # BM25 global (keyword) — top GLOBAL_CANDIDATE_K
        bm25_ranks = self._bm25_top_k(question, GLOBAL_CANDIDATE_K)

        # Year-era FAISS passes (only when FAISS available)
        detected_years = self._extract_years(question)
        year_era_ranks: dict[int, int] = {}
        year_near_ranks: dict[int, int] = {}

        if detected_years and self._index is not None:
            era_indices  = self._year_era_indices(detected_years, before=1, after=8)
            near_indices = self._year_era_indices(detected_years, before=0, after=5)
            logger.info(
                "Year-era: detected years=%s, wide=%d near=%d",
                detected_years, len(era_indices), len(near_indices),
            )

            def _era_search(pool: list[int]) -> dict[int, int]:
                if not pool:
                    return {}
                content_pool = self._year_content_indices(detected_years, set(pool))
                search = content_pool if content_pool else pool
                mat = np.array([self._index.reconstruct(i) for i in search], dtype=np.float32)
                scores = (mat @ query_vec.T).flatten()
                order = np.argsort(scores)[::-1][:ERA_CANDIDATE_K]
                return {search[int(i)]: rank for rank, i in enumerate(order)}

            year_era_ranks  = _era_search(era_indices)
            year_near_ranks = _era_search(near_indices)
            logger.info(
                "Year-era ranks: wide=%d near=%d",
                len(year_era_ranks), len(year_near_ranks),
            )
        elif detected_years and self._bm25 is not None:
            era_set = set(self._year_era_indices(detected_years, before=1, after=8))
            content_set = set()
            for yr in detected_years:
                content_set.update(self._year_content_index.get(yr, []))
            for idx in list(bm25_ranks.keys()):
                if idx in content_set and idx in era_set:
                    bm25_ranks[idx] = max(0, bm25_ranks[idx] - 10)

        # RRF fusion
        if self._index is not None:
            all_idx = set(faiss_ranks) | set(bm25_ranks) | set(year_era_ranks) | set(year_near_ranks)
            rrf_scores: dict[int, float] = {}
            has_era  = bool(year_era_ranks)
            has_near = bool(year_near_ranks)
            fallback_rank = max(GLOBAL_CANDIDATE_K, ERA_CANDIDATE_K)
            for idx in all_idx:
                score = 0.0
                if has_era and has_near:
                    score = (
                        0.25 / (RRF_K + faiss_ranks.get(idx, fallback_rank) + 1)
                        + 0.10 / (RRF_K + bm25_ranks.get(idx, fallback_rank) + 1)
                        + 0.25 / (RRF_K + year_era_ranks.get(idx, fallback_rank) + 1)
                        + 0.40 / (RRF_K + year_near_ranks.get(idx, fallback_rank) + 1)
                    )
                elif has_era:
                    score = (
                        0.40 / (RRF_K + faiss_ranks.get(idx, fallback_rank) + 1)
                        + 0.20 / (RRF_K + bm25_ranks.get(idx, fallback_rank) + 1)
                        + 0.40 / (RRF_K + year_era_ranks.get(idx, fallback_rank) + 1)
                    )
                else:
                    score = (
                        0.50 / (RRF_K + faiss_ranks.get(idx, fallback_rank) + 1)
                        + 0.50 / (RRF_K + bm25_ranks.get(idx, fallback_rank) + 1)
                    )
                rrf_scores[idx] = score
            top_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:self._top_k]
        else:
            # BM25-only mode
            top_rrf = sorted(bm25_ranks.items(), key=lambda x: x[1])[:self._top_k]
            top_rrf = [(idx, 1.0 / (RRF_K + rank + 1)) for idx, rank in top_rrf]

        return [self._make_result(self._chunks[idx], score) for idx, score in top_rrf]

    def _make_result(self, chunk: dict, score: float) -> RetrievedContext:
        content = chunk["content"]
        if chunk.get("type") == "table" and chunk.get("table_summary"):
            content = f"[Table Summary] {chunk['table_summary']}\n\n{content}"
        return RetrievedContext(source=chunk["source"], content=content[:8000], score=score)