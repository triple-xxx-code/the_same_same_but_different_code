# pip install streamlit numpy scikit-learn rank-bm25 langchain langchain-community langchain-experimental diskcache pandas tqdm ollama pymorphy3
# ollama serve
# ollama pull nomic-embed-text-v2-moe:latest
# python -m streamlit run search_TXTMD_version3.py --server.port 1667

import streamlit as st
import numpy as np
import os
import glob
import shutil
import json
import hashlib
import pickle
import time
import re
import html
import csv
from typing import List, Tuple, Dict, Optional, Any, Set
from datetime import datetime
import subprocess
import requests

from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_core.documents import Document as LCDocument
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    class LCDocument:
        def __init__(self, page_content: str = "", metadata: Optional[Dict] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

try:
    from diskcache import Cache as DiskCache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False

try:
    import pymorphy3
    MORPHOLOGY_AVAILABLE = True
except ImportError:
    MORPHOLOGY_AVAILABLE = False

import ollama

# ==========================================
# 0. ГЛОБАЛЬНЫЕ КОНСТАНТЫ
# ==========================================

DEFAULT_EMBED_MODEL = "nomic-embed-text-v2-moe:latest"
DEFAULT_LLM_MODEL = "llama3.2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 5
DEFAULT_SEMANTIC_WEIGHT = 0.7
DEFAULT_CACHE_TTL_DAYS = 7
DEFAULT_MIN_RELEVANCE = 0.3

EMBEDDING_BATCH_SIZE = 16

DEFAULT_SOURCE_DIR = "/data/public/bioarticles/"
DEFAULT_SAVE_DIR = DEFAULT_SOURCE_DIR + "found"

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DIR = os.path.join(_BASE_DIR, "embed_cache")
DEFAULT_INDEX_CACHE_DIR = os.path.join(_BASE_DIR, "rag_index_cache")
DEFAULT_CHUNK_EXPORT_DIR = os.path.join(_BASE_DIR, "chunk_exports")   # NEW

SUPPORTED_EXTENSIONS = ("txt", "md", "markdown")

for d in [DEFAULT_CACHE_DIR, DEFAULT_INDEX_CACHE_DIR,
          DEFAULT_CHUNK_EXPORT_DIR, os.path.join(_BASE_DIR, "temp_files")]:
    os.makedirs(d, exist_ok=True)

WORD_RE = re.compile(r'[а-яА-Яa-zA-Z0-9\-]+')
_CHUNK_HEADER_RE = re.compile(r'^\[Файл:\s*([^\]]+)\]\s*\n?')


# ==========================================
# 1. ТЕМЫ
# ==========================================

def init_theme():
    if 'theme' not in st.session_state:
        st.session_state.theme = "dark"


def apply_theme(theme="dark"):
    dark_theme = """
    <style>
        :root {
            --primary-color: #FF8C00; --primary-dark: #CC7000; --primary-light: #FFA333;
            --bg-dark: #0E1117; --bg-darker: #0A0C10; --bg-card: #1E1E24;
            --text-light: #FAFAFA; --text-dim: #B0B0B0;
        }
        .stApp { background: linear-gradient(135deg, var(--bg-dark) 0%, var(--bg-darker) 100%); color: var(--text-light); }
        section[data-testid="stSidebar"] { background: linear-gradient(180deg, var(--bg-darker) 0%, #080A0D 100%); border-right: 2px solid var(--primary-dark); }
        h1, h2, h3 { color: var(--primary-light) !important; }
        .stButton > button { background: linear-gradient(45deg, var(--primary-dark), var(--primary-color)); color: white !important; border: none !important; border-radius: 8px; }
        .stTextInput > div > div > input { background-color: var(--bg-card); color: var(--text-light); border: 2px solid var(--primary-dark); border-radius: 6px; }
        mark { background: rgba(255,140,0,0.25); color: inherit; border-radius: 3px; padding: 0 2px; font-weight: 500; }
    </style>
    """
    light_theme = """
    <style>
        :root {
            --primary-color: #FF8C00; --primary-dark: #CC7000; --primary-light: #FFA333;
            --bg-light: #FFFFFF; --bg-lighter: #F5F7FA; --bg-card: #FFFFFF;
            --text-dark: #2C3E50;
        }
        .stApp { background: linear-gradient(135deg, var(--bg-light) 0%, var(--bg-lighter) 100%); color: var(--text-dark); }
        section[data-testid="stSidebar"] { background: linear-gradient(180deg, #F8F9FC 0%, #EDF2F7 100%); border-right: 2px solid var(--primary-light); }
        h1, h2, h3 { color: var(--primary-dark) !important; }
        .stButton > button { background: linear-gradient(45deg, var(--primary-color), var(--primary-light)); color: white !important; border: none !important; border-radius: 8px; }
        .stTextInput > div > div > input { background-color: var(--bg-card); color: var(--text-dark); border: 2px solid var(--primary-light); border-radius: 6px; }
        mark { background: rgba(255,140,0,0.15); color: inherit; border-radius: 3px; padding: 0 2px; font-weight: 500; }
    </style>
    """
    st.markdown(dark_theme if theme == "dark" else light_theme, unsafe_allow_html=True)


def theme_selector():
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌙 Тёмная", key="dark_theme_btn", use_container_width=True):
            st.session_state.theme = "dark"
            st.rerun()
    with col2:
        if st.button("☀️ Светлая", key="light_theme_btn", use_container_width=True):
            st.session_state.theme = "light"
            st.rerun()


# ==========================================
# 2. КЭШ ЭМБЕДДИНГОВ
# ==========================================

class EmbeddingCache:
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR, ttl_days: int = DEFAULT_CACHE_TTL_DAYS):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_days * 86400
        self.stats_file = os.path.join(cache_dir, "cache_stats.json")
        self.use_diskcache = DISKCACHE_AVAILABLE
        self.cache = DiskCache(cache_dir, size_limit=2 ** 30) if self.use_diskcache else None
        self.stats = {"hits": 0, "misses": 0, "size_bytes": 0}
        self._load_stats()

    def _generate_key(self, text: str, model_name: str) -> str:
        return f"emb_{hashlib.sha256(f'{model_name}:{text}'.encode('utf-8')).hexdigest()}"

    def _load_stats(self):
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
            except Exception:
                pass

    def _save_stats(self):
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2)
        except Exception:
            pass

    def _update_stats(self, hit: bool):
        self.stats["hits" if hit else "misses"] += 1
        self._save_stats()

    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        key = self._generate_key(text, model_name)
        if self.use_diskcache:
            result = self.cache.get(key)
            if result is not None:
                self._update_stats(hit=True)
                return result
        else:
            cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_path):
                try:
                    if time.time() - os.path.getmtime(cache_path) < self.ttl_seconds:
                        with open(cache_path, 'rb') as f:
                            result = pickle.load(f)
                        self._update_stats(hit=True)
                        return result
                    else:
                        os.remove(cache_path)
                except Exception:
                    pass
        self._update_stats(hit=False)
        return None

    def set(self, text: str, model_name: str, embedding: List[float]):
        if not embedding:
            return
        key = self._generate_key(text, model_name)
        if self.use_diskcache:
            self.cache.set(key, embedding, expire=self.ttl_seconds)
        else:
            try:
                with open(os.path.join(self.cache_dir, f"{key}.pkl"), 'wb') as f:
                    pickle.dump(embedding, f)
            except Exception:
                pass

    def clear(self):
        if self.use_diskcache:
            self.cache.clear()
        else:
            for f in glob.glob(os.path.join(self.cache_dir, "emb_*.pkl")):
                try:
                    os.remove(f)
                except Exception:
                    pass
        if os.path.exists(self.stats_file):
            try:
                os.remove(self.stats_file)
            except Exception:
                pass
        self.stats = {"hits": 0, "misses": 0, "size_bytes": 0}

    def get_stats(self) -> Dict:
        if self.use_diskcache:
            size = sum(os.path.getsize(os.path.join(self.cache_dir, f))
                       for f in os.listdir(self.cache_dir)
                       if os.path.isfile(os.path.join(self.cache_dir, f)))
        else:
            size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(self.cache_dir, "emb_*.pkl")))
        self.stats["size_bytes"] = size
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
        return {
            "hits": self.stats["hits"], "misses": self.stats["misses"],
            "hit_rate": hit_rate, "size_mb": self.stats["size_bytes"] / (1024 * 1024),
            "entries": len(self.cache) if self.use_diskcache else len(glob.glob(os.path.join(self.cache_dir, "emb_*.*")))
        }


# ==========================================
# 3. OLLAMA UTILS
# ==========================================

OLLAMA_BATCH_SUPPORTED = hasattr(ollama, "embed")


def _model_name(m) -> Optional[str]:
    if isinstance(m, dict):
        return m.get('name') or m.get('model')
    return getattr(m, 'name', None) or getattr(m, 'model', None)


def check_ollama_connection() -> Tuple[bool, str, List[str]]:
    try:
        response = ollama.list()
        raw_models = response.get('models', []) if isinstance(response, dict) else getattr(response, 'models', [])
        models = [n for m in raw_models if (n := _model_name(m))]
        return (True, "Ollama подключен", models) if models else (False, "Список моделей пуст", [])
    except Exception as e:
        return False, f"Ошибка: {str(e)}", []


def preload_ollama_model(model_name: str, keep_alive: int = -1) -> Tuple[bool, str]:
    try:
        ollama.generate(model=model_name, prompt="", keep_alive=keep_alive)
        return True, f"Модель '{model_name}' загружена в память"
    except Exception as e:
        return False, f"Ошибка предзагрузки: {e}"


def _ollama_embedding(text: str, model_name: str) -> List[float]:
    response = ollama.embeddings(model=model_name, prompt=text)
    embedding = response.get('embedding') or response.get('embeddings')
    if not embedding:
        raise RuntimeError(f"Ollama вернул пустой эмбеддинг для модели '{model_name}'")
    return embedding


def _ollama_embedding_batch(texts: List[str], model_name: str) -> List[List[float]]:
    if not texts:
        return []
    if not OLLAMA_BATCH_SUPPORTED:
        return [_ollama_embedding(t, model_name) for t in texts]
    try:
        response = ollama.embed(model=model_name, input=texts)
    except Exception:
        return [_ollama_embedding(t, model_name) for t in texts]
    embeddings = response.get('embeddings') if isinstance(response, dict) else getattr(response, 'embeddings', None)
    if not embeddings or len(embeddings) != len(texts):
        return [_ollama_embedding(t, model_name) for t in texts]
    return embeddings


def get_embedding_with_cache(text: str, model_name: str, cache: Optional[EmbeddingCache] = None) -> List[float]:
    if cache:
        cached = cache.get(text, model_name)
        if cached is not None:
            return cached
    embedding = _ollama_embedding(text, model_name)
    if cache:
        cache.set(text, model_name, embedding)
    return embedding


def get_ollama_embeddings_batched(texts: List[str], model_name: str,
                                   cache: Optional[EmbeddingCache] = None,
                                   batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
    if not texts:
        return np.empty((0, 0))
    embeddings_map: Dict[int, List[float]] = {}
    to_compute: List[Tuple[int, str]] = []
    for i, text in enumerate(texts):
        if cache and (emb := cache.get(text, model_name)) is not None:
            embeddings_map[i] = emb
        else:
            to_compute.append((i, text))
    if to_compute:
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        total = len(to_compute)
        for batch_start in range(0, total, batch_size):
            batch = to_compute[batch_start:batch_start + batch_size]
            batch_end = min(batch_start + batch_size, total)
            status_text.text(f"Эмбеддинги: {batch_end}/{total} (батч {len(batch)})")
            batch_texts = [t for _, t in batch]
            try:
                batch_embeddings = _ollama_embedding_batch(batch_texts, model_name)
            except Exception as e:
                progress_bar.empty(); status_text.empty()
                st.error(f"❌ Ошибка генерации эмбеддингов: {e}")
                raise
            for (idx, text), emb in zip(batch, batch_embeddings):
                if cache:
                    cache.set(text, model_name, emb)
                embeddings_map[idx] = emb
            progress_bar.progress(batch_end / total)
        progress_bar.empty(); status_text.empty()
    return np.array([embeddings_map[i] for i in range(len(texts))])


def get_ollama_query_embedding(query: str, model_name: str, cache: Optional[EmbeddingCache] = None) -> np.ndarray:
    return np.array(get_embedding_with_cache(query, model_name, cache))


# ==========================================
# 4. TXT/MD PROCESSING
# ==========================================

def extract_text_from_bytes(file_bytes: bytes) -> str:
    return file_bytes.decode('utf-8', errors='ignore')


def extract_text_from_path(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def strip_markdown_syntax(text: str) -> str:
    text = re.sub(r'^\s*---\s*\n.*?\n---\s*\n', '', text, flags=re.DOTALL)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'^\s*```[^\n]*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*```\s*$', '', text, flags=re.MULTILINE)
    return text


def chunk_text_simple(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                      overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    if chunk_size <= 0:
        chunk_size = DEFAULT_CHUNK_SIZE
    if overlap >= chunk_size or overlap < 0:
        overlap = 0
    chunks, start, step = [], 0, chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def chunk_text_langchain(docs: List[LCDocument], method: str, chunk_size: int, chunk_overlap: int,
                         model_name: str = DEFAULT_EMBED_MODEL, semantic_threshold: float = 90.0,
                         cache: Optional[EmbeddingCache] = None) -> List[LCDocument]:
    if not LANGCHAIN_AVAILABLE:
        return chunk_text_simple_fallback(docs, chunk_size, chunk_overlap)
    try:
        if method == "semantic":
            class CachedEmbedWrapper:
                def __init__(self, mn, c):
                    self.mn, self.c = mn, c
                def embed_documents(self, texts):
                    return [get_embedding_with_cache(t, self.mn, self.c) for t in texts]
                def embed_query(self, text):
                    return get_embedding_with_cache(text, self.mn, self.c)
            splitter = SemanticChunker(
                embeddings=CachedEmbedWrapper(model_name, cache),
                breakpoint_threshold_type='percentile',
                breakpoint_threshold_amount=semantic_threshold
            )
            return splitter.split_documents(docs)
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, is_separator_regex=True
            )
            return splitter.split_documents(docs)
    except Exception as e:
        st.warning(f"⚠️ Ошибка LangChain разбиения: {e}. Использую простое.")
        return chunk_text_simple_fallback(docs, chunk_size, chunk_overlap)


def chunk_text_simple_fallback(docs: List[LCDocument], chunk_size: int, chunk_overlap: int) -> List[LCDocument]:
    return [
        LCDocument(page_content=c, metadata=doc.metadata.copy())
        for doc in docs for c in chunk_text_simple(doc.page_content, chunk_size, chunk_overlap)
    ]


def scan_source_files(dir_path: str) -> List[str]:
    files: List[str] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(os.path.join(dir_path, "**", f"*.{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(dir_path, "**", f"*.{ext.upper()}"), recursive=True))
    return sorted(set(files))


# ==========================================
# 4b. ЭКСПОРТ ЧАНКОВ В TXT/CSV (NEW)
# ==========================================

def _split_chunk_header(chunk: str) -> Tuple[str, str]:
    """Возвращает (source_name, body), отрезая '[Файл: xxx]\\n', если есть."""
    m = _CHUNK_HEADER_RE.match(chunk)
    if m:
        return m.group(1).strip(), chunk[m.end():]
    return "unknown", chunk


def _sanitize_filename(name: str, max_len: int = 80) -> str:
    base, ext = os.path.splitext(name)
    base = re.sub(r'[^\w\-]+', '_', base).strip('_')[:max_len] or "file"
    return base + ext


def export_chunks_to_txt(
    chunks: List[str],
    target_dir: str,
    index_key: Optional[str] = None,
    mode: str = "single",
    include_header: bool = True,
) -> Tuple[int, str]:
    """
    Сохраняет чанки для промежуточного анализа.

    mode = "single"     → один .txt: все чанки с разделителями
    mode = "per_chunk"  → отдельный .txt на чанк, сгруппированный по источнику
    mode = "csv"        → CSV (chunk_id, source, chars, text) для pandas

    Возвращает (количество чанков, путь к файлу/папке).
    """
    os.makedirs(target_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    key_suffix = f"_{index_key}" if index_key else ""

    if mode == "single":
        out_path = os.path.join(target_dir, f"chunks_all{key_suffix}_{ts}.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            if include_header:
                f.write(f"# index_key: {index_key or '-'}\n")
                f.write(f"# total_chunks: {len(chunks)}\n")
                f.write(f"# created: {ts}\n")
                f.write("#" + "=" * 70 + "\n\n")
            for i, chunk in enumerate(chunks):
                source, body = _split_chunk_header(chunk)
                f.write(
                    f"=== CHUNK {i:05d} | source={source} | chars={len(body)} ===\n"
                    if include_header else f"=== CHUNK {i:05d} ===\n"
                )
                f.write(body)
                f.write("\n\n")
        return len(chunks), out_path

    if mode == "per_chunk":
        by_source: Dict[str, List[Tuple[int, str]]] = {}
        for i, chunk in enumerate(chunks):
            source, body = _split_chunk_header(chunk)
            by_source.setdefault(source, []).append((i, body))
        root_dir = os.path.join(target_dir, f"per_chunk{key_suffix}")
        for source, items in by_source.items():
            src_dir = os.path.join(root_dir, _sanitize_filename(source))
            os.makedirs(src_dir, exist_ok=True)
            for i, body in items:
                with open(os.path.join(src_dir, f"{i:05d}.txt"), 'w', encoding='utf-8') as f:
                    if include_header:
                        f.write(f"# chunk_index: {i}\n# source: {source}\n# chars: {len(body)}\n")
                        if index_key:
                            f.write(f"# index_key: {index_key}\n")
                        f.write("#" + "-" * 60 + "\n")
                    f.write(body)
        return len(chunks), root_dir

    if mode == "csv":
        out_path = os.path.join(target_dir, f"chunks{key_suffix}_{ts}.csv")
        with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(["chunk_id", "source", "chars", "text"])
            for i, chunk in enumerate(chunks):
                source, body = _split_chunk_header(chunk)
                w.writerow([i, source, len(body), body])
        return len(chunks), out_path

    raise ValueError(f"Неизвестный режим экспорта: {mode}")


# ==========================================
# 5. ПОИСК
# ==========================================

def tokenize_text(text: str) -> List[str]:
    text = re.sub(r'[^\w\sа-яёa-z0-9]', ' ', text.lower(), flags=re.IGNORECASE)
    tokens = [t for t in text.split() if len(t) > 2]
    return tokens or [t for t in text.split() if t]


def create_bm25_index(chunks: List[str]) -> BM25Okapi:
    return BM25Okapi([tokenize_text(c) for c in chunks])


def search_bm25(query: str, bm25: BM25Okapi, chunks: List[str], top_k: int = 5) -> List[Dict]:
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return []
    scores = bm25.get_scores(query_tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [
        {"text": chunks[i], "score": float(scores[i]), "method": "bm25"}
        for i in top_idx if scores[i] > 0
    ]


def normalize_scores(scores: List[float]) -> np.ndarray:
    if not scores:
        return np.array([])
    arr = np.array([s if s is not None else 0.0 for s in scores], dtype=float)
    if len(arr) == 0:
        return np.array([])
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.ones_like(arr)
    return np.clip((arr - mn) / (mx - mn + 1e-8), 0.0, 1.0)


def search_similar(query_emb, chunk_embs, chunks, top_k=5):
    if len(chunk_embs) == 0:
        return []
    sims = cosine_similarity([query_emb], chunk_embs)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [
        {"text": chunks[i], "score": float(sims[i]), "method": "semantic"}
        for i in top_idx if sims[i] > 0
    ]


def hybrid_search(query: str, query_emb: np.ndarray, chunk_embs: np.ndarray, chunks: List[str],
                  bm25: BM25Okapi, top_k: int = 5,
                  semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
                  mode: str = "hybrid") -> List[Dict]:
    if mode == "bm25":
        return search_bm25(query, bm25, chunks, top_k)
    if mode == "semantic":
        return search_similar(query_emb, chunk_embs, chunks, top_k)
    sem_res = search_similar(query_emb, chunk_embs, chunks, top_k * 2)
    bm25_res = search_bm25(query, bm25, chunks, top_k * 2)
    combined: Dict[str, Dict] = {}
    for r in sem_res:
        combined[r['text']] = {'text': r['text'], 'semantic': r['score'], 'bm25': 0.0}
    for r in bm25_res:
        if r['text'] in combined:
            combined[r['text']]['bm25'] = r['score']
        else:
            combined[r['text']] = {'text': r['text'], 'semantic': 0.0, 'bm25': r['score']}
    if not combined:
        return []
    items = list(combined.values())
    sem_scores = normalize_scores([x['semantic'] for x in items])
    bm25_scores = normalize_scores([x['bm25'] for x in items])
    hybrid_scores = semantic_weight * sem_scores + (1 - semantic_weight) * bm25_scores
    sorted_items = sorted(zip(items, hybrid_scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"text": it['text'], "score": float(sc),
         "semantic_score": float(it['semantic']), "bm25_score": float(it['bm25']),
         "method": "hybrid"}
        for it, sc in sorted_items if sc > 0
    ]


# ==========================================
# 5b. МОРФОЛОГИЧЕСКИЙ ПОИСК
# ==========================================

_LEMMA_CACHE: Dict[str, Optional[str]] = {}


@st.cache_resource
def get_morph():
    if not MORPHOLOGY_AVAILABLE:
        return None
    try:
        return pymorphy3.MorphAnalyzer()
    except Exception:
        return None


def _lemma(word: str) -> Optional[str]:
    w = word.lower()
    if w in _LEMMA_CACHE:
        return _LEMMA_CACHE[w]
    morph = get_morph()
    lemma = None
    if morph is not None:
        try:
            lemma = morph.parse(w)[0].normal_form
        except Exception:
            lemma = None
    _LEMMA_CACHE[w] = lemma
    return lemma


def should_use_morphology_search(query: str) -> bool:
    if not MORPHOLOGY_AVAILABLE:
        return False
    q = query.strip()
    if not q:
        return False
    tokens = q.split()
    return len(tokens) == 1 and bool(re.fullmatch(r'[а-яА-Яa-zA-Z\-]+', tokens[0]))


def build_chunk_lemmas(chunks: List[str]) -> List[Set[str]]:
    if not MORPHOLOGY_AVAILABLE:
        return []
    progress = st.progress(0.0)
    status = st.empty()
    total = len(chunks)
    result: List[Set[str]] = []
    for i, chunk in enumerate(chunks):
        lemmas: Set[str] = set()
        for w in WORD_RE.findall(chunk):
            lemma = _lemma(w)
            if lemma:
                lemmas.add(lemma)
        result.append(lemmas)
        if i % 200 == 0 or i == total - 1:
            progress.progress((i + 1) / total)
            status.text(f"Морфология: {i + 1}/{total}")
    progress.empty(); status.empty()
    return result


def search_morphology(chunks: List[str], chunk_lemmas: List[Set[str]],
                      query: str, top_k: int = 10) -> List[Dict]:
    if not chunk_lemmas or not chunks:
        return []
    query_lemmas: Set[str] = set()
    for w in WORD_RE.findall(query):
        lemma = _lemma(w)
        if lemma:
            query_lemmas.add(lemma)
    if not query_lemmas:
        return []
    results: List[Dict] = []
    for i, lemmas in enumerate(chunk_lemmas):
        matched = query_lemmas & lemmas
        if not matched:
            continue
        count = sum(1 for w in WORD_RE.findall(chunks[i]) if _lemma(w) in query_lemmas)
        if count == 0:
            continue
        score = min(1.0, 0.55 + 0.15 * (count - 1))
        results.append({
            "text": chunks[i], "score": float(score), "method": "morphology",
            "matched_lemmas": sorted(matched), "occurrences": count,
        })
    results.sort(key=lambda x: x["occurrences"], reverse=True)
    return results[:top_k]


# ==========================================
# 6. LLM GENERATION
# ==========================================

def _chunk_delta(chunk) -> str:
    try:
        msg = chunk["message"]
    except (KeyError, TypeError, IndexError):
        msg = getattr(chunk, "message", None)
    if msg is None:
        return ""
    try:
        content = msg["content"]
    except (KeyError, TypeError, IndexError):
        content = getattr(msg, "content", "")
    return content or ""


def stream_llm_answer(question: str, context: str, model: str,
                      temperature: float = 0.4, max_tokens: int = 3000,
                      system_prompt: str = "") -> str:
    sys_prompt = system_prompt or """Отвечай ТОЛЬКО на основе предоставленного контекста. 
Если информации недостаточно — скажи "В контексте нет ответа на этот вопрос". 
Исправляй грамматические ошибки для читаемости. Указывай источник информации в формате [Файл: имя_файла]."""
    formatted = f"Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
    placeholder = st.empty()
    full_response = ""
    try:
        stream = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": formatted}
            ],
            options={'temperature': temperature, 'num_predict': max_tokens},
            stream=True
        )
        for chunk in stream:
            delta = _chunk_delta(chunk)
            if delta:
                full_response += delta
                placeholder.markdown(full_response + "▌")
        if full_response:
            placeholder.markdown(full_response)
        else:
            placeholder.warning("⚠️ Модель вернула пустой ответ.")
        return full_response
    except Exception as e:
        if full_response:
            placeholder.markdown(full_response)
        st.error(f"❌ Ошибка генерации: {e}")
        return full_response


# ==========================================
# 7. FILE MANAGER
# ==========================================

class FileManager:
    @staticmethod
    def save_found_files(search_results: List[dict], file_paths: Dict[str, str],
                         source_dir: Optional[str] = None, target_dir: str = DEFAULT_SAVE_DIR,
                         uploaded_bytes: Optional[bytes] = None,
                         uploaded_name: Optional[str] = None) -> List[str]:
        saved, errors = [], []
        os.makedirs(target_dir, exist_ok=True)
        unique_files = set()
        for r in search_results:
            txt = r.get('text', '')
            if "[Файл:" in txt:
                try:
                    unique_files.add(txt.split("[Файл:")[1].split("]")[0].strip())
                except Exception:
                    pass
        for fname in unique_files:
            try:
                src = file_paths.get(fname)
                if src == "uploaded" or src is None:
                    if uploaded_bytes and fname == uploaded_name:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        base, ext = os.path.splitext(fname)
                        save_name = f"{base}_{ts}{ext}"
                        with open(os.path.join(target_dir, save_name), 'wb') as f:
                            f.write(uploaded_bytes)
                        saved.append(save_name)
                    else:
                        errors.append(f"Файл {fname} загружен, но байты недоступны")
                    continue
                if not src and source_dir:
                    src = os.path.join(source_dir, fname)
                if src and os.path.exists(src):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base, ext = os.path.splitext(fname)
                    save_name = f"{base}_{ts}{ext}"
                    shutil.copy2(src, os.path.join(target_dir, save_name))
                    saved.append(save_name)
                else:
                    errors.append(f"Файл не найден: {src}")
            except Exception as e:
                errors.append(f"Ошибка сохранения {fname}: {e}")
        if errors and not saved:
            with st.expander("🔍 Детали ошибок"):
                for err in errors:
                    st.markdown(f"- {err}")
        return saved


# ==========================================
# 8. SESSION STATE
# ==========================================

def init_session_state():
    defaults = {
        'theme': 'dark', 'load_mode': 'directory',
        'source_directory': DEFAULT_SOURCE_DIR,
        'uploaded_file_bytes': None, 'uploaded_file_name': None,
        'chunk_method': 'simple',
        'chunk_size': DEFAULT_CHUNK_SIZE,
        'chunk_overlap': DEFAULT_CHUNK_OVERLAP,
        'semantic_threshold': 90,
        'search_mode': 'hybrid',
        'semantic_weight': DEFAULT_SEMANTIC_WEIGHT,
        'top_k': DEFAULT_TOP_K, 'min_relevance': DEFAULT_MIN_RELEVANCE,
        'cache_enabled': True, 'cache_ttl_days': DEFAULT_CACHE_TTL_DAYS,
        'embedding_cache': None,
        'llm_model': DEFAULT_LLM_MODEL, 'llm_temperature': 0.4,
        'llm_max_tokens': 3000, 'system_prompt': '',
        'generate_answer': True, 'strip_md': True,
        'ollama_ok': False, 'ollama_msg': '', 'ollama_models': [],
        'processed': False, 'chunks': [], 'chunk_docs': [], 'embeddings': None,
        'bm25_index': None, 'chunk_lemmas': None,
        'file_paths': {}, 'processed_files': [], 'search_results': [],
        'last_query': '', 'last_search_time': None, 'last_answer_time': None,
        'max_chunks_limit': 8000, 'used_sig': None, 'index_disk_cache': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_index_state(preserve_source: bool = True):
    st.session_state.processed = False
    st.session_state.chunks = []
    st.session_state.chunk_docs = []
    st.session_state.embeddings = None
    st.session_state.bm25_index = None
    st.session_state.chunk_lemmas = None
    st.session_state.search_results = []
    st.session_state.used_sig = None
    if not preserve_source:
        st.session_state.file_paths = {}
        st.session_state.processed_files = []
        st.session_state.uploaded_file_bytes = None
        st.session_state.uploaded_file_name = None


# ==========================================
# 9. ПОДСВЕТКА
# ==========================================

def highlight_query_terms(text: str, query: str) -> str:
    safe_text = html.escape(text)
    raw_terms = [t for t in tokenize_text(query) if len(t) > 3]
    if not raw_terms:
        return safe_text
    pattern = r'(' + '|'.join(re.escape(html.escape(t)) for t in raw_terms) + r')'
    return re.sub(pattern, r'<mark>\1</mark>', safe_text, flags=re.IGNORECASE)


# ==========================================
# 10. ДИСКОВЫЙ КЭШ ИНДЕКСА
# ==========================================

def compute_source_manifest(load_mode: str, dir_path: str,
                             uploaded_name: Optional[str] = None,
                             uploaded_bytes: Optional[bytes] = None) -> List[Tuple[str, int, str]]:
    if load_mode == "file":
        if uploaded_name and uploaded_bytes:
            return [(uploaded_name, len(uploaded_bytes),
                     hashlib.sha256(uploaded_bytes).hexdigest()[:16])]
        return []
    if not dir_path or not os.path.isdir(dir_path):
        return []
    manifest = []
    for fp in scan_source_files(dir_path):
        try:
            st_info = os.stat(fp)
            manifest.append((os.path.relpath(fp, dir_path),
                             st_info.st_size, str(st_info.st_mtime_ns)))
        except OSError:
            continue
    return sorted(manifest)


def compute_index_key(config_sig: Tuple, source_manifest: List[Tuple],
                      embed_model: str) -> str:
    payload = {"config": list(config_sig), "sources": source_manifest,
               "embed_model": embed_model}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:24]


class IndexDiskCache:
    FILES = ("chunks.pkl", "chunk_docs.pkl", "embeddings.npy",
             "bm25.pkl", "file_paths.pkl", "meta.json")

    def __init__(self, cache_dir: str = DEFAULT_INDEX_CACHE_DIR, keep: int = 3):
        self.cache_dir = cache_dir
        self.keep = keep
        os.makedirs(cache_dir, exist_ok=True)

    def _dir(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"idx_{key}")

    def exists(self, key: str) -> bool:
        d = self._dir(key)
        return all(os.path.exists(os.path.join(d, f)) for f in self.FILES)

    def save(self, key, chunks, chunk_docs, embeddings, bm25_index,
             file_paths, meta, chunk_lemmas: Optional[List[Set[str]]] = None) -> None:
        d = self._dir(key)
        os.makedirs(d, exist_ok=True)
        try:
            with open(os.path.join(d, "chunks.pkl"), "wb") as f:
                pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
            docs_plain = [{"page_content": dc.page_content, "metadata": dc.metadata}
                          for dc in chunk_docs]
            with open(os.path.join(d, "chunk_docs.pkl"), "wb") as f:
                pickle.dump(docs_plain, f, protocol=pickle.HIGHEST_PROTOCOL)
            np.save(os.path.join(d, "embeddings.npy"), embeddings)
            with open(os.path.join(d, "bm25.pkl"), "wb") as f:
                pickle.dump(bm25_index, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(d, "file_paths.pkl"), "wb") as f:
                pickle.dump(file_paths, f, protocol=pickle.HIGHEST_PROTOCOL)
            if chunk_lemmas is not None:
                with open(os.path.join(d, "chunk_lemmas.pkl"), "wb") as f:
                    pickle.dump(chunk_lemmas, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.warning(f"⚠️ Не удалось сохранить индекс на диск: {e}")
            return
        self._cleanup_old(keep_key=key)

    def save_chunk_lemmas(self, key: str, chunk_lemmas: List[Set[str]]) -> None:
        d = self._dir(key)
        if not os.path.isdir(d):
            return
        try:
            with open(os.path.join(d, "chunk_lemmas.pkl"), "wb") as f:
                pickle.dump(chunk_lemmas, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    def load(self, key: str) -> Optional[Dict]:
        if not self.exists(key):
            return None
        d = self._dir(key)
        try:
            with open(os.path.join(d, "chunks.pkl"), "rb") as f:
                chunks = pickle.load(f)
            with open(os.path.join(d, "chunk_docs.pkl"), "rb") as f:
                docs_plain = pickle.load(f)
            chunk_docs = [LCDocument(page_content=x["page_content"], metadata=x["metadata"])
                          for x in docs_plain]
            embeddings = np.load(os.path.join(d, "embeddings.npy"))
            with open(os.path.join(d, "bm25.pkl"), "rb") as f:
                bm25_index = pickle.load(f)
            with open(os.path.join(d, "file_paths.pkl"), "rb") as f:
                file_paths = pickle.load(f)
            chunk_lemmas = None
            lemmas_path = os.path.join(d, "chunk_lemmas.pkl")
            if os.path.exists(lemmas_path):
                try:
                    with open(lemmas_path, "rb") as f:
                        chunk_lemmas = pickle.load(f)
                except Exception:
                    chunk_lemmas = None
            meta = {}
            meta_path = os.path.join(d, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            return {"chunks": chunks, "chunk_docs": chunk_docs,
                    "embeddings": embeddings, "bm25_index": bm25_index,
                    "file_paths": file_paths, "chunk_lemmas": chunk_lemmas,
                    "meta": meta}
        except Exception as e:
            st.warning(f"⚠️ Не удалось загрузить индекс из кэша: {e}")
            return None

    def _cleanup_old(self, keep_key: str) -> None:
        try:
            entries = []
            for name in os.listdir(self.cache_dir):
                p = os.path.join(self.cache_dir, name)
                if name.startswith("idx_") and os.path.isdir(p):
                    entries.append((os.path.getmtime(p), name))
            entries.sort(reverse=True)
            allowed = {f"idx_{keep_key}"} | {name for _, name in entries[: self.keep]}
            for _, name in entries:
                if name not in allowed:
                    shutil.rmtree(os.path.join(self.cache_dir, name), ignore_errors=True)
        except Exception:
            pass

    def list_indexes(self) -> List[Dict]:
        out = []
        for name in os.listdir(self.cache_dir):
            p = os.path.join(self.cache_dir, name)
            if not name.startswith("idx_") or not os.path.isdir(p):
                continue
            meta = {}
            meta_path = os.path.join(p, "meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass
            size = sum(os.path.getsize(os.path.join(p, f))
                       for f in os.listdir(p)
                       if os.path.isfile(os.path.join(p, f)))
            out.append({
                "key": name[len("idx_"):],
                "created": meta.get("created", "?"),
                "chunks": meta.get("chunks", "?"),
                "files": meta.get("files", "?"),
                "size_mb": size / (1024 * 1024),
                "morph": os.path.exists(os.path.join(p, "chunk_lemmas.pkl")),
            })
        return sorted(out, key=lambda x: x["created"], reverse=True)

    def delete(self, key: str) -> None:
        shutil.rmtree(self._dir(key), ignore_errors=True)

    def clear(self) -> None:
        for name in os.listdir(self.cache_dir):
            p = os.path.join(self.cache_dir, name)
            if name.startswith("idx_") and os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)


# ==========================================
# 11. MAIN
# ==========================================

def main():
    st.set_page_config(page_title="🔍 UnUniveRRAG: only TXT/MD ", page_icon="📄", layout="wide")
    init_theme()
    init_session_state()
    apply_theme(st.session_state.theme)

    if not st.session_state.ollama_ok:
        ok, msg, models = check_ollama_connection()
        st.session_state.ollama_ok = ok
        st.session_state.ollama_msg = msg
        st.session_state.ollama_models = models

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.title("🔍 UnUniveRRAG: TXT/MD")
        theme_selector()
        st.divider()

        st.subheader("🤖 Ollama")
        if st.session_state.ollama_ok:
            st.success(f"✅ {st.session_state.ollama_msg}")
            st.caption(f"Доступно моделей: {len(st.session_state.ollama_models)}")
        else:
            st.error(f"❌ {st.session_state.ollama_msg}")
            st.code("ollama serve\nollama pull nomic-embed-text-v2-moe:latest", language="bash")
            if st.button("🔄 Проверить", key="retry_ollama"):
                ok, msg, models = check_ollama_connection()
                st.session_state.ollama_ok = ok
                st.session_state.ollama_msg = msg
                st.session_state.ollama_models = models
                st.rerun()
            st.stop()

        st.caption(f"🧬 Морфология: {'✅ доступна' if MORPHOLOGY_AVAILABLE else '❌ не установлена'}")
        st.divider()

        # ---- Источник ----
        st.subheader("📁 Источник")
        load_mode = st.radio(
            "Режим:", ["file", "directory"],
            format_func=lambda x: "📄 Файл" if x == "file" else "📂 Каталог",
            index=0 if st.session_state.load_mode == "file" else 1
        )
        if load_mode != st.session_state.load_mode:
            st.session_state.load_mode = load_mode
            reset_index_state(preserve_source=False)
            st.rerun()

        if st.session_state.load_mode == "file":
            uploaded = st.file_uploader("Загрузить TXT/MD", type=list(SUPPORTED_EXTENSIONS))
            if uploaded:
                if (not st.session_state.processed
                        or uploaded.name != st.session_state.uploaded_file_name):
                    with st.spinner("Обработка..."):
                        raw_bytes = uploaded.getvalue()
                        raw = extract_text_from_bytes(raw_bytes)
                        ext = os.path.splitext(uploaded.name)[1].lower().lstrip(".")
                        if st.session_state.strip_md and ext in ("md", "markdown"):
                            raw = strip_markdown_syntax(raw)
                        reset_index_state(preserve_source=False)
                        st.session_state.uploaded_file_bytes = raw_bytes
                        st.session_state.uploaded_file_name = uploaded.name
                        st.session_state.chunk_docs = [
                            LCDocument(page_content=raw, metadata={"source": uploaded.name})
                        ]
                        st.session_state.file_paths = {uploaded.name: "uploaded"}
                        st.session_state.processed_files = [uploaded.name]
                        st.session_state.processed = True
                    st.success("✅ Обработано!")
        else:
            dir_path = st.text_input("Путь к каталогу:", value=st.session_state.source_directory)
            if st.button("🚀 Обработать", type="primary", use_container_width=True):
                if dir_path and os.path.exists(dir_path):
                    if dir_path != st.session_state.source_directory:
                        reset_index_state(preserve_source=False)
                    st.session_state.source_directory = dir_path
                    reset_index_state(preserve_source=False)
                    st.rerun()
                else:
                    st.error("Каталог не найден!")

            if dir_path and os.path.exists(dir_path) and not st.session_state.processed:
                files = scan_source_files(dir_path)
                if files:
                    prog = st.progress(0.0)
                    status = st.empty()
                    docs, paths, names = [], {}, []
                    total = len(files)
                    for i, fp in enumerate(files):
                        status.text(f"Чтение {i+1}/{total}: {os.path.basename(fp)}")
                        try:
                            content = extract_text_from_path(fp)
                            fname = os.path.basename(fp)
                            ext = os.path.splitext(fname)[1].lower().lstrip(".")
                            if st.session_state.strip_md and ext in ("md", "markdown"):
                                content = strip_markdown_syntax(content)
                            if content.strip():
                                docs.append(LCDocument(page_content=content, metadata={"source": fname}))
                                paths[fname] = fp
                                names.append(fname)
                        except Exception as e:
                            st.warning(f"⚠️ {fp}: {e}")
                        prog.progress((i + 1) / total)
                    prog.empty(); status.empty()
                    if docs:
                        st.session_state.chunk_docs = docs
                        st.session_state.file_paths = paths
                        st.session_state.processed_files = names
                        st.session_state.processed = True
                        st.success(f"✅ Загружено {len(names)} файлов (.txt/.md)")
                        st.rerun()
                    else:
                        st.warning("Пустые или ошибочные файлы")
                else:
                    st.warning("Файлы .txt или .md не найдены")

        st.divider()

        # ---- Обработка ----
        with st.expander("⚙️ Обработка", expanded=False):
            new_strip = st.checkbox(
                "🧹 Очищать разметку Markdown (front-matter, ограждения ```)",
                value=st.session_state.strip_md,
                help="Влияет только на .md/.markdown файлы. Требует переиндексации."
            )
            if new_strip != st.session_state.strip_md:
                st.session_state.strip_md = new_strip
                st.session_state.processed = False

            method_options = ["simple", "recursive", "semantic"] if LANGCHAIN_AVAILABLE else ["simple"]
            new_method = st.selectbox(
                "Разбиение:", method_options,
                index=method_options.index(st.session_state.chunk_method) if st.session_state.chunk_method in method_options else 0,
                format_func=lambda x: {"simple": "📏 Простое", "recursive": "🔄 Recursive",
                                        "semantic": "🧠 Семантическое"}.get(x, x)
            )
            if new_method != st.session_state.chunk_method:
                st.session_state.chunk_method = new_method
                st.session_state.processed = False

            new_size = st.slider("Размер чанка", 100, 2000, st.session_state.chunk_size, 50)
            new_overlap = st.slider("Перекрытие", 0, 500, st.session_state.chunk_overlap, 25)
            if new_size != st.session_state.chunk_size or new_overlap != st.session_state.chunk_overlap:
                st.session_state.chunk_size = new_size
                st.session_state.chunk_overlap = new_overlap
                st.session_state.processed = False

            if st.session_state.chunk_method == "semantic":
                new_th = st.slider("Чувствительность (percentile)", 50, 99, st.session_state.semantic_threshold)
                if new_th != st.session_state.semantic_threshold:
                    st.session_state.semantic_threshold = new_th
                    st.session_state.processed = False

            if st.session_state.processed and st.button("🔄 Пересоздать индекс", use_container_width=True):
                if (st.session_state.index_disk_cache is not None
                        and st.session_state.used_sig):
                    st.session_state.index_disk_cache.delete(st.session_state.used_sig)
                reset_index_state(preserve_source=True)
                st.rerun()

        # ---- Поиск ----
        with st.expander("🔎 Поиск", expanded=False):
            st.session_state.search_mode = st.selectbox(
                "Режим:", ["hybrid", "semantic", "bm25"],
                index=["hybrid", "semantic", "bm25"].index(st.session_state.search_mode),
                format_func=lambda x: {"hybrid": "⚡ Гибридный",
                                        "semantic": "🧠 Семантический",
                                        "bm25": "🔤 BM25"}[x]
            )
            if st.session_state.search_mode == "hybrid":
                st.session_state.semantic_weight = st.slider(
                    "Вес семантики", 0.0, 1.0, st.session_state.semantic_weight, 0.05
                )
            st.session_state.top_k = st.slider("Top-K", 1, 20, st.session_state.top_k, 1)

        st.session_state.min_relevance = st.slider(
            "🎯 Минимальная релевантность",
            min_value=0.0, max_value=1.0,
            value=st.session_state.min_relevance, step=0.05,
            help="Фрагменты ниже порога не пойдут в контекст для LLM."
        )

        # ---- Кэш эмбеддингов ----
        with st.expander("💾 Кэш эмбеддингов", expanded=st.session_state.cache_enabled):
            st.session_state.cache_enabled = st.checkbox("✅ Включить", value=st.session_state.cache_enabled)
            if st.session_state.cache_enabled:
                ttl = st.slider("TTL (дни)", 1, 30, st.session_state.cache_ttl_days)
                if ttl != st.session_state.cache_ttl_days:
                    st.session_state.cache_ttl_days = ttl
                if (st.session_state.embedding_cache is None
                        or st.session_state.embedding_cache.ttl_seconds != ttl * 86400):
                    st.session_state.embedding_cache = EmbeddingCache(ttl_days=ttl)
                stats = st.session_state.embedding_cache.get_stats()
                st.progress(stats['hit_rate'] / 100)
                c1, c2 = st.columns(2)
                c1.metric("✅ Hits", stats['hits'], f"{stats['hit_rate']:.1f}%")
                c2.metric("❌ Misses", stats['misses'])
                st.caption(f"💾 {stats['size_mb']:.1f} MB | {stats['entries']} записей")
                if st.button("🗑️ Очистить кэш", use_container_width=True):
                    st.session_state.embedding_cache.clear()
                    st.success("Кэш очищен!")
                    st.rerun()

        # ---- NEW: Экспорт чанков в TXT/CSV ----
        with st.expander("📤 Экспорт чанков (для анализа)", expanded=False):
            if not st.session_state.chunks:
                st.caption("Чанки ещё не созданы. Сначала постройте индекс.")
            else:
                st.caption(f"Доступно чанков: **{len(st.session_state.chunks)}**")
                export_dir = st.text_input(
                    "Папка:", value=DEFAULT_CHUNK_EXPORT_DIR, key="chunk_export_dir"
                )
                export_mode = st.radio(
                    "Формат:",
                    ["single", "per_chunk", "csv"],
                    format_func=lambda x: {
                        "single": "📄 Один общий .txt",
                        "per_chunk": "🗂️ Отдельный .txt на чанк (по источникам)",
                        "csv": "📊 CSV (для pandas)",
                    }[x],
                    index=0, key="chunk_export_mode"
                )
                include_header = st.checkbox(
                    "Добавлять шапку (индекс, источник, длина)",
                    value=True, key="chunk_export_header",
                    disabled=(export_mode == "csv")
                )
                if len(st.session_state.chunks) > 5000 and export_mode == "per_chunk":
                    st.warning(
                        f"⚠️ {len(st.session_state.chunks)} файлов в per_chunk-режиме "
                        "может занять несколько секунд."
                    )
                if st.button("💾 Экспортировать", use_container_width=True, key="chunk_export_btn"):
                    with st.spinner(f"Экспорт {len(st.session_state.chunks)} чанков..."):
                        try:
                            n, path = export_chunks_to_txt(
                                st.session_state.chunks,
                                export_dir,
                                index_key=st.session_state.used_sig,
                                mode=export_mode,
                                include_header=include_header,
                            )
                            st.success(f"✅ Экспортировано {n} чанков")
                            st.code(path, language="text")
                        except Exception as e:
                            st.error(f"❌ Ошибка экспорта: {e}")

        # ---- Производительность ----
        with st.expander("⚡ Производительность", expanded=False):
            st.session_state.max_chunks_limit = st.slider(
                "Максимум чанков (предупреждение)",
                min_value=500, max_value=20000,
                value=st.session_state.max_chunks_limit, step=100
            )

        # ---- Генерация ----
        with st.expander("🤖 Генерация ответа", expanded=st.session_state.generate_answer):
            st.session_state.generate_answer = st.checkbox(
                "Включить генерацию", value=st.session_state.generate_answer
            )
            if st.session_state.generate_answer:
                models_avail = [m for m in st.session_state.ollama_models
                                if 'embed' not in m.lower()] or [DEFAULT_LLM_MODEL]
                idx = models_avail.index(st.session_state.llm_model) if st.session_state.llm_model in models_avail else 0
                st.session_state.llm_model = st.selectbox("Модель:", models_avail, index=idx)
                if st.button("📥 Загрузить модель в память", use_container_width=True,
                             help="Ускоряет первый запрос."):
                    with st.spinner(f"Загрузка {st.session_state.llm_model}..."):
                        ok, msg = preload_ollama_model(st.session_state.llm_model, keep_alive=-1)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                st.session_state.llm_temperature = st.slider(
                    "Температура", 0.0, 1.0, st.session_state.llm_temperature, 0.1
                )
                st.session_state.llm_max_tokens = st.slider(
                    "Макс. токенов", 100, 8000, st.session_state.llm_max_tokens, 100
                )
                st.session_state.system_prompt = st.text_area(
                    "Системный промт", value=st.session_state.system_prompt, height=100
                )

        st.divider()

        # ---- Сохранить ----
        st.subheader("💾 Сохранить")
        save_dir = st.text_input("Директория:", value=DEFAULT_SAVE_DIR)
        if st.button(
            "💾 Сохранить найденные",
            disabled=not st.session_state.processed or not st.session_state.search_results,
            use_container_width=True
        ):
            with st.spinner("Сохранение..."):
                saved = FileManager.save_found_files(
                    st.session_state.search_results,
                    st.session_state.file_paths,
                    st.session_state.source_directory if st.session_state.load_mode == "directory" else None,
                    save_dir,
                    st.session_state.uploaded_file_bytes,
                    st.session_state.uploaded_file_name
                )
                if saved:
                    st.success(f"✅ Сохранено: {len(saved)}")
                    for s in saved:
                        st.markdown(f"- 📄 `{s}`")
                else:
                    st.warning("⚠️ Ничего не сохранено")

        free_gpu_clicked = st.button(
            "🧹 Очистить память GPU", key='free',
            help='Модели Ollama освобождают память только по нажатию на эту кнопку.',
            use_container_width=True
        )
        if free_gpu_clicked:
            def get_loaded_model():
                result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True)
                lines = result.stdout.strip().split('\n')
                return lines[1].split()[0] if len(lines) >= 2 else None

            def release_gpu_memory(model_name):
                url = 'http://localhost:11434/api/generate'
                headers = {'Content-Type': 'application/json'}
                data = {"model": model_name, "keep_alive": 0, "prompt": ""}
                response = requests.post(url, headers=headers, data=json.dumps(data))
                if response.status_code == 200:
                    st.success(f"Память GPU для модели '{model_name}' успешно освобождена.")
                else:
                    st.error(f"Ошибка: {response.status_code}, {response.text}")

            model = get_loaded_model()
            if model:
                release_gpu_memory(model)
            else:
                st.info("Нет загруженных моделей Ollama.")

    # ---------------- MAIN AREA ----------------
    st.markdown("<h1 style='text-align:center'>📄 гибрИИдный поиск только по TXT/MD</h1>",
                unsafe_allow_html=True)

    if not st.session_state.processed:
        st.markdown(
            "<div style='text-align:center;padding:40px'>"
            "<h2>👉 Загрузите данные или укажите каталог в панели слева</h2>"
            "</div>", unsafe_allow_html=True
        )
        return

    if st.session_state.index_disk_cache is None:
        st.session_state.index_disk_cache = IndexDiskCache(DEFAULT_INDEX_CACHE_DIR, keep=3)
    index_cache: IndexDiskCache = st.session_state.index_disk_cache

    source_manifest = compute_source_manifest(
        st.session_state.load_mode,
        st.session_state.source_directory,
        st.session_state.uploaded_file_name,
        st.session_state.uploaded_file_bytes,
    )
    config_sig = (
        st.session_state.chunk_method,
        st.session_state.chunk_size,
        st.session_state.chunk_overlap,
        st.session_state.semantic_threshold,
        st.session_state.strip_md,
    )
    index_key = compute_index_key(config_sig, source_manifest, DEFAULT_EMBED_MODEL)

    need_rebuild = (
        st.session_state.used_sig != index_key
        or not st.session_state.chunks
        or st.session_state.embeddings is None
        or len(st.session_state.embeddings) == 0
    )

    if need_rebuild:
        loaded = index_cache.load(index_key)
        if loaded:
            st.session_state.chunks = loaded["chunks"]
            st.session_state.chunk_docs = loaded["chunk_docs"]
            st.session_state.embeddings = loaded["embeddings"]
            st.session_state.bm25_index = loaded["bm25_index"]
            st.session_state.file_paths = loaded["file_paths"]
            st.session_state.chunk_lemmas = loaded.get("chunk_lemmas")
            st.session_state.processed_files = list(loaded["file_paths"].keys())
            st.session_state.used_sig = index_key
            morph_marker = " + 🧬" if st.session_state.chunk_lemmas else ""
            st.success(
                f"⚡ Индекс загружен с диска: {len(loaded['chunks'])} чанков{morph_marker} "
                f"(создан {loaded['meta'].get('created', '?')})"
            )
        else:
            with st.spinner("Подготовка документов..."):
                docs = st.session_state.chunk_docs.copy()
                if st.session_state.chunk_method == "simple":
                    chunks = []
                    for doc in docs:
                        base = f"[Файл: {doc.metadata.get('source', 'unknown')}]\n"
                        for c in chunk_text_simple(doc.page_content,
                                                   st.session_state.chunk_size,
                                                   st.session_state.chunk_overlap):
                            chunks.append(base + c)
                else:
                    chunked = chunk_text_langchain(
                        docs, st.session_state.chunk_method,
                        st.session_state.chunk_size, st.session_state.chunk_overlap,
                        model_name=DEFAULT_EMBED_MODEL,
                        semantic_threshold=st.session_state.semantic_threshold,
                        cache=st.session_state.embedding_cache if st.session_state.cache_enabled else None
                    )
                    chunks = [f"[Файл: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
                              for d in chunked]
                if not chunks:
                    st.error("❌ Не удалось получить ни одного чанка из документов.")
                    return
                st.session_state.chunks = chunks

                cache = st.session_state.embedding_cache if st.session_state.cache_enabled else None
                try:
                    st.session_state.embeddings = get_ollama_embeddings_batched(
                        chunks, DEFAULT_EMBED_MODEL, cache, batch_size=EMBEDDING_BATCH_SIZE
                    )
                except Exception:
                    st.session_state.embeddings = None
                    st.session_state.used_sig = None
                    return
                st.session_state.bm25_index = create_bm25_index(chunks)
                st.session_state.used_sig = index_key
                st.session_state.chunk_lemmas = None

                meta = {
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "chunks": len(chunks),
                    "files": len(st.session_state.processed_files),
                    "config": list(config_sig),
                    "embed_model": DEFAULT_EMBED_MODEL,
                }
                index_cache.save(
                    index_key, chunks, docs, st.session_state.embeddings,
                    st.session_state.bm25_index, st.session_state.file_paths, meta,
                    chunk_lemmas=None,
                )
                st.success(f"✅ Индекс готов: {len(chunks)} чанков (сохранён на диск)")

    if len(st.session_state.chunks) > st.session_state.max_chunks_limit:
        st.warning(
            f"⚠️ **Слишком много чанков:** {len(st.session_state.chunks)} "
            f"(текущий лимит: {st.session_state.max_chunks_limit}).\n\n"
            f"💡 **Рекомендации:** Увеличьте лимит в боковой панели или увеличьте `Размер чанка`."
        )

    with st.expander("🗂️ Кэш индекса", expanded=False):
        indexes = index_cache.list_indexes()
        if indexes:
            st.caption(f"Сохранено индексов: {len(indexes)}")
            for it in indexes:
                marker = "🟢" if it["key"] == index_key else "⚪"
                morph = " · 🧬" if it.get("morph") else ""
                st.markdown(
                    f"{marker} `{it['key']}` — {it['chunks']} чанков, "
                    f"{it['files']} файлов, {it['size_mb']:.1f} МБ, "
                    f"{it['created']}{morph}"
                )
        else:
            st.caption("Пока ни одного индекса на диске.")
        if st.button("🗑️ Очистить кэш индекса", use_container_width=True):
            index_cache.clear()
            st.session_state.used_sig = None
            st.success("Кэш индекса очищен!")
            st.rerun()

    query = st.text_input("Ваш вопрос:", placeholder="Например: спутник, МКС, зондирование...")

    if len(st.session_state.processed_files) > 1:
        file_filter = st.selectbox("Фильтр по файлу (для отображения):",
                                   ["Все"] + st.session_state.processed_files)
    else:
        file_filter = "Все"

    if query:
        if len(query.strip()) < 3:
            st.warning("🔍 Введите запрос длиннее 3 символов")
        elif st.button("🔍 Найти и получить ответ", type="primary", use_container_width=True):
            search_start = time.time()
            with st.spinner("Поиск..."):
                cache = st.session_state.embedding_cache if st.session_state.cache_enabled else None
                try:
                    q_emb = get_ollama_query_embedding(query, DEFAULT_EMBED_MODEL, cache)
                except Exception as e:
                    st.error(f"❌ Не удалось получить эмбеддинг запроса: {e}")
                    return
                results = hybrid_search(
                    query, q_emb, st.session_state.embeddings, st.session_state.chunks,
                    st.session_state.bm25_index,
                    top_k=st.session_state.top_k,
                    semantic_weight=st.session_state.semantic_weight,
                    mode=st.session_state.search_mode
                )
                if should_use_morphology_search(query):
                    if (st.session_state.chunk_lemmas is None and st.session_state.chunks):
                        with st.spinner("Строим морфологический индекс (один раз)..."):
                            st.session_state.chunk_lemmas = build_chunk_lemmas(
                                st.session_state.chunks
                            )
                            if (st.session_state.used_sig
                                    and st.session_state.index_disk_cache):
                                st.session_state.index_disk_cache.save_chunk_lemmas(
                                    st.session_state.used_sig, st.session_state.chunk_lemmas
                                )
                    if st.session_state.chunk_lemmas:
                        morph_results = search_morphology(
                            st.session_state.chunks, st.session_state.chunk_lemmas,
                            query, top_k=st.session_state.top_k
                        )
                        if morph_results:
                            morph_texts = {r["text"] for r in morph_results}
                            results = morph_results + [r for r in results if r["text"] not in morph_texts]
                            results = results[:st.session_state.top_k]
                if file_filter != "Все":
                    results = [r for r in results if f"[Файл: {file_filter}]" in r.get('text', '')]
                st.session_state.search_results = results
                st.session_state.last_query = query
            search_time = time.time() - search_start
            st.session_state.last_search_time = search_time

            if not results:
                st.info("Совпадений не найдено")
            else:
                mode_label = {"hybrid": "⚡ Гибрид", "semantic": "🧠 Семантика", "bm25": "🔤 BM25"}[st.session_state.search_mode]
                st.subheader(f"✅ Найдено релевантных фрагментов: {len(results)}")
                cache_stats = (st.session_state.embedding_cache.get_stats()
                               if st.session_state.embedding_cache else None)
                caption = f"⏱️ Поиск: {search_time:.2f} сек"
                if cache_stats:
                    caption += f" | Кэш: {cache_stats['hit_rate']:.0f}% hit rate"
                st.caption(caption)

                for i, res in enumerate(results):
                    score = int(min(res.get('score', 0), 1.0) * 100)
                    fname = (res['text'].split("[Файл:")[1].split("]")[0].strip()
                             if "[Файл:" in res['text'] else "unknown")
                    res_label = "🧬 Морфология" if res.get("method") == "morphology" else mode_label
                    with st.expander(f"#{i+1} | {res_label} {score}% | 📄 {fname}", expanded=(i == 0)):
                        st.progress(min(res.get('score', 0), 1.0))
                        display = res['text'].replace(f"[Файл: {fname}]\n", "")
                        st.markdown(
                            f"<div style='background:var(--bg-card);padding:12px;"
                            f"border-radius:6px;border-left:3px solid var(--primary-color)'>"
                            f"{highlight_query_terms(display, query)}</div>",
                            unsafe_allow_html=True
                        )
                        if 'semantic_score' in res:
                            c1, c2, c3 = st.columns(3)
                            c1.metric("🧠 Семантическая", f"{res['semantic_score']:.3f}")
                            c2.metric("🔤 BM25", f"{res['bm25_score']:.3f}")
                            c3.metric("⚡ Гибридная", f"{res['score']:.3f}")
                        if res.get("method") == "morphology":
                            st.caption(
                                f"🧬 Совпавшие леммы: {', '.join(res.get('matched_lemmas', []))} "
                                f"| вхождений: {res.get('occurrences', 0)}"
                            )

                if st.session_state.generate_answer:
                    st.divider()
                    st.subheader("🤖 Ответ модели")
                    threshold = st.session_state.min_relevance
                    filtered_results = [r for r in results if r.get('score', 0) >= threshold]
                    if not filtered_results:
                        st.warning(f"⚠️ Ни один фрагмент не достиг порога {threshold:.2f}.")
                    else:
                        context_chunks, source_files = [], set()
                        for r in filtered_results[:5]:
                            text = r['text']
                            if "[Файл:" in text:
                                source_files.add(text.split("[Файл:")[1].split("]")[0].strip())
                            context_chunks.append(text)
                        context = "\n\n---\n\n".join(context_chunks)
                        status_box = st.empty()
                        status_box.info(
                            f"🤖 Генерация ответа... | 📊 Фрагментов: {len(filtered_results)} "
                            f"| 📁 Источники: {', '.join(list(source_files)[:3])}"
                        )
                        answer_start = time.time()
                        answer = stream_llm_answer(
                            query, context, st.session_state.llm_model,
                            st.session_state.llm_temperature,
                            st.session_state.llm_max_tokens,
                            st.session_state.system_prompt
                        )
                        answer_time = time.time() - answer_start
                        st.session_state.last_answer_time = answer_time
                        status_box.success(
                            f"✅ Ответ готов за {answer_time:.2f} сек | "
                            f"📊 Фрагментов: {len(filtered_results)}"
                        )
                        col_copy, _ = st.columns([1, 5])
                        with col_copy:
                            if st.button("📋 Показать для копирования",
                                         key=f"copy_{hash(answer + query)}",
                                         use_container_width=True):
                                st.code(answer, language="text")
                        with st.expander("📚 Источники для ответа"):
                            for j, r in enumerate(filtered_results[:5], 1):
                                fn = (r['text'].split("[Файл:")[1].split("]")[0].strip()
                                      if "[Файл:" in r['text'] else "?")
                                st.markdown(f"**{j}.** `{fn}` | Релевантность: `{r.get('score', 0):.3f}`")
                                snippet = r['text'].replace(f"[Файл: {fn}]\n", "")
                                if len(snippet) > 300:
                                    snippet = snippet[:300] + "..."
                                st.markdown(
                                    f"<div>{highlight_query_terms(snippet, query)}</div>",
                                    unsafe_allow_html=True
                                )


if __name__ == "__main__":
    main()