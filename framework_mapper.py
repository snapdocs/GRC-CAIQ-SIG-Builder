#!/usr/bin/env python3
"""
Universal Framework Mapper
==========================
Reads a blank security questionnaire (CSV), consults local PDF policy documents
and a historical Q&A CSV, then uses an LLM via a LiteLLM proxy to generate
definitive answers for each question.

Uses RAG (retrieval-augmented generation) to send only relevant policy chunks
per question rather than the full document corpus, dramatically reducing cost.

Installation (run once in your terminal):
    pip install -r requirements.txt

Usage:
    python framework_mapper.py
"""

import argparse
import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

load_dotenv()


@dataclass
class Config:
    litellm_base_url: str
    litellm_api_key: str
    model: str = "gemini-2.5-flash"
    embedding_model: str = "all-MiniLM-L6-v2"
    policies_folder: str = "policies_folder"
    historical_qa_path: str = "historical_qa.csv"
    chunk_size: int = 1500
    chunk_overlap: int = 150
    top_k_chunks: int = 10
    top_k_qa: int = 15
    chunks_cache: str = "chunks_cache.json"
    embeddings_cache: str = "embeddings_cache.npy"
    qa_embeddings_cache: str = "qa_embeddings_cache.npy"
    sleep_between_calls: float = 0.5
    max_workers: int = 10
    max_retries: int = 2
    model_costs: dict = None

    def __post_init__(self):
        if self.model_costs is None:
            self.model_costs = {
                "gemini-2.5-flash": (0.0000003, 0.0000025),
                "claude-sonnet-4-6": (0.000003, 0.000015),
            }

    def output_path(self, framework_csv: str) -> str:
        stem = os.path.splitext(os.path.basename(framework_csv))[0]
        return f"completed_{stem}.csv"


def load_config() -> Config:
    base_url = os.getenv("LITELLM_BASE_URL", "")
    api_key = os.getenv("LITELLM_API_KEY", "")

    if not base_url or not api_key:
        raise EnvironmentError(
            "LITELLM_BASE_URL and LITELLM_API_KEY must be set in your .env file."
        )

    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    return Config(litellm_base_url=base_url, litellm_api_key=api_key)


# ---------------------------------------------------------------------------
# 1. Embedding model (singleton, thread-safe for inference)
# ---------------------------------------------------------------------------

_embedder: SentenceTransformer | None = None
_embedder_lock = threading.Lock()


def get_embedder(embedding_model: str) -> SentenceTransformer:
    """Load the local embedding model once and reuse across all threads."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                print(f"[INFO] Loading local embedding model '{embedding_model}' …")
                _embedder = SentenceTransformer(embedding_model)
    return _embedder


def embed_texts(texts: list[str], cfg: Config) -> np.ndarray:
    """
    Embed a list of strings using the local sentence-transformers model.
    Returns a 2D numpy array of shape (len(texts), embedding_dim).
    Free, no API call required.
    """
    embedder = get_embedder(cfg.embedding_model)
    embeddings = embedder.encode(texts, show_progress_bar=len(texts) > 10, convert_to_numpy=True)
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. PDF text extraction
# ---------------------------------------------------------------------------


def extract_pdf_text(cfg: Config) -> list[dict]:
    """
    Walk policies_folder, open every *.pdf with PyMuPDF, and return a list of
    dicts: [{"source": filename, "text": full_text}, ...]
    """
    docs: list[dict] = []

    if not os.path.isdir(cfg.policies_folder):
        print(f"[WARNING] Policies folder not found: '{cfg.policies_folder}'. Skipping PDF extraction.")
        return docs

    pdf_files = sorted(f for f in os.listdir(cfg.policies_folder) if f.lower().endswith(".pdf"))

    if not pdf_files:
        print(f"[WARNING] No PDF files found in '{cfg.policies_folder}'.")
        return docs

    print(f"\n[INFO] Extracting text from {len(pdf_files)} PDF(s) in '{cfg.policies_folder}' …")

    for filename in pdf_files:
        filepath = os.path.join(cfg.policies_folder, filename)
        try:
            doc = fitz.open(filepath)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            docs.append({"source": filename, "text": text})
            print(f"  ✓ {filename}")
        except Exception as exc:
            print(f"  ✗ Could not read '{filename}': {exc}")

    return docs


# ---------------------------------------------------------------------------
# 3. Chunking
# ---------------------------------------------------------------------------


def chunk_documents(docs: list[dict], cfg: Config) -> tuple[list[dict], list[int]]:
    """
    Split each document's text into overlapping chunks of ~chunk_size characters.
    Returns (chunks, overview_indices) where overview_indices contains the index of
    the first chunk for each document — these contain governance body names, framework
    references, and named mechanisms that make answers authoritative.
    """
    chunks: list[dict] = []
    overview_indices: list[int] = []

    for doc in docs:
        text = doc["text"]
        source = doc["source"]
        start = 0
        is_first_chunk = True

        while start < len(text):
            end = start + cfg.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                if is_first_chunk:
                    overview_indices.append(len(chunks))
                    is_first_chunk = False
                chunks.append({"source": source, "text": chunk_text})
            start += cfg.chunk_size - cfg.chunk_overlap

    return chunks, overview_indices


# ---------------------------------------------------------------------------
# 4. Embedding cache
# ---------------------------------------------------------------------------


def load_or_build_embeddings(chunks: list[dict], cfg: Config) -> np.ndarray:
    """
    Load embeddings from cache if available and chunks haven't changed,
    otherwise embed all chunks and save to disk.
    """
    if os.path.isfile(cfg.chunks_cache) and os.path.isfile(cfg.embeddings_cache):
        print(f"\n[INFO] Loading embeddings from cache ({cfg.embeddings_cache}) …")
        with open(cfg.chunks_cache) as f:
            cached_chunks = json.load(f)

        if cached_chunks == chunks:
            embeddings = np.load(cfg.embeddings_cache)
            print(f"[INFO] Cache valid. Loaded {len(embeddings)} embeddings.")
            return embeddings
        else:
            print("[INFO] Chunks have changed — rebuilding embeddings cache …")

    print(f"\n[INFO] Embedding {len(chunks)} chunks (one-time cost) …")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, cfg)

    np.save(cfg.embeddings_cache, embeddings)
    with open(cfg.chunks_cache, "w") as f:
        json.dump(chunks, f)

    print(f"[INFO] Embeddings saved to '{cfg.embeddings_cache}'.")
    return embeddings


# ---------------------------------------------------------------------------
# 5. Historical Q&A loader
# ---------------------------------------------------------------------------


def load_historical_qa(cfg: Config) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load the historical Q&A CSV and embed each row (Question + Response) for retrieval.
    Returns (dataframe, embeddings) so only the top relevant rows are sent per question.
    """
    if not os.path.isfile(cfg.historical_qa_path):
        print(f"[WARNING] Historical Q&A file not found: '{cfg.historical_qa_path}'. Skipping.")
        return pd.DataFrame(), np.array([])

    try:
        df = pd.read_csv(cfg.historical_qa_path, dtype=str).fillna("")
        print(f"[INFO] Loaded {len(df)} historical Q&A rows from '{cfg.historical_qa_path}'.")

        texts = (df.get("Question", df.iloc[:, 2]) + " " + df.get("Response", df.iloc[:, 3])).tolist()

        if os.path.isfile(cfg.qa_embeddings_cache):
            print(f"[INFO] Loading Q&A embeddings from cache ({cfg.qa_embeddings_cache}) …")
            qa_embeddings = np.load(cfg.qa_embeddings_cache)
            if len(qa_embeddings) == len(texts):
                return df, qa_embeddings
            print("[INFO] Q&A cache stale — rebuilding …")

        print(f"[INFO] Embedding {len(texts)} historical Q&A rows (one-time cost) …")
        qa_embeddings = embed_texts(texts, cfg)
        np.save(cfg.qa_embeddings_cache, qa_embeddings)
        print(f"[INFO] Q&A embeddings saved to '{cfg.qa_embeddings_cache}'.")
        return df, qa_embeddings

    except Exception as exc:
        print(f"[WARNING] Could not load historical Q&A: {exc}")
        return pd.DataFrame(), np.array([])


# ---------------------------------------------------------------------------
# 6. Retrieval
# ---------------------------------------------------------------------------


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and every row in matrix."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norm @ query_norm


def retrieve_historical_qa(
    question_text: str,
    qa_df: pd.DataFrame,
    qa_embeddings: np.ndarray,
    cfg: Config,
) -> str:
    """Retrieve the top relevant historical Q&A rows for a question."""
    if qa_df.empty or len(qa_embeddings) == 0:
        return ""

    query_embedding = embed_texts([question_text], cfg)[0]
    scores = cosine_similarity(query_embedding, qa_embeddings)
    top_indices = np.argsort(scores)[::-1][:cfg.top_k_qa]

    parts = ["=== HISTORICAL Q&A (PAST ANSWERS) ==="]
    for idx in top_indices:
        row = qa_df.iloc[idx]
        question_col = row.get("Question", row.iloc[2])
        response_col = row.get("Response", row.iloc[3])
        parts.append(f"Q: {question_col}\nA: {response_col}")

    return "\n\n".join(parts) + "\n"


def retrieve_relevant_context(
    question_text: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    overview_indices: list[int],
    qa_df: pd.DataFrame,
    qa_embeddings: np.ndarray,
    cfg: Config,
) -> str:
    """
    Embed the question, find the top relevant policy chunks, and return
    them as a formatted string combined with the top relevant historical Q&A rows.
    Always includes the first chunk of every policy document (overview) so that
    named governance bodies, specific frameworks, and exact mechanisms are available
    to the model regardless of which chunks score highest for similarity.
    """
    query_embedding = embed_texts([question_text], cfg)[0]
    scores = cosine_similarity(query_embedding, embeddings)

    # Always include overview chunks; fill remaining slots with top scored chunks
    overview_set = set(overview_indices)
    top_scored = [i for i in np.argsort(scores)[::-1] if i not in overview_set][:cfg.top_k_chunks]
    selected_indices = list(overview_set) + top_scored

    parts: list[str] = ["=== RELEVANT POLICY EXCERPTS ==="]
    for idx in selected_indices:
        chunk = chunks[idx]
        parts.append(f"--- Source: {chunk['source']} ---\n{chunk['text']}")

    historical_context = retrieve_historical_qa(question_text, qa_df, qa_embeddings, cfg)
    parts.append(historical_context)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 7. LLM call
# ---------------------------------------------------------------------------


def call_llm(
    client: OpenAI,
    context_string: str,
    domain: str,
    question_text: str,
    question_type: str,
    cfg: Config,
) -> dict:
    """
    Send a single questionnaire row to the LiteLLM proxy and return a dict
    with keys 'selection', 'explanation', '_input_tokens', '_output_tokens'.
    Retries up to cfg.max_retries times on failure.
    """
    system_content = (
        "You are a strict cybersecurity and GRC expert completing an enterprise "
        "security questionnaire. Use ONLY the provided company policies and "
        "historical Q&A context to answer the question. Do not invent capabilities.\n\n"
        "Look at the expected 'Question Type' options provided by the user "
        "(e.g., Yes, No, N/A, NA, Short Answer). You MUST select exactly one of "
        "those literal text options as your primary answer.\n\n"
        "Crucially, for EVERY single question, regardless of whether the answer is "
        "Yes, No, or N/A, you MUST provide a concise 1-2 sentence narrative "
        "explanation. This explanation should explicitly state HOW the company meets "
        "the requirement, referencing the relevant policy name or context provided.\n\n"
        "When referencing a policy in your explanation, use clean, professional names "
        "(e.g., 'Information Security Policy' or 'Access Management Policy'). "
        "You MUST strictly omit any internal document ID numbers (e.g., L01D001), "
        "version numbers (e.g., v1.6), or file extensions.\n\n"
        "NEVER reference 'historical Q&A', 'prior responses', 'context provided', "
        "or 'historical context' as a source in your explanation — these are internal "
        "retrieval mechanisms, not citable sources. This is an absolute rule with no "
        "exceptions. Always cite the specific policy document or report name instead "
        "(e.g., 'Risk Management Policy', 'SOC 2 Type 2 Report').\n\n"
        "Do not parrot or restate the question in your explanation. Instead, describe "
        "the specific control, process, or policy that satisfies the requirement.\n\n"
        "Adopt a confident, definitive, outward-facing enterprise vendor voice. "
        "Do not merely repeat the phrasing of the prompt. NEVER use phrases like "
        "'there is no evidence in the policies,' 'the policies do not mention,' or "
        "explain what the documents lack. If a specific capability is not present in "
        "the provided context, do not point out the gap; instead, simply state "
        "affirmatively what the company DOES do based on the context provided.\n\n"
        "You MUST respond with ONLY a valid JSON object in this exact format "
        "(no markdown fences, no extra text):\n"
        '{"selection": "Exact Match to Question Type", '
        '"explanation": "1-2 sentence narrative explanation referencing policy"}\n\n'
        "=== COMPANY CONTEXT (POLICIES + HISTORICAL Q&A) ===\n"
        + context_string
    )

    user_content = (
        f"Domain: {domain}\n"
        f"Question Text: {question_text}\n"
        f"Question Type (valid answer options): {question_type}\n\n"
        "Respond ONLY with the JSON object described in the system instructions."
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_content}]},
        {"role": "user", "content": [{"type": "text", "text": user_content}]},
    ]

    last_exc = None
    for attempt in range(cfg.max_retries + 1):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)

            response = client.chat.completions.create(
                model=cfg.model,
                max_tokens=2048,
                messages=messages,
            )

            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            raw_text: str = response.choices[0].message.content or ""
            raw_text = raw_text.strip()

            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            try:
                result = json.loads(raw_text)
                # Guard against empty selection — fall through to regex if missing
                if result.get("selection", "").strip():
                    result["_input_tokens"] = input_tokens
                    result["_output_tokens"] = output_tokens
                    return result
            except json.JSONDecodeError:
                pass

            # Fallback: extract via regex
            selection_match = re.search(r'"selection"\s*:\s*"([^"]*)"', raw_text)
            explanation_match = re.search(r'"explanation"\s*:\s*"(.*?)(?:"\s*\}|$)', raw_text, re.DOTALL)

            # If explanation exists but selection is missing/empty, infer N/A from explanation text
            explanation_text = explanation_match.group(1).strip() if explanation_match else ""
            if not explanation_text and not selection_match:
                raise json.JSONDecodeError("Could not parse response", raw_text, 0)

            selection_value = selection_match.group(1).strip() if selection_match else ""
            if not selection_value:
                na_signals = ["not applicable", "inapplicable", "does not apply", "does not utilize", "does not maintain",
                              "does not operate", "does not collect", "does not have", "does not use", "not relevant"]
                if any(sig in explanation_text.lower() for sig in na_signals):
                    selection_value = "N/A"
                else:
                    raise json.JSONDecodeError("Could not parse response", raw_text, 0)

            return {
                "selection": selection_value,
                "explanation": explanation_text or "See policy documents.",
                "_input_tokens": input_tokens,
                "_output_tokens": output_tokens,
            }

        except Exception as exc:
            last_exc = exc
            if attempt < cfg.max_retries:
                print(f"  [RETRY {attempt + 1}] {type(exc).__name__}: {exc}")

    raise last_exc


# ---------------------------------------------------------------------------
# 8. Row processor (top-level for clarity)
# ---------------------------------------------------------------------------


def process_row(
    row_tuple: tuple,
    client: OpenAI,
    chunks: list[dict],
    embeddings: np.ndarray,
    overview_indices: list[int],
    qa_df: pd.DataFrame,
    qa_embeddings: np.ndarray,
    cfg: Config,
) -> tuple:
    """Process a single questionnaire row. Returns (idx, question_id, selection, explanation, in_tok, out_tok)."""
    idx, question_id, domain, question_text, question_type = row_tuple
    time.sleep(cfg.sleep_between_calls)
    try:
        context_string = retrieve_relevant_context(
            question_text=question_text,
            chunks=chunks,
            embeddings=embeddings,
            overview_indices=overview_indices,
            qa_df=qa_df,
            qa_embeddings=qa_embeddings,
            cfg=cfg,
        )
        result = call_llm(
            client=client,
            context_string=context_string,
            domain=domain,
            question_text=question_text,
            question_type=question_type,
            cfg=cfg,
        )
        return (
            idx, question_id,
            str(result.get("selection", "ERROR")).strip(),
            str(result.get("explanation", "ERROR")).strip(),
            result.get("_input_tokens", 0),
            result.get("_output_tokens", 0),
        )
    except Exception as exc:
        return idx, question_id, "ERROR", f"API_ERROR: {type(exc).__name__}: {exc}", 0, 0


# ---------------------------------------------------------------------------
# 9. Main workflow
# ---------------------------------------------------------------------------


BAD_PHRASES = [
    "historical Q&A",
    "prior responses",
    "previous responses",
    "context provided",
    "historical context",
    "there is no evidence",
    "the policies do not mention",
    "do not contain information",
]


def find_violations(df: pd.DataFrame) -> list[str]:
    """Find rows whose explanation contains a banned phrase, including blank-selection rows."""
    violated = []
    has_explanation = df["AI_Explanation"].notna() & (df["AI_Explanation"].str.strip() != "") & (df["AI_Explanation"].str.strip() != "nan")
    for _, row in df[has_explanation].iterrows():
        expl = str(row["AI_Explanation"])
        for phrase in BAD_PHRASES:
            if phrase.lower() in expl.lower():
                violated.append(row["Question ID"])
                break
    return violated


def reset_rows(df: pd.DataFrame, question_ids: list[str]) -> pd.DataFrame:
    mask = df["Question ID"].isin(question_ids)
    df.loc[mask, "AI_Selection"] = ""
    df.loc[mask, "AI_Explanation"] = ""
    return df


def rerun_questions(question_ids: list[str], output_path: str, cfg: Config) -> None:
    """Rerun specific questions by resetting them and calling the main processing loop."""
    df = pd.read_csv(output_path, dtype=str).fillna("")
    df = reset_rows(df, question_ids)
    df.to_csv(output_path, index=False)

    chunks, overview_indices = build_rag_index(cfg)
    qa_df, qa_embeddings = load_historical_qa(cfg)
    client = OpenAI(api_key=cfg.litellm_api_key, base_url=cfg.litellm_base_url)

    df = pd.read_csv(output_path, dtype=str).fillna("")
    target_ids = set(question_ids)
    unanswered_rows = [
        (idx, row.get("Question ID", ""), row.get("Domain", ""), row.get("Question Text", ""), row.get("Question Type", ""))
        for idx, row in df.iterrows()
        if row.get("Question ID", "") in target_ids
        and str(df.at[idx, "AI_Selection"]).strip() in ("", "nan")
    ]

    lock = threading.Lock()
    total_in = total_out = 0

    print(f"[INFO] Rerunning {len(unanswered_rows)} question(s) with model '{cfg.model}' …")
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {
            executor.submit(process_row, row, client, chunks, embeddings_cache_for_cleanup, overview_indices, qa_df, qa_embeddings, cfg): row
            for row in unanswered_rows
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result is None:
                continue
            idx, question_id, selection, explanation, in_tok, out_tok = result
            with lock:
                df.at[idx, "AI_Selection"] = selection
                df.at[idx, "AI_Explanation"] = explanation
                df.to_csv(output_path, index=False)
                total_in += in_tok
                total_out += out_tok
            print(f"  [{completed}/{len(unanswered_rows)}] {question_id} → {selection}")

    in_rate, out_rate = cfg.model_costs.get(cfg.model, (0.000003, 0.000015))
    cost = total_in * in_rate + total_out * out_rate
    print(f"  Tokens in/out: {total_in:,} / {total_out:,}  |  Cost: ${cost:.4f}")


# Module-level cache so rerun_questions can access embeddings without rebuilding
embeddings_cache_for_cleanup: np.ndarray = None


def build_rag_index(cfg: Config):
    """Build or load the RAG index. Returns (chunks, overview_indices)."""
    global embeddings_cache_for_cleanup
    docs = extract_pdf_text(cfg)
    chunks, overview_indices = chunk_documents(docs, cfg)
    embeddings_cache_for_cleanup = load_or_build_embeddings(chunks, cfg)
    return chunks, overview_indices


def run_cleanup(cfg: Config) -> None:
    print("\n" + "=" * 60)
    print("  Cleanup Mode")
    print("=" * 60)

    # Pick output file
    print("\nAvailable completed CSV files:")
    csv_files = [f for f in os.listdir(".") if f.lower().startswith("completed_") and f.lower().endswith(".csv")]
    for i, name in enumerate(csv_files, 1):
        print(f"  [{i}] {name}")
    output_path = input("\nEnter the completed CSV to clean up: ").strip()
    if not os.path.isfile(output_path):
        raise FileNotFoundError(f"File not found: '{output_path}'")

    df = pd.read_csv(output_path, dtype=str).fillna("")

    # ------------------------------------------------------------------
    # Step 1: Fix parser failures (free — no LLM call)
    # ------------------------------------------------------------------
    parser_failures = df[
        (df["AI_Explanation"].notna()) &
        (df["AI_Explanation"].str.strip() != "") &
        (df["AI_Explanation"].str.strip() != "nan") &
        (df["AI_Selection"].isna() | (df["AI_Selection"].str.strip() == "") | (df["AI_Selection"].str.strip() == "nan"))
    ]

    print(f"\n{'=' * 60}")
    print(f"  Step 1: Parser failures")
    print(f"{'=' * 60}")
    print(f"  Found {len(parser_failures)} row(s) with an explanation but no selection.")
    if len(parser_failures) > 0:
        print("  These are all N/A answers where JSON parsing dropped the selection.")
        for _, row in parser_failures.iterrows():
            print(f"    {row['Question ID']}: {str(row['AI_Explanation'])[:80]}…")
        answer = input("\n  Auto-set these to N/A? (y/n): ").strip().lower()
        if answer == "y":
            df.loc[parser_failures.index, "AI_Selection"] = "N/A"
            df.to_csv(output_path, index=False)
            print(f"  ✓ Fixed {len(parser_failures)} parser failures → N/A")
        else:
            print("  Skipped.")

    # ------------------------------------------------------------------
    # Step 2: Scan for violations
    # ------------------------------------------------------------------
    df = pd.read_csv(output_path, dtype=str).fillna("")
    violations = find_violations(df)

    print(f"\n{'=' * 60}")
    print(f"  Step 2: Violation scan")
    print(f"{'=' * 60}")
    print(f"  Found {len(violations)} violation(s):")
    for qid in violations:
        expl = str(df.loc[df["Question ID"] == qid, "AI_Explanation"].values[0])
        phrase = next(p for p in BAD_PHRASES if p.lower() in expl.lower())
        print(f"    {qid}: \"{phrase}\"")

    if not violations:
        print("  No violations found. All done!")
        return

    est_cost = len(violations) * 0.0073
    answer = input(f"\n  Rerun these {len(violations)} with Gemini (~${est_cost:.2f})? (y/n): ").strip().lower()
    if answer != "y":
        print("  Stopped.")
        return

    # ------------------------------------------------------------------
    # Step 3: Gemini rerun
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Step 3: Gemini rerun ({len(violations)} questions)")
    print(f"{'=' * 60}")
    cfg.model = "gemini-2.5-flash"
    rerun_questions(violations, output_path, cfg)

    # ------------------------------------------------------------------
    # Step 4: Scan again
    # ------------------------------------------------------------------
    df = pd.read_csv(output_path, dtype=str).fillna("")
    still_violated = find_violations(df)

    print(f"\n{'=' * 60}")
    print(f"  Step 4: Post-Gemini scan")
    print(f"{'=' * 60}")
    print(f"  {len(still_violated)} violation(s) remain after Gemini rerun:")
    for qid in still_violated:
        expl = str(df.loc[df["Question ID"] == qid, "AI_Explanation"].values[0])
        phrase = next(p for p in BAD_PHRASES if p.lower() in expl.lower())
        print(f"    {qid}: \"{phrase}\"")

    if not still_violated:
        print("  All violations resolved. Done!")
        return

    est_cost_sonnet = len(still_violated) * 0.05  # Sonnet is ~7x more expensive
    answer = input(f"\n  Rerun these {len(still_violated)} with Sonnet (~${est_cost_sonnet:.2f})? (y/n): ").strip().lower()
    if answer != "y":
        print("  Stopped.")
        return

    # ------------------------------------------------------------------
    # Step 5: Sonnet rerun
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Step 5: Sonnet rerun ({len(still_violated)} questions)")
    print(f"{'=' * 60}")
    cfg.model = "claude-sonnet-4-6"
    rerun_questions(still_violated, output_path, cfg)

    df = pd.read_csv(output_path, dtype=str).fillna("")
    final_violations = find_violations(df)
    print(f"\n  Final violation count: {len(final_violations)}")
    if final_violations:
        print("  Remaining (may need manual review):")
        for qid in final_violations:
            print(f"    {qid}")
    else:
        print("  All clean. Done!")


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show retrieved context per question without calling the LLM")
    parser.add_argument("--dry-run-limit", type=int, default=5, help="Number of questions to preview in dry-run mode (default: 5)")
    parser.add_argument("--limit", type=int, default=0, help="Process only this many questions (0 = all)")
    parser.add_argument("--model", type=str, default=cfg.model, help=f"LiteLLM model name to use (default: {cfg.model})")
    parser.add_argument("--cleanup", action="store_true", help="Fix parser failures and rerun violations with Gemini then Sonnet")
    args = parser.parse_args()

    cfg.model = args.model

    if args.cleanup:
        run_cleanup(cfg)
        return

    print("\n" + "=" * 60)
    print("  Universal Framework Mapper", "[DRY RUN]" if args.dry_run else "")
    print(f"  Model: {cfg.model}")
    print("=" * 60)

    client = OpenAI(api_key=cfg.litellm_api_key, base_url=cfg.litellm_base_url)

    # ------------------------------------------------------------------
    # Ask the user which framework CSV to process
    # ------------------------------------------------------------------
    print("\nAvailable CSV files in this directory:")
    csv_files = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
    for i, name in enumerate(csv_files, 1):
        print(f"  [{i}] {name}")

    framework_csv: str = input(
        "\nEnter the name of the framework CSV you want to process "
        "(e.g., blank_caiq_questions.csv): "
    ).strip()

    if not os.path.isfile(framework_csv):
        raise FileNotFoundError(
            f"Framework CSV not found: '{framework_csv}'. "
            "Please make sure the file is in the current directory."
        )

    output_path = cfg.output_path(framework_csv)

    # ------------------------------------------------------------------
    # Load the framework questionnaire
    # ------------------------------------------------------------------
    print(f"\n[INFO] Loading framework questionnaire: '{framework_csv}' …")
    framework_df = pd.read_csv(framework_csv, dtype=str).fillna("")

    required_columns = {"Question ID", "Domain", "Question Text", "Question Type"}
    missing = required_columns - set(framework_df.columns)
    if missing:
        raise ValueError(
            f"The framework CSV is missing required column(s): {missing}\n"
            f"Found columns: {list(framework_df.columns)}"
        )

    print(f"[INFO] Found {len(framework_df)} question(s) to process.")
    print(f"[INFO] Output will be saved to: '{output_path}'")

    if "AI_Selection" not in framework_df.columns:
        framework_df["AI_Selection"] = ""
    if "AI_Explanation" not in framework_df.columns:
        framework_df["AI_Explanation"] = ""

    # ------------------------------------------------------------------
    # Build RAG index — warm up embedder before threading
    # ------------------------------------------------------------------
    print("\n[INFO] Building RAG index from policy documents …")
    docs = extract_pdf_text(cfg)
    chunks, overview_indices = chunk_documents(docs, cfg)
    print(f"[INFO] {len(chunks)} chunks created from {len(docs)} PDF(s). {len(overview_indices)} overview chunks pinned.")

    embeddings = load_or_build_embeddings(chunks, cfg)
    qa_df, qa_embeddings = load_historical_qa(cfg)

    # Warm up the embedder in the main thread so threads reuse it without reloading
    get_embedder(cfg.embedding_model)

    # ------------------------------------------------------------------
    # Process each row
    # ------------------------------------------------------------------
    total_rows = len(framework_df)
    dry_run_count = 0
    run_start = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    unanswered_rows = []

    if args.dry_run:
        print(f"\n[DRY RUN] Previewing retrieved context for first {args.dry_run_limit} unanswered questions. No API calls will be made.\n")
    else:
        print(f"\n[INFO] Starting processing loop ({total_rows} rows) …\n")

    for idx, row in framework_df.iterrows():
        question_id = row.get("Question ID", f"row-{idx}")
        domain = row.get("Domain", "")
        question_text = row.get("Question Text", "")
        question_type = row.get("Question Type", "")

        if str(framework_df.at[idx, "AI_Selection"]).strip() not in ("", "nan"):
            continue

        if args.dry_run:
            if dry_run_count >= args.dry_run_limit:
                break
            context_string = retrieve_relevant_context(
                question_text=question_text,
                chunks=chunks,
                embeddings=embeddings,
                overview_indices=overview_indices,
                qa_df=qa_df,
                qa_embeddings=qa_embeddings,
                cfg=cfg,
            )
            print(f"{'=' * 60}")
            print(f"Question {dry_run_count + 1}: {question_id}")
            print(f"Q: {question_text}")
            print(f"Type: {question_type}")
            print(f"\n--- Retrieved context ({len(context_string):,} chars) ---")
            excerpts = context_string.split("=== HISTORICAL Q&A")[0]
            print(excerpts[:2000], "…" if len(excerpts) > 2000 else "")
            print()
            dry_run_count += 1
            continue

        unanswered_rows.append((idx, question_id, domain, question_text, question_type))
        if args.limit and len(unanswered_rows) >= args.limit:
            break

    if not unanswered_rows:
        print("[INFO] All questions already answered.")
    else:
        csv_lock = threading.Lock()
        completed = [0]

        print(f"\n[INFO] Processing {len(unanswered_rows)} questions with {cfg.max_workers} workers …\n")

        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = {
                executor.submit(process_row, row, client, chunks, embeddings, overview_indices, qa_df, qa_embeddings, cfg): row
                for row in unanswered_rows
            }
            for future in as_completed(futures):
                idx, question_id, selection, explanation, in_tok, out_tok = future.result()
                framework_df.at[idx, "AI_Selection"] = selection
                framework_df.at[idx, "AI_Explanation"] = explanation
                total_input_tokens += in_tok
                total_output_tokens += out_tok
                completed[0] += 1
                print(f"  [{completed[0]}/{len(unanswered_rows)}] {question_id} → {selection}")
                with csv_lock:
                    framework_df.to_csv(output_path, index=False)

    # ------------------------------------------------------------------
    # Final save and summary
    # ------------------------------------------------------------------
    framework_df.to_csv(output_path, index=False)

    answered = (framework_df["AI_Selection"] != "").sum()
    errors = (framework_df["AI_Selection"] == "ERROR").sum()

    elapsed = time.time() - run_start
    minutes, seconds = divmod(int(elapsed), 60)
    in_cost_per_tok, out_cost_per_tok = cfg.model_costs.get(cfg.model, (0, 0))
    total_cost = (total_input_tokens * in_cost_per_tok) + (total_output_tokens * out_cost_per_tok)

    print("\n" + "=" * 60)
    print("  Processing Complete")
    print("=" * 60)
    print(f"  Total questions : {total_rows}")
    print(f"  Answered        : {answered - errors}")
    print(f"  Errors          : {errors}")
    print(f"  Output saved to : {output_path}")
    print(f"  Model           : {cfg.model}")
    print(f"  Run time        : {minutes}m {seconds}s")
    print(f"  Tokens in/out   : {total_input_tokens:,} / {total_output_tokens:,}")
    print(f"  Estimated cost  : ${total_cost:.4f}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
