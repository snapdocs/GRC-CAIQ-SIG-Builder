# GRC CAIQ / SIG Builder — Claude Instructions

## What This Project Does

Automatically completes enterprise security questionnaires (CAIQ and SIG) using Snapdocs policy documents and historical Q&A, via RAG (retrieval-augmented generation).

- **Input:** blank questionnaire CSVs (`blank_caiq_questions.csv`, `blank_sig_questions.csv`)
- **Output:** completed CSVs with `AI_Selection` and `AI_Explanation` columns filled
- **Cost:** ~$0.007/question via Gemini 2.5 Flash (full SIG ≈ $5.50, full CAIQ ≈ $2.00)

## Key Files

| File | Purpose |
|------|---------|
| `framework_mapper.py` | Main script — all logic lives here |
| `blank_caiq_questions.csv` | 283-question CAIQ template |
| `blank_sig_questions.csv` | 755-question SIG template |
| `requirements.txt` | Python dependencies |
| `.env` | LiteLLM credentials (not committed) |
| `chunks_cache.json` | Cached policy PDF chunks (not committed) |
| `embeddings_cache.npy` | Cached policy embeddings (not committed) |
| `qa_embeddings_cache.npy` | Cached historical Q&A embeddings (not committed) |

Files not in the repo (internal Snapdocs data):
- `policies_folder/` — 23 Snapdocs policy PDFs
- `historical_qa.csv` — historical Q&A rows used for retrieval
- `completed_*.csv` — completed output files

## Setup

```bash
pip install -r requirements.txt
```

Create `.env`:
```
LITELLM_BASE_URL=https://app.litellm.snpd.io
LITELLM_API_KEY=your_key_here
```

Add policy PDFs to `policies_folder/` and `historical_qa.csv` to project root.

## Common Commands

```bash
# Standard run (prompts for which CSV to use)
python3 framework_mapper.py

# Use a specific model
python3 framework_mapper.py --model claude-sonnet-4-6

# Dry run — test retrieval quality, zero API cost
python3 framework_mapper.py --dry-run
python3 framework_mapper.py --dry-run --dry-run-limit 10

# Limit to N questions (useful for testing)
python3 framework_mapper.py --limit 10

# Cleanup mode — fix violations in a completed output file
python3 framework_mapper.py --cleanup
```

## RAG Architecture

- Local model `all-MiniLM-L6-v2` embeds policy chunks and historical Q&A (free, no API cost)
- Chunk size: 1,500 characters; first chunk of every PDF is always pinned (overview)
- Per question: top 10 policy chunks + 23 pinned overview chunks + top 15 historical Q&A rows
- LLM: Gemini 2.5 Flash via LiteLLM proxy (default); Claude Sonnet 4.6 used in cleanup

## Cleanup Mode (`--cleanup`)

Interactive 5-stage flow to fix answer quality issues after a full run:

1. **Parser failures** — free fix; infers N/A from explanation text when JSON selection is missing
2. **Scan for violations** — finds explanations containing banned citation phrases
3. **Gemini rerun** — reruns violating rows (~$0.007/question)
4. **Rescan** — checks if violations remain
5. **Sonnet rerun** — escalates stubborn violations to Claude Sonnet (~$0.05/question)

Each stage pauses with a y/n prompt before proceeding.

### Banned Citation Phrases

These phrases indicate the model cited retrieval internals instead of actual policy:

- "historical Q&A"
- "prior responses"
- "previous responses"
- "context provided"
- "historical context"
- "there is no evidence"
- "the policies do not mention"
- "do not contain information"

## Cache Behavior

Embeddings are cached to disk on first run. If you add or update policy PDFs, delete `chunks_cache.json` and `embeddings_cache.npy` to force a rebuild.

## Answer Quality Notes

- The model is instructed to adopt a confident, outward-facing enterprise vendor voice
- Never cite retrieval mechanisms ("historical Q&A", "prior responses") — cite actual policy names
- N/A answers are inferred from explanation text when JSON parsing fails (signal words: "not applicable", "does not apply", etc.)
- Output is resumable — already-answered rows are skipped on rerun
