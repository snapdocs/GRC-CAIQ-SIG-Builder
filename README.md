# GRC CAIQ / SIG Builder

Automatically completes enterprise security questionnaires (CAIQ and SIG) using Snapdocs policy documents and historical Q&A, via RAG (retrieval-augmented generation).

## Background: Why RAG and a Local Model?

Security questionnaires like the CAIQ and SIG can have hundreds of questions. The naive approach — sending all 23 Snapdocs policy PDFs to an AI for every single question — would cost thousands of dollars per run, because AI models charge based on how much text they process.

Instead, this tool uses a technique called **RAG (Retrieval-Augmented Generation)**:

1. Before any questions are answered, a small **local AI model** (running entirely on your machine, no internet required, no cost) reads all the policy documents and converts them into a mathematical representation of their meaning — like a searchable index.
2. When answering each question, the tool first searches that index to find the most relevant policy excerpts and past Q&A answers.
3. Only those relevant excerpts — a small fraction of the total — are sent to the AI model (Gemini) to generate the answer.

The local model (`all-MiniLM-L6-v2`) only handles the search step. It runs on your Mac, costs nothing, and its results are cached to disk so it only does the work once.

### Retrieval Quality Improvements

Two techniques improve answer quality beyond basic similarity search:

- **Pinned overview chunks** — the first chunk of every policy document is always included, regardless of similarity score. Policy introductions contain named governance bodies (e.g. Information Security Council), exact framework versions (ISO/IEC 27001:2022), and specific mechanisms that may not score highly for individual questions but are essential for accurate answers.
- **Larger chunk size** — chunks are 1,500 characters (vs a naive 500) to preserve complete sentences and named entities across context boundaries.

## How It Works

1. **Embeds** 23 Snapdocs policy PDFs into 548 chunks using a local model (`all-MiniLM-L6-v2`) — free, no API cost, cached to disk. 23 overview chunks are always pinned.
2. **Embeds** 1,708 historical Q&A rows the same way
3. **Per question:** retrieves the top 10 relevant policy chunks + 23 pinned overview chunks + top 15 relevant historical Q&A rows
4. **Calls** Gemini 2.5 Flash via the Snapdocs LiteLLM proxy to generate a structured answer
5. **Saves** results to CSV after every answer (supports resuming if interrupted)

Cost: ~$0.007/question. Full SIG (755 questions) ≈ $5.50. Full CAIQ (283 questions) ≈ $2.00.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

This includes `sentence-transformers`, which will automatically download the `all-MiniLM-L6-v2` model (~90MB) from HuggingFace on first run. Internet access is required for this one-time download; subsequent runs use the cached model.

### 2. Create `.env` file

```
LITELLM_BASE_URL=https://app.litellm.snpd.io
LITELLM_API_KEY=your_key_here
```

### 3. Add policy PDFs

Place all Snapdocs policy PDFs in the `policies_folder/` directory.

### 4. Add historical Q&A

Place `historical_qa.csv` in the project root with columns: `Number, Subject Area, Question, Response`

## Usage

### Standard Run

```bash
# Run (prompts for CSV file)
python3 framework_mapper.py

# Use a specific model
python3 framework_mapper.py --model claude-sonnet-4-6

# Dry run — test retrieval quality, zero API cost
python3 framework_mapper.py --dry-run
python3 framework_mapper.py --dry-run --dry-run-limit 10

# Limit to N questions (useful for testing)
python3 framework_mapper.py --limit 10
```

When prompted, enter one of:
- `blank_caiq_questions.csv` — 283 questions
- `blank_sig_questions.csv` — 755 questions

### Cleanup Mode

After a full run, use `--cleanup` to fix answer quality issues in a completed output file. It walks you through each step with a y/n confirmation before proceeding.

```bash
python3 framework_mapper.py --cleanup
```

The cleanup runs in stages, pausing at each step to show you the findings and ask for confirmation before proceeding:

1. **Fix parser failures (free)** — scans for rows where the model returned an explanation but no selection (typically N/A answers where JSON parsing failed). Shows you the affected rows and auto-sets them to N/A at no cost if you confirm.

2. **Scan for violations** — scans all explanations for banned citation phrases (e.g. "historical Q&A", "prior responses"). Shows you the full list with the offending phrase for each row and an estimated cost before asking you to proceed.

3. **Gemini rerun** — reruns only the violating rows with Gemini. Costs ~$0.007/question.

4. **Scan again** — automatically rescans after the Gemini rerun to find any rows that are still violating. Shows you what remains and asks for confirmation before escalating.

5. **Sonnet rerun** — the remaining violations (typically <10) are sent to Claude Sonnet, which follows prompt instructions more strictly. Costs ~$0.05/question.

At each pause you can type `n` to stop — for example if the violation count after Gemini is low enough that you'd rather just accept them than pay for Sonnet.

In practice, a full SIG run (755 questions) produces ~60 violations on the first pass. After one Gemini cleanup rerun, this drops to ~10. Sonnet cleans up the rest, leaving 0 violations.

#### Banned citation phrases

The following phrases are flagged as violations — the model is citing internal retrieval mechanisms instead of actual policy documents:

- "historical Q&A"
- "prior responses"
- "previous responses"
- "context provided"
- "historical context"
- "there is no evidence"
- "the policies do not mention"
- "do not contain information"

## Output

Results are saved to a file named after the input CSV (e.g. `completed_blank_sig_questions.csv`) with columns:

| Column | Description |
|--------|-------------|
| Question ID | Original question identifier |
| Domain | Security domain (e.g. Cryptography, Access Control) |
| Question Text | The question |
| Question Type | Valid answer options (Yes/No/NA etc.) |
| AI_Selection | The answer (must match one of the Question Type options) |
| AI_Explanation | 1-2 sentence narrative citing the relevant policy |

The script skips already-answered rows, so it is safe to interrupt and resume.

## Cache Files

Do not delete these — they avoid re-embedding on every run:

| File | Contents |
|------|----------|
| `chunks_cache.json` | 548 policy text chunks (1,500 chars each) |
| `embeddings_cache.npy` | Embeddings for policy chunks |
| `qa_embeddings_cache.npy` | Embeddings for historical Q&A rows |

If you add or update policy PDFs, delete `chunks_cache.json` and `embeddings_cache.npy` so they rebuild on the next run.

## Available Models

| Model | Cost (in/out per 1M tokens) | Notes |
|-------|-----------------------------|-------|
| `gemini-2.5-flash` | $0.30 / $2.50 | Default — fast, cheap, good quality |
| `claude-sonnet-4-6` | $3.00 / $15.00 | Used in cleanup for stubborn violations |

## Architecture

```
Policy PDFs (23) ──► Chunker (1,500 char chunks) ──► Embedder (local) ──► Cache
                          │                                                    │
                    Overview chunks                                            │
                    (first chunk per doc,                                      │
                     always pinned)                                            │
                                                                               │
Historical Q&A ─────────────────────────► Embedder (local) ──► Cache         │
                                                                               │
Question ──► Embedder (local) ──► Cosine similarity ──► Top chunks ◄──────────┘
                                                               │
                                                        LiteLLM Proxy
                                                               │
                                                        Gemini 2.5 Flash
                                                               │
                                                        JSON answer
                                                               │
                                                  completed_blank_sig_questions.csv
                                                  completed_blank_caiq_questions.csv
```
