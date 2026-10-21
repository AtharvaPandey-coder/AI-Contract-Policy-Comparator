# Contract / Policy Comparator Agent

An AI agent that compares two versions of a contract or policy (PDF), finds which clauses were changed, added or removed, and classifies each change by type and risk level with a short explanation.

## The problem

Reviewing a new version of a contract or policy means reading both documents side by side and checking every clause. Clauses get reworded, moved or dropped, so a plain text diff misses most of what matters. This project automates that check and returns a short report of only the clauses that changed.

## How it works

1. **Load and chunk:** each PDF is read and split into small, clause-sized chunks.
2. **Embed and index:** the chunks of Document B are embedded and stored in a FAISS index.
3. **Semantic alignment:** each chunk of Document A is searched against that index to find its closest match in Document B by meaning, so a reworded clause is still paired correctly.
4. **LLM verdict:** each matched pair goes to the LLM, which returns a structured verdict: `change_type`, `risk_level` and `explaination`.
5. **Filter and report:** clauses marked "No Change" are dropped. Chunks in Document B that no clause in Document A matched are reported as "Added".
6. **Agent:** the whole pipeline is wrapped as a tool (`compare_documents`) that a LangChain agent calls and returns verbatim.

```
PDF A, PDF B
   -> chunks
   -> FAISS index of B
   -> A-to-B semantic matching
   -> LLM verdict per pair
   -> filtered report
```

## Tech stack

- **LangChain:** agent (`create_agent`) and tool calling
- **Groq** (`openai/gpt-oss-120b`): clause comparison and risk classification
- **Pydantic structured output:** forces the verdict into fixed fields (change type, risk level, explanation)
- **sentence-transformers** (`all-MiniLM-L6-v2`): local embeddings
- **FAISS:** vector similarity search for clause alignment
- **PyPDFLoader (pypdf):** PDF text extraction
- **Streamlit:** web UI for uploading the two documents

## Setup

```bash
pip install langchain langchain-groq langchain-huggingface langchain-community pypdf faiss-cpu sentence-transformers streamlit python-dotenv
```

Create a `.env` file in the project folder:

```
GROQ_API_KEY=your_key_here
```

Get a free key at console.groq.com.

## Run

```bash
streamlit run app.py
```

Upload the older version as Document A and the newer version as Document B, then click Compare.

## Example output

Each changed clause appears as one line:

```
[High] [Materially Changed] [Retention period increased from 2 years to 5 years and the legal-requirement exception was removed.]
[High] [Materially Changed] [External sharing no longer requires employee consent.]
[Medium] [Added] [A new clause allows automated productivity evaluation with no process to challenge it.]
```

## Testing

Tested on:
- a sample Employee Data Protection Policy (original vs revised version) with known changes: retention period, data sharing consent, removed correction rights and a new automated decision-making clause
- real consulting agreements filed with the SEC (EDGAR), which have different structure and wording

## Design notes

- **Chunk size 400, overlap 0:** keeps each chunk close to one clause. Overlap is avoided because it would repeat the same sentence in two chunks and blur the matching.
- **Structured output instead of text parsing:** the LLM fills fixed fields, so the risk level can only be Low, Medium or High and no output parsing is needed.
- **Added clauses:** the A-to-B matching only finds counterparts for clauses in A, so unmatched chunks in B are reported separately as additions.
- **Agent prompt:** the agent returns the tool report verbatim, so the model cannot reword the risk labels.

## Limitations

- Works on text-based PDFs only. Scanned PDFs need OCR first.
- Matching is one-to-one from A to B, so if two clauses in A map to the same clause in B, one of them may be paired imprecisely.
- LLM risk levels are a first-pass screen and not legal advice.
