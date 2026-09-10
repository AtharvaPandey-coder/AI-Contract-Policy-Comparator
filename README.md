Contract / Policy Comparator Agent

An agentic system that compares two contract or policy documents (PDFs) clause-by-clause, semantically aligns clauses even when reworded, and uses an LLM to classify each change by type and risk level.

What it does

Given two versions of a contract or policy (e.g. an older and a newer version), the agent:

Extracts and chunks the text of both documents
Semantically matches each clause in Document A to its closest counterpart in Document B — so a reworded clause is still correctly paired, not just an exact-text match
Sends each matched clause pair to an LLM to classify the change (No Change / Reworded / Materially Changed / Removed / Added), assign a risk level (Low / Medium / High), and explain why
Returns a report of only the clauses that actually changed
Tech stack
LangChain (create_agent, tool-calling) — agent orchestration
Groq (openai/gpt-oss-120b) — LLM for clause comparison and reasoning
sentence-transformers (all-MiniLM-L6-v2) via langchain-huggingface — local embeddings for semantic clause matching
FAISS — vector similarity search to align clauses between documents
pypdf (PyPDFLoader) — PDF text extraction
Streamlit — web UI for uploading documents and viewing the report
Project structure
├── document_loader.py   # Loads a PDF and splits it into clause-sized chunks
├── clause_matcher.py     # Embeds chunks and finds each clause's closest semantic match in the other document
├── comparator.py         # Sends matched clause pairs to the LLM for change-type and risk classification
├── agent.py               # Wraps the pipeline as a LangChain tool-calling agent
├── app.py                 # Streamlit UI for uploading documents and viewing the report
└── .env                    # Holds GROQ_API_KEY (not committed)
How it works
document_loader.py reads each PDF page-by-page and splits it into ~800-character chunks, trying to break at paragraph or sentence boundaries so clauses stay intact.
clause_matcher.py embeds Document B's chunks into a FAISS index. For every chunk in Document A, it searches that index for the closest match by meaning, not exact words — this is what allows it to catch a clause that was reworded rather than deleted.
comparator.py takes each matched pair and asks the LLM (temperature 0, for consistent output) to classify the change type, assign a risk level, and give a short explanation, in a fixed, parseable format.
agent.py wires the above into a single LangChain tool (compare_documents) that an agent calls when given two document paths, then returns the compiled report.
app.py provides a simple upload-two-PDFs-and-compare interface using Streamlit.
Setup
bash
pip install langchain langchain-groq langchain-huggingface langchain-community pypdf faiss-cpu sentence-transformers streamlit python-dotenv

Create a .env file in the project root:

GROQ_API_KEY=your_key_here
Run
bash
streamlit run app.py

Upload two PDF versions of a contract or policy and click Compare.

Example use case

Tested against real, publicly filed SEC EDGAR consulting agreements to verify the pipeline correctly identifies additions, removals, and material rewording between two independently drafted documents of the same type.
