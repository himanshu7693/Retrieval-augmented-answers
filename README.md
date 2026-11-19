# Document RAG — Source-Grounded Q&A

## Task

Answer questions from your own documents instead of relying on the model's memory.

This project is a small Retrieval-Augmented Generation (RAG) system. It reads local text documents, chunks them, creates embeddings, retrieves the most relevant passages for a question, and asks an LLM to answer using only those passages.

## What it demonstrates

- **Documents chunked:** `.txt` files in `documents/` are split into overlapping word chunks.
- **Documents embedded:** Sentence Transformers creates a vector embedding for every chunk.
- **Retrieval:** The question is embedded and compared with document chunks using cosine similarity.
- **Grounded generation:** Gemini receives only the retrieved passages and is instructed not to use outside knowledge.
- **Sources shown:** The program prints the source filename, chunk number, similarity score, and retrieved passage alongside the answer.
- **Honest fallback:** If retrieval is not sufficiently relevant, the program returns `Not in the provided documents.` instead of guessing.

## Project structure

```text
document-rag/
├── app.py
├── requirements.txt
├── README.md
├── company_policy.txt
├── product_guide.txt
└── support_faq.txt
```

## How it works

```text
TXT documents
     |
     v
Chunking with overlap
     |
     v
Sentence Transformer embeddings
     |
     v
In-memory vector index
     |
     v
User question
     |
     v
Question embedding + cosine similarity
     |
     v
Top 3 passages
     |
     v
Gemini using ONLY those passages
     |
     +----> Answer
     |
     +----> Retrieved source passages
```

## Requirements

- Python 3.10+
- Internet access the first time the embedding model is downloaded
- A Gemini API key for answer generation

## Setup

Create a virtual environment if desired:

```bash
python -m venv .venv
```

Activate it:

### Linux/macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the Gemini API key.

### Linux/macOS

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Windows PowerShell

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

Run:

```bash
python app.py
```

The first run may download the Sentence Transformer model.

## Test questions

Try:

```text
What is the standard return window?
```

The answer should be supported by `company_policy.txt`, and the program will display the retrieved passage.

Then try:

```text
What programming language was used to build the company's mobile app?
```

The supplied documents do not contain that information. The program should respond:

```text
Not in the provided documents.
```

This demonstrates that the system does not simply invent an answer when the information is absent.

## Example output

```text
Answer:
The standard return window is 30 days from the delivery date.

Retrieved passages:

[1] company_policy.txt — chunk 1 (similarity: 0.7xx)
Customers may return eligible products within 30 days...
```

Exact similarity scores and generated wording can vary.

## Design decisions

The implementation deliberately uses a small in-memory vector index instead of adding a separate vector database. This keeps the project easy to run and inspect while still demonstrating the core RAG pipeline.

A similarity threshold is used as an additional safety check. If the best retrieved passage is too weakly related to the question, the application does not call the model and instead returns the explicit "Not in the provided documents" response.

The final answer is generated only from retrieved passages, and those passages are printed with source filenames and chunk IDs so the answer can be traced back to the documents.

## Security note

The Gemini API key is read from an environment variable. Never put the key in `app.py`, README files, or commit it to GitHub.
