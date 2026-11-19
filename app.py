"""
Task 3: Document Q&A with Retrieval-Augmented Generation (RAG)

This small RAG app:
1. Loads documents from ./documents
2. Chunks the documents
3. Creates local embeddings with Sentence Transformers
4. Retrieves relevant passages with cosine similarity
5. Generates an answer using Gemini only from retrieved passages
6. Shows the retrieved passages as citations
7. Honestly says when the answer is not supported by the documents

Set GEMINI_API_KEY before running.
"""

import os
from pathlib import Path

import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer


DOCS_DIR = Path(__file__).parent / "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
MIN_SIMILARITY = 0.30


def load_documents():
    documents = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append((path.name, text))
    if not documents:
        raise RuntimeError("No .txt documents found in the documents directory.")
    return documents


def chunk_text(text, chunk_size=120, overlap=25):
    """Create small overlapping word chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap

    return chunks


def build_index(documents, model):
    chunks = []

    for filename, text in documents:
        for index, chunk in enumerate(chunk_text(text), start=1):
            chunks.append({
                "source": filename,
                "chunk_id": index,
                "text": chunk,
            })

    embeddings = model.encode(
        [item["text"] for item in chunks],
        normalize_embeddings=True,
    )

    return chunks, np.asarray(embeddings)


def retrieve(question, chunks, embeddings, model, top_k=TOP_K):
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    )[0]

    # Dot product of normalized vectors = cosine similarity.
    scores = embeddings @ query_embedding
    ranked = np.argsort(scores)[::-1][:top_k]

    results = []
    for index in ranked:
        results.append({
            **chunks[index],
            "score": float(scores[index]),
        })

    return results


def answer_question(question, retrieved):
    # If the best match is too weak, do not ask the model to guess.
    if not retrieved or retrieved[0]["score"] < MIN_SIMILARITY:
        return (
            "Not in the provided documents.\n\n"
            "Retrieved passages were not sufficiently relevant to support an answer."
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add your Gemini API key as an environment variable."
        )

    context = "\n\n".join(
        f"[{i}] Source: {item['source']} (chunk {item['chunk_id']})\n{item['text']}"
        for i, item in enumerate(retrieved, start=1)
    )

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the supplied document passages.
Do not use outside knowledge.
If the passages do not contain enough information to answer the question,
reply exactly:
Not in the provided documents.

User question:
{question}

Document passages:
{context}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text.strip()


def print_sources(retrieved):
    print("\nRetrieved passages:")
    for i, item in enumerate(retrieved, start=1):
        print(f"\n[{i}] {item['source']} — chunk {item['chunk_id']} "
              f"(similarity: {item['score']:.3f})")
        print(item["text"])


def main():
    print("Loading documents...")
    documents = load_documents()

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Chunking and embedding documents...")
    chunks, embeddings = build_index(documents, embedding_model)

    print(f"Indexed {len(chunks)} chunks from {len(documents)} documents.")

    question = input("\nAsk a question about the documents: ").strip()
    if not question:
        print("Please enter a question.")
        return

    retrieved = retrieve(question, chunks, embeddings, embedding_model)

    answer = answer_question(question, retrieved)

    print("\nAnswer:")
    print(answer)
    print_sources(retrieved)


if __name__ == "__main__":
    main()
