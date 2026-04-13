"""
RAG Knowledge Base using ChromaDB for educational resource retrieval.
Embeds curated study resources and provides a retriever for the agent.
"""

import os
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.rag.resource_data import STUDY_RESOURCES


# Use a persistent directory within the project
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")


def _build_documents() -> tuple[list[str], list[dict], list[str]]:
    """Convert resource data into documents, metadata, and IDs for ChromaDB."""
    documents = []
    metadatas = []
    ids = []

    for i, resource in enumerate(STUDY_RESOURCES):
        # Combine all fields into a rich document for embedding
        doc_text = (
            f"{resource['title']} — {resource['subject']} ({resource['level']})\n"
            f"{resource['description']}\n"
            f"URL: {resource['url']}"
        )
        documents.append(doc_text)
        metadatas.append({
            "title": resource["title"],
            "url": resource["url"],
            "subject": resource["subject"],
            "level": resource["level"],
        })
        ids.append(f"resource_{i}")

    return documents, metadatas, ids


def initialize_knowledge_base() -> chromadb.Collection:
    """
    Initialize or load the ChromaDB collection with study resources.
    Uses ChromaDB's built-in default embedding function (all-MiniLM-L6-v2).
    """
    abs_chroma_dir = os.path.abspath(CHROMA_DIR)
    client = chromadb.PersistentClient(path=abs_chroma_dir)

    collection = client.get_or_create_collection(
        name="study_resources",
        metadata={"description": "Curated educational resources for student recommendations"},
    )

    # Only populate if empty
    if collection.count() == 0:
        documents, metadatas, ids = _build_documents()
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"✅ Knowledge base initialized with {len(documents)} resources.")
    else:
        print(f"📚 Knowledge base loaded with {collection.count()} resources.")

    return collection


def query_resources(collection: chromadb.Collection, query: str, n_results: int = 5, subject_filter: str = None) -> list[dict]:
    """
    Query the knowledge base for relevant resources.

    Args:
        collection: ChromaDB collection
        query: Search query describing what the student needs
        n_results: Number of results to return
        subject_filter: Optional subject filter ('Math', 'Science', 'English', etc.)

    Returns:
        List of resource dicts with title, url, subject, level, and relevance score
    """
    where_filter = None
    if subject_filter:
        where_filter = {"subject": subject_filter}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
        include=["metadatas", "documents", "distances"],
    )

    resources = []
    if results and results["metadatas"] and results["metadatas"][0]:
        for i, metadata in enumerate(results["metadatas"][0]):
            resource = {
                "title": metadata["title"],
                "url": metadata["url"],
                "subject": metadata["subject"],
                "level": metadata["level"],
                "description": results["documents"][0][i] if results["documents"] else "",
                "relevance_score": 1 - (results["distances"][0][i] if results["distances"] else 0),
            }
            resources.append(resource)

    return resources


def search_resources_for_student(collection: chromadb.Collection, weak_subjects: list[str], classification: str, has_internet: bool = True) -> list[dict]:
    """
    Search for resources tailored to a student's specific needs.

    Args:
        collection: ChromaDB collection
        weak_subjects: List of subjects the student is weak in
        classification: Student classification (At-Risk, Average, etc.)
        has_internet: Whether the student has internet access

    Returns:
        Combined list of relevant resources
    """
    all_resources = []
    seen_urls = set()

    # Search for each weak subject
    for subject in weak_subjects:
        query = f"Resources for improving {subject} scores for {classification} students"
        results = query_resources(collection, query, n_results=3, subject_filter=subject)
        for r in results:
            if r["url"] not in seen_urls:
                all_resources.append(r)
                seen_urls.add(r["url"])

    # Add study skills resources
    study_query = f"Study techniques and time management for {classification} students"
    study_results = query_resources(collection, study_query, n_results=2)
    for r in study_results:
        if r["url"] not in seen_urls:
            all_resources.append(r)
            seen_urls.add(r["url"])

    # If no internet, prioritize offline resources
    if not has_internet:
        offline_query = "Offline learning resources downloadable no internet"
        offline_results = query_resources(collection, offline_query, n_results=3)
        for r in offline_results:
            if r["url"] not in seen_urls:
                all_resources.append(r)
                seen_urls.add(r["url"])

    return all_resources
