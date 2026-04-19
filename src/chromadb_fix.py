"""
chromadb_fix.py
---------------
Utility to fix ChromaDB compatibility issues and provide helper functions.

ChromaDB recently updated their API. This file helps manage the transition.
"""

import chromadb
from pathlib import Path
from typing import Optional


def get_chromadb_client(persist_dir: Optional[str] = None):
    """
    Get a ChromaDB client that works with both old and new API versions.
    
    Args:
        persist_dir: Directory to persist data (None for in-memory)
    
    Returns:
        ChromaDB client
    """
    if persist_dir:
        persist_dir = str(Path(persist_dir))
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            # New API (ChromaDB 0.4+)
            print(f"📦 Using ChromaDB PersistentClient at {persist_dir}")
            return chromadb.PersistentClient(path=persist_dir)
        except (TypeError, AttributeError):
            # Fallback: old API
            print(f"⚠️ Falling back to legacy ChromaDB API")
            try:
                from chromadb.config import Settings
                settings = Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=persist_dir,
                    anonymized_telemetry=False
                )
                return chromadb.Client(settings)
            except Exception as e:
                print(f"⚠️ Legacy API also failed: {e}")
                print(f"   Using in-memory client instead")
                return chromadb.Client()
    else:
        # In-memory client
        print("📦 Using in-memory ChromaDB client")
        return chromadb.Client()


def get_or_create_collection(client, name: str = "jobs"):
    """
    Get or create a collection with compatible API.
    
    Args:
        client: ChromaDB client
        name: Collection name
    
    Returns:
        Collection object
    """
    try:
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
    except TypeError:
        # Newer API might not accept metadata parameter
        return client.get_or_create_collection(name=name)


def get_collection(client, name: str = "jobs"):
    """
    Get an existing collection.
    
    Args:
        client: ChromaDB client
        name: Collection name
    
    Returns:
        Collection object, or None if not found
    """
    try:
        return client.get_collection(name=name)
    except Exception as e:
        print(f"⚠️ Collection '{name}' not found: {e}")
        return None


# Test function
if __name__ == "__main__":
    print("\n=== ChromaDB Compatibility Test ===\n")
    
    print("1. Testing in-memory client...")
    client = get_chromadb_client()
    print("   ✅ Success\n")
    
    print("2. Creating test collection...")
    collection = get_or_create_collection(client, "test")
    print(f"   ✅ Collection created: {collection.name}\n")
    
    print("3. Adding test data...")
    collection.add(
        ids=["test_1"],
        documents=["This is a test document"],
        embeddings=[[0.1, 0.2, 0.3] * 256]  # 768-dimensional
    )
    print("   ✅ Data added\n")
    
    print("4. Retrieving test data...")
    results = collection.get(ids=["test_1"])
    print(f"   ✅ Retrieved: {results['ids']}\n")
    
    print("=== All tests passed! ===\n")
