


import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.google import GeminiEmbedding
from utils import is_useful_node

load_dotenv()

Settings.embed_model = GeminiEmbedding(
    model_name="models/gemini-embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY"),
    batch_size=20
)
# retriver.py — replace GeminiEmbedding entirely
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Settings.embed_model = HuggingFaceEmbedding(
#     model_name="BAAI/bge-small-en-v1.5",  # runs locally, no API calls, no limits
# )

def build_hybrid_index(policy_nodes):
    # --- STEP 1: Metadata Enrichment ---
    before = len(policy_nodes)
    policy_nodes = [n for n in policy_nodes if is_useful_node(n)]
    print(f"🔍 Filtered: {before} → {len(policy_nodes)} nodes")

    current_heading = "General Policy"
    for node in policy_nodes:
        if node.metadata.get("role") in ["heading", "title"]:
            current_heading = node.text
        node.metadata["section_title"] = current_heading

    # --- STEP 2: Simple in-memory vector index (no ChromaDB) ---
    vector_index = VectorStoreIndex(
        policy_nodes,
        show_progress=True,
    )

    # --- STEP 3: BM25 ---
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=policy_nodes,
        similarity_top_k=5,
    )

    return vector_index, bm25_retriever

def run_per_item_retrieval(vector_index, bm25_retriever, bill_item, top_k=7):
    """
    Retrieve the most relevant policy clauses for a single bill line item.

    Parameters
    ----------
    bill_item : dict with keys 'description' and optionally 'code'
    """

    description = bill_item.get("description", "")
    code        = bill_item.get("code", "")

    if code:
        query_str = f"Coverage, limits, and exclusions for procedure code {code}: {description}"
    else:
        query_str = f"Coverage, limits, and exclusions for: {description}"

    # Hybrid fusion: vector similarity + BM25 keyword match
    retriever = QueryFusionRetriever(
        [
            vector_index.as_retriever(similarity_top_k=top_k),
            bm25_retriever,
        ],
        num_queries=1,
        use_async=False,          # set True only if you're in an async context
        similarity_top_k=top_k,
        mode="reciprocal_rerank",
    )

    initial_nodes = retriever.retrieve(query_str)

    # # Re-rank to the top 3 most legally relevant clauses
    # reranker    = LLMRerank(top_n=3)
    # final_nodes = reranker.postprocess_nodes(initial_nodes, query_str=query_str)
    return initial_nodes[:3]


# from llama_index.core import VectorStoreIndex, StorageContext
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.retrievers.bm25 import BM25Retriever
# from llama_index.core.retrievers import QueryFusionRetriever

# import chromadb
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.core import VectorStoreIndex, StorageContext
# from llama_index.retrievers.bm25 import BM25Retriever
# from llama_index.core.retrievers import QueryFusionRetriever
# from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter


# def build_hybrid_index(policy_nodes):
#     # --- STEP 1: Metadata Enrichment (Keep your logic) ---
#     current_heading = "General Policy"
#     for node in policy_nodes:
#         if node.metadata.get("role") in ["heading", "title"]:
#             current_heading = node.text
#         node.metadata["section_title"] = current_heading
#         node.metadata["effective_year"] = 2026 

#     # --- STEP 2: Initialize ChromaDB ---
#     # "path" is where the data will be stored on your laptop/server
#     db = chromadb.PersistentClient(path="./chroma_db")
    
#     # Create or get a collection (like a table in SQL)
#     chroma_collection = db.get_or_create_collection("insurance_policies")
    
#     # Wrap it in LlamaIndex's ChromaVectorStore
#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
#     # Assign the vector store to the storage context
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)

#     # --- STEP 3: Create the Vector Index ---
#     # This will now push your nodes into the ChromaDB storage
#     vector_index = VectorStoreIndex(
#         policy_nodes, 
#         storage_context=storage_context,
#         show_progress=True
#     )
    
#     # --- STEP 4: Keyword (BM25) Retriever ---
#     # BM25 is usually in-memory for speed, but uses the same nodes
#     bm25_retriever = BM25Retriever.from_defaults(
#         nodes=policy_nodes, 
#         similarity_top_k=5
#     )
    
#     return vector_index, bm25_retriever


# from llama_index.core.postprocessor import LLMRerank
# from llama_index.core.retrievers import QueryFusionRetriever
# from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# def run_per_item_retrieval(vector_index, bm25_retriever, bill_item, top_k=5):
#     """
#     Retrieves policy clauses specifically for one bill line item.
#     bill_item: dict containing 'description', 'amount', and optionally 'code'
#     """
    
#     # --- STEP A: Targeted Query Construction ---
#     description = bill_item.get('description', '')
#     code = bill_item.get('code', '') # CPT or ICD code if extracted
    
#     # If a code exists, we prioritize it in the query string
#     if code:
#         query_str = f"Coverage, limits, and exclusions for procedure code {code}: {description}"
#     else:
#         query_str = f"Coverage, limits, and exclusions for: {description}"

#     # --- STEP B: Dynamic Filtering (Year/Policy Type) ---
#     # filters = MetadataFilters(filters=[
#     #     ExactMatchFilter(key="effective_year", value=2026),
#     # ])

#     # --- STEP C: Hybrid Fusion ---
#     retriever = QueryFusionRetriever(
#         [
#             vector_index.as_retriever(similarity_top_k=top_k),
#             bm25_retriever
#         ],
#         num_queries=1,
#         use_async=True,
#         similarity_top_k=top_k,
#         mode="reciprocal_rerank"
#     )

#     # --- STEP D: Reranking for Legal Precision ---
#     initial_nodes = retriever.retrieve(query_str)
    
#     # We narrow down to the top 3 most relevant legal clauses for this specific item
#     reranker = LLMRerank(top_n=3) 
#     final_nodes = reranker.postprocess_nodes(initial_nodes, query_str=query_str)
    
#     return final_nodes


# def build_hybrid_index(policy_nodes):
#     # --- STEP 1: Metadata Enrichment ---
#     current_heading = "General Policy"
#     for node in policy_nodes:
#         if node.metadata.get("role") in ["heading", "title"]:
#             current_heading = node.text
#         node.metadata["section_title"] = current_heading

#     # --- STEP 2: ChromaDB persistent store ---
#     db = chromadb.PersistentClient(path="./data")
#     chroma_collection = db.get_or_create_collection("insurance_policies")
#     vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#     storage_context = StorageContext.from_defaults(vector_store=vector_store)

#     # --- STEP 3: Skip re-embedding if already indexed ---
#     if chroma_collection.count() > 0:
#         vector_index = VectorStoreIndex.from_vector_store(
#             vector_store=vector_store,
#         )
#     else:
#         vector_index = VectorStoreIndex(
#             policy_nodes,
#             storage_context=storage_context,
#             show_progress=True,
#         )

#     # --- STEP 4: BM25 keyword retriever ---
#     bm25_retriever = BM25Retriever.from_defaults(
#         nodes=policy_nodes,
#         similarity_top_k=5,
#     )

#     return vector_index, bm25_retriever
