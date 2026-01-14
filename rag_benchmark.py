import os
import time
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llm_client import LLMClient

# --- Configuration ---
# Stage 1: Index Loading Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# # optimized index, chunk size 5, overlap 1
# FAISS_INDEX_PATH = "index_optimized_5_1.faiss"

FAISS_INDEX_PATH = "index_optimized_sentence_3_1.faiss"

# FAISS_INDEX_PATH = "index_optimized_linebased.faiss"

# # original index
# FAISS_INDEX_PATH = "index_my_document.faiss"


# FAISS_INDEX_PATH = None

# Stage 2: Search & Retrieval Configuration
TOP_K = 3 # Number of relevant chunks to retrieve

# USER_QUERY = "What was the Sinclair Sovereign? Include what type of device it was, the year it was introduced, its price range, and one notable or special fact about it."

# USER_QUERY = "Is there a university in Boca Raton, Florida ?"

USER_QUERY = "What is the song Bossy about?"

# Stage 3: LLM Response Configuration
LLAMA_SERVER_BASE_URL = "http://localhost:8080/v1"  # llama-server OpenAI-compatible API
DEFAULT_LLM_SERVER_MODEL = "dummy"  # Model name (can be any string when running single model)
N_LLM_RUNS = 5  # Number of times to repeat LLM generation for averaging
LLM_GEN_TEMPERATURE = 0.0  # Temperature for generation (0=deterministic, 0.8-1.0=creative, default was ~0.8)
MAX_LLM_GEN_TOKENS = 200  # Maximum tokens to generate (controls output length and reduces variance)
PROMPT_FORMAT = "lfm2-rag"  # "default" or "lfm2-rag" (for LFM2-RAG model)
DEBUG_PROMPT = True  # Set to True to print the full prompt sent to the LLM

# --- Helper Functions ---

def load_index(faiss_index_path, embedding_model_name):
    """
    Load an existing FAISS index and its associated chunks.

    Args:
        faiss_index_path: Path to the FAISS index file
        embedding_model_name: Name of the sentence transformer model

    Returns:
        Tuple of (index, chunks, model, loading_duration)
    """
    print(f"Loading existing index from '{faiss_index_path}'...")
    start_time_loading = time.time()

    # Load the embedding model
    print(f"Loading embedding model: {embedding_model_name}...")
    model = SentenceTransformer(embedding_model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Embedding dimension: {embedding_dim}")

    # Load the index
    index = faiss.read_index(faiss_index_path)

    # Load the chunks
    with open(faiss_index_path + ".json", 'r') as f:
        chunks = json.load(f)

    end_time_loading = time.time()
    loading_duration = end_time_loading - start_time_loading

    print(f"Loaded {len(chunks)} chunks from existing index.")
    print("-----------------------------------------------------")
    print(f"BENCHMARK: Loading index took {loading_duration:.4f} seconds.")
    print("-----------------------------------------------------")

    return index, chunks, model, loading_duration

# --- Main Benchmarking Script ---

def main():
    print("--- RAG Performance Benchmark on Raspberry Pi ---")


    # ==================================================================
    # STAGE 1: LOAD INDEX
    # ==================================================================
    print("\n--- STAGE 1: LOAD INDEX ---")

    if not FAISS_INDEX_PATH:
        print("Not loading index -- will just use LLM without retrieval.")
        indexing_duration = 0
    else:
        # Check if index exists
        if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_INDEX_PATH + ".json")):
            print(f"Error: Index not found at '{FAISS_INDEX_PATH}'")
            print("\nTo create an index, run:")
            print(f"  python index_generation.py --index-path {FAISS_INDEX_PATH}")
            return

        index, chunks, model, indexing_duration = load_index(
            faiss_index_path=FAISS_INDEX_PATH,
            embedding_model_name=EMBEDDING_MODEL_NAME
        )


    # ==================================================================
    # STAGE 2: SEARCH & RETRIEVAL
    # ==================================================================
    query = USER_QUERY
    print(f"Sample Query: '{query}'")


    if not FAISS_INDEX_PATH:
        print("\nIndex not loaded; skipping search & retrieval stage.")
        encoding_duration = 0
        retrieval_duration = 0
    else:
        print("\n--- STAGE 2: SEARCH & RETRIEVAL ---")
        

        

        # Embed the query
        start_time_encoding = time.time()
        query_embedding = model.encode([query])
        encoding_duration = time.time() - start_time_encoding

        # Search the FAISS index
        # D: distances, I: indices of the nearest neighbors
        start_time_retrieval = time.time()
        D, I = index.search(np.array(query_embedding).astype('float32'), TOP_K)

        # Retrieve the actual text chunks
        retrieved_chunks = [chunks[i] for i in I[0]]
        
        end_time_retrieval = time.time()
        retrieval_duration = end_time_retrieval - start_time_retrieval

        print(f"\nTop {TOP_K} relevant chunks found:")
        for i, chunk in enumerate(retrieved_chunks):
            print(f"  {i+1}. {chunk}")

        print("-----------------------------------------------------")
        print(f"BENCHMARK: Query encoding took {encoding_duration:.4f} seconds.")
        print(f"BENCHMARK: Retrieval took {retrieval_duration:.4f} seconds.")
        print("-----------------------------------------------------")


    # ==================================================================
    # WARMUP: LLM
    # ==================================================================
    print("\n--- Warming up LLM ---")
    try:
        # Use a long string to warm up KV cache (content doesn't matter, just length)
        warmup_prompt = "warmup " * 100  # ~100 tokens to warm up the model

        with LLMClient(base_url=LLAMA_SERVER_BASE_URL, api_key="dummy") as client:
            start_warmup = time.time()
            _ = client.chat.completions.create(
                model=DEFAULT_LLM_SERVER_MODEL,
                messages=[{"role": "user", "content": warmup_prompt}],
                temperature=LLM_GEN_TEMPERATURE,
                max_tokens=MAX_LLM_GEN_TOKENS,
                stream=False
            )
            warmup_duration = time.time() - start_warmup
            print(f"LLM warmed up in {warmup_duration:.2f} seconds.")
    except Exception as e:
        print(f"Warning: Could not warm up LLM: {e}")
        print("Continuing with benchmark - first LLM call may be slower.")


    # ==================================================================
    # STAGE 3: LLM RESPONSE GENERATION
    # ==================================================================
    print("\n--- STAGE 3: LLM RESPONSE GENERATION ---")
    print(f"Using prompt format: {PROMPT_FORMAT}")

    # Prepare the messages based on the selected prompt format
    if not FAISS_INDEX_PATH:
        system_message = "Instructions: Provide clear, concise answers based on what you know. Limit your response to 3-4 sentences maximum. Be direct and avoid unnecessary elaboration."
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
    else:
        if PROMPT_FORMAT == "lfm2-rag":
            # LFM2-RAG format: system message with documents, user message with question
            documents_str = ""
            for i, chunk in enumerate(retrieved_chunks, 1):
                documents_str += f"<document{i}>\n{chunk}\n</document{i}>\n\n"

            system_message = f"""The following documents may provide you additional information to answer questions:

    {documents_str.strip()}

    Instructions: Provide clear, concise answers based only on the information in the documents. Limit your response to 3-4 sentences maximum. Be direct and avoid unnecessary elaboration."""
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
        else:
            # Default format: single user message with context and question
            context_str = "\n\n".join(retrieved_chunks)
            prompt = f"""Based on the following context, please answer the user's question concisely and directly.
    If the context does not contain the answer, state that the information is not available in the provided context.
    Limit your response to 3-4 sentences maximum. Be clear and focused - avoid unnecessary elaboration.

    Context:
    {context_str}

    Question:
    {query}

    Answer:
    """
            messages = [
                {"role": "user", "content": prompt}
            ]

    # Debug: Print the prompt if DEBUG_PROMPT is enabled
    if DEBUG_PROMPT:
        print("\n" + "="*60)
        print("DEBUG: Prompt sent to LLM:")
        print("="*60)
        for msg in messages:
            print(f"\n[{msg['role'].upper()}]")
            print(msg['content'])
            print("-"*60)
        print("="*60 + "\n")

    print(f"Running LLM generation {N_LLM_RUNS} times for statistics...")
    print(f"(Using temperature={LLM_GEN_TEMPERATURE} and max_tokens={MAX_LLM_GEN_TOKENS} for consistency)\n")
    llm_durations = []
    generated_text = ""

    try:
        # Initialize the LLM client
        with LLMClient(base_url=LLAMA_SERVER_BASE_URL, api_key="dummy") as client:
            for run in range(N_LLM_RUNS):
                print(f"  Run {run + 1}/{N_LLM_RUNS}...")
                start_time_llm = time.time()

                response = client.chat.completions.create(
                    model=DEFAULT_LLM_SERVER_MODEL,
                    messages=messages,
                    temperature=LLM_GEN_TEMPERATURE,
                    max_tokens=MAX_LLM_GEN_TOKENS,
                    stream=False
                )

                end_time_llm = time.time()
                llm_duration = end_time_llm - start_time_llm
                llm_durations.append(llm_duration)

                current_output = response.choices[0].message.content.strip()
                print(f"    Time: {llm_duration:.4f}s")
                print(f"    Output: {current_output}\n")

                # Save the first response to display
                if run == 0:
                    generated_text = current_output

    except Exception as e:
        print(f"\nError connecting to llama-server: {e}")
        print("Please make sure llama-server is running at the configured URL.")
        generated_text = "Error: Could not get a response from the LLM."
        llm_durations = [0.0]  # Placeholder for error case

    # Calculate statistics
    llm_mean = np.mean(llm_durations)
    llm_std = np.std(llm_durations)

    print("-----------------------------------------------------")
    print(f"BENCHMARK: LLM Generation (avg over {N_LLM_RUNS} runs): {llm_mean:.4f} ± {llm_std:.4f} seconds")
    print(f"           Min: {min(llm_durations):.4f}s, Max: {max(llm_durations):.4f}s")
    print("-----------------------------------------------------")
    
    print("\n--- Benchmark Summary ---")
    print(f"  Indexing:            {indexing_duration:.4f} seconds")
    print("--------------------------")
    print(f"  Encoding Query:      {encoding_duration:.4f} seconds")
    print(f"  Retrieval:           {retrieval_duration:.4f} seconds")
    print(f"  LLM Generation:      {llm_mean:.4f} ± {llm_std:.4f} seconds (avg of {N_LLM_RUNS} runs)")
    print("--------------------------")
    print(f"  Total RAG Pipeline:  {encoding_duration + retrieval_duration + llm_mean:.4f} seconds (excluding one-time indexing)")


if __name__ == "__main__":
    main()
