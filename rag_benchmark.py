import os
import time
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from llm_client import LLMClient

# --- Configuration ---
# Stage 1: Indexing Configuration
RECREATE_INDEX = False  # Set to True to rebuild the index from scratch
TEXT_FILE_PATH = "data/wikitext2.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "index_my_document.faiss"

# Stage 2: Search & Retrieval Configuration
TOP_K = 3 # Number of relevant chunks to retrieve

# Stage 3: LLM Response Configuration
LLAMA_SERVER_BASE_URL = "http://localhost:8080/v1"  # llama-server OpenAI-compatible API
DEFAULT_LLM_SERVER_MODEL = "dummy"  # Model name (can be any string when running single model)
N_LLM_RUNS = 5  # Number of times to repeat LLM generation for averaging
LLM_GEN_TEMPERATURE = 0.0  # Temperature for generation (0=deterministic, 0.8-1.0=creative, default was ~0.8)
MAX_LLM_GEN_TOKENS = 200  # Maximum tokens to generate (controls output length and reduces variance)

# --- Helper Functions ---

def simple_text_splitter(text, chunk_size=3, chunk_overlap=1):
    """
    A very simple text splitter that splits by sentences.
    A more robust solution would use LangChain's RecursiveCharacterTextSplitter.
    """
    sentences = text.replace("\n", " ").split('. ')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences), chunk_size - chunk_overlap):
        chunk = ". ".join(sentences[i:i + chunk_size])
        if chunk:
            chunks.append(chunk + ".")
            
    return chunks

# --- Main Benchmarking Script ---

def main():
    print("--- RAG Performance Benchmark on Raspberry Pi ---")

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
                stream=False
            )
            warmup_duration = time.time() - start_warmup
            print(f"LLM warmed up in {warmup_duration:.2f} seconds.")
    except Exception as e:
        print(f"Warning: Could not warm up LLM: {e}")
        print("Continuing with benchmark - first LLM call may be slower.")

    # ==================================================================
    # STAGE 1: INDEXING
    # ==================================================================
    print("\n--- STAGE 1: INDEXING ---")

    # Load the embedding model
    # The first time this runs, it will download the model. This is a one-time cost.
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Embedding dimension: {embedding_dim}")

    # Check if index exists and whether to recreate
    index_exists = os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_INDEX_PATH + ".json")

    if index_exists and not RECREATE_INDEX:
        print(f"Loading existing index from '{FAISS_INDEX_PATH}'...")
        start_time_indexing = time.time()

        # Load the index
        index = faiss.read_index(FAISS_INDEX_PATH)

        # Load the chunks
        with open(FAISS_INDEX_PATH + ".json", 'r') as f:
            chunks = json.load(f)

        end_time_indexing = time.time()
        indexing_duration = end_time_indexing - start_time_indexing

        print(f"Loaded {len(chunks)} chunks from existing index.")
        print("-----------------------------------------------------")
        print(f"BENCHMARK: Loading index took {indexing_duration:.4f} seconds.")
        print("-----------------------------------------------------")
    else:
        if RECREATE_INDEX:
            print("RECREATE_INDEX is True. Building index from scratch...")
        else:
            print("No existing index found. Building index from scratch...")

        start_time_indexing = time.time()

        # Load and chunk the document
        try:
            with open(TEXT_FILE_PATH, 'r', encoding='utf-8') as f:
                text_content = f.read()
        except FileNotFoundError:
            print(f"Error: The file '{TEXT_FILE_PATH}' was not found.")
            return

        # Using a simple sentence-based chunking strategy
        chunks = simple_text_splitter(text_content)
        print(f"Document split into {len(chunks)} chunks.")

        # Generate embeddings for each chunk
        print("Generating embeddings for all chunks...")
        chunk_embeddings = model.encode(chunks, show_progress_bar=True)

        # Create a FAISS index
        print("Creating FAISS index...")
        # Using IndexFlatL2 - a simple L2 distance (Euclidean) index
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(np.array(chunk_embeddings).astype('float32'))

        # Save the index and the chunks
        # We need to save the chunks themselves to retrieve the text later
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(FAISS_INDEX_PATH + ".json", 'w') as f:
            json.dump(chunks, f)

        end_time_indexing = time.time()
        indexing_duration = end_time_indexing - start_time_indexing

        print(f"FAISS index created and saved to '{FAISS_INDEX_PATH}'")
        print("-----------------------------------------------------")
        print(f"BENCHMARK: Indexing took {indexing_duration:.4f} seconds.")
        print("-----------------------------------------------------")


    # ==================================================================
    # STAGE 2: SEARCH & RETRIEVAL
    # ==================================================================
    print("\n--- STAGE 2: SEARCH & RETRIEVAL ---")
    
    query = "What was the Sinclair Sovereign and how much did it cost?"
    print(f"Sample Query: '{query}'")

    

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
    # STAGE 3: LLM RESPONSE GENERATION
    # ==================================================================
    print("\n--- STAGE 3: LLM RESPONSE GENERATION ---")

    # Prepare the context for the LLM
    context_str = "\n\n".join(retrieved_chunks)

    # Create the prompt
    prompt = f"""
Based on the following context, please answer the user's question.
If the context does not contain the answer, state that the information is not available in the provided context.

Context:
{context_str}

Question:
{query}

Answer:
"""

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
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
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
