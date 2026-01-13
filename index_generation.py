import time
import json
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


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


def create_index(
    text_file_path,
    faiss_index_path,
    embedding_model_name,
    chunk_size=3,
    chunk_overlap=1
):
    """
    Create a new FAISS index from a text file.

    Args:
        text_file_path: Path to the text file to index
        faiss_index_path: Path to save the FAISS index
        embedding_model_name: Name of the sentence transformer model
        chunk_size: Number of sentences per chunk
        chunk_overlap: Number of overlapping sentences between chunks

    Returns:
        Tuple of (index, chunks, model, indexing_duration)
    """
    print("Building index from scratch...")
    start_time_indexing = time.time()

    # Load the embedding model
    print(f"Loading embedding model: {embedding_model_name}...")
    model = SentenceTransformer(embedding_model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Embedding dimension: {embedding_dim}")

    # Load and chunk the document
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{text_file_path}' was not found.")

    # Using a simple sentence-based chunking strategy
    chunks = simple_text_splitter(text_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
    faiss.write_index(index, faiss_index_path)
    with open(faiss_index_path + ".json", 'w') as f:
        json.dump(chunks, f)

    end_time_indexing = time.time()
    indexing_duration = end_time_indexing - start_time_indexing

    print(f"FAISS index created and saved to '{faiss_index_path}'")
    print("-----------------------------------------------------")
    print(f"BENCHMARK: Indexing took {indexing_duration:.4f} seconds.")
    print("-----------------------------------------------------")

    return index, chunks, model, indexing_duration


def main():
    """
    Main function to run index generation from command line.
    """
    parser = argparse.ArgumentParser(description='Generate FAISS index from text file for RAG')
    parser.add_argument(
        '--text-file',
        type=str,
        default='data/wikitext2.txt',
        help='Path to the text file to index (default: data/wikitext2.txt)'
    )
    parser.add_argument(
        '--index-path',
        type=str,
        default='index_my_document.faiss',
        help='Path to save the FAISS index (default: index_my_document.faiss)'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Sentence transformer model name (default: sentence-transformers/all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=3,
        help='Number of sentences per chunk (default: 3)'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=1,
        help='Number of overlapping sentences between chunks (default: 1)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FAISS Index Generation")
    print("=" * 60)
    print(f"Text file:       {args.text_file}")
    print(f"Index path:      {args.index_path}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Chunk size:      {args.chunk_size}")
    print(f"Chunk overlap:   {args.chunk_overlap}")
    print("=" * 60)

    try:
        index, chunks, model, duration = create_index(
            text_file_path=args.text_file,
            faiss_index_path=args.index_path,
            embedding_model_name=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )

        print("\n" + "=" * 60)
        print("Index Generation Complete!")
        print("=" * 60)
        print(f"Total chunks:    {len(chunks)}")
        print(f"Total duration:  {duration:.2f} seconds")
        print(f"Index saved to:  {args.index_path}")
        print(f"Chunks saved to: {args.index_path}.json")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
