import time
import json
import argparse
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def parse_wiki_topics(text):
    """
    Parse Wikipedia-style topic headers from text.
    Returns a list of (topic_hierarchy, content) tuples.

    Example:
        = Main Topic =
        Some text.
        == Subtopic ==
        More text.

    Returns:
        [("Main Topic", "Some text."), ("Main Topic > Subtopic", "More text.")]

    If no topic headers are found, returns the entire text as a single section.
    """
    lines = text.split('\n')
    sections = []
    current_topic_stack = []
    current_content = []
    found_any_topics = False

    for line in lines:
        # Strip leading/trailing whitespace to handle indented headers
        stripped_line = line.strip()

        # Check if line is a topic header
        # Wikipedia format can be: "= Topic =" or "= = Topic = =" (with spaces between =)
        # We need to count the equals signs and extract the topic name
        if stripped_line.startswith('=') and stripped_line.endswith('='):
            # Count leading equals (may have spaces: "= = =" counts as 3)
            leading_equals = re.match(r'^(=\s*)+', stripped_line)
            if leading_equals:
                # Count number of '=' characters in the leading part
                level = leading_equals.group(0).count('=')

                # Extract the topic name (everything between the equals signs)
                # Remove leading/trailing equals and spaces
                topic_name = re.sub(r'^(=\s*)+', '', stripped_line)  # Remove leading = and spaces
                topic_name = re.sub(r'(\s*=)+$', '', topic_name)     # Remove trailing = and spaces
                topic_name = topic_name.strip()

                if topic_name:  # Only process if we have a valid topic name
                    found_any_topics = True
                    # Save previous section if it has content
                    if current_content and current_topic_stack:
                        topic_path = " > ".join(current_topic_stack)
                        content = "\n".join(current_content).strip()
                        if content:
                            sections.append((topic_path, content))

                    # Adjust stack to current level
                    current_topic_stack = current_topic_stack[:level-1] + [topic_name]
                    current_content = []
                    continue

        # If not a topic header, add line to current content
        current_content.append(line)

    # Don't forget the last section
    if current_content and current_topic_stack:
        topic_path = " > ".join(current_topic_stack)
        content = "\n".join(current_content).strip()
        if content:
            sections.append((topic_path, content))

    # If no topics were found, treat entire document as one section
    if not found_any_topics and text.strip():
        sections.append(("Document", text.strip()))

    return sections


def simple_text_splitter(text, chunk_size=3, chunk_overlap=1):
    """
    A simple text splitter that splits by sentences.

    Args:
        text: Text to split
        chunk_size: Number of sentences per chunk
        chunk_overlap: Number of overlapping sentences between chunks

    Returns:
        List of text chunks
    """
    # Split by sentence boundaries
    sentences = text.replace("\n", " ").split('. ')
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    for i in range(0, len(sentences), chunk_size - chunk_overlap):
        chunk = ". ".join(sentences[i:i + chunk_size])
        if chunk:
            # Add period at end if not present
            if not chunk.endswith('.'):
                chunk += "."
            chunks.append(chunk)

    return chunks


def optimized_text_splitter(text, chunk_size=5, chunk_overlap=1, verbose=False):
    """
    Optimized text splitter that parses Wikipedia topics and prepends
    topic context to each chunk.

    Args:
        text: Full text content
        chunk_size: Number of sentences per chunk
        chunk_overlap: Number of overlapping sentences between chunks
        verbose: If True, print each chunk as it's generated

    Returns:
        List of chunks with topic context prepended
    """
    # Parse the text into topic sections
    sections = parse_wiki_topics(text)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Found {len(sections)} topic sections")
        print(f"{'='*60}\n")

    all_chunks = []

    for topic_path, content in sections:
        # Split the content into sentence-based chunks
        content_chunks = simple_text_splitter(content, chunk_size, chunk_overlap)

        # Prepend topic context to each chunk
        for chunk in content_chunks:
            contextualized_chunk = f"[Topic: {topic_path}]\n{chunk}"
            all_chunks.append(contextualized_chunk)

            if verbose:
                print(f"Full chunk (as it will be indexed):")
                print(contextualized_chunk)
                print(f"{'-'*60}\n")

    return all_chunks


def create_index_optimized(
    text_file_path,
    faiss_index_path,
    embedding_model_name,
    chunk_size=5,
    chunk_overlap=1,
    verbose=False
):
    """
    Create a new FAISS index from a text file using optimized chunking.

    Args:
        text_file_path: Path to the text file to index
        faiss_index_path: Path to save the FAISS index
        embedding_model_name: Name of the sentence transformer model
        chunk_size: Number of sentences per chunk
        chunk_overlap: Number of overlapping sentences between chunks
        verbose: If True, print chunks as they're generated

    Returns:
        Tuple of (index, chunks, model, indexing_duration)
    """
    print("Building optimized index from scratch...")
    start_time_indexing = time.time()

    # Load the embedding model
    print(f"Loading embedding model: {embedding_model_name}...")
    model = SentenceTransformer(embedding_model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Embedding dimension: {embedding_dim}")

    # Load the document
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{text_file_path}' was not found.")

    # Use optimized chunking strategy with topic context
    print(f"Processing document with optimized chunking...")
    chunks = optimized_text_splitter(
        text_content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        verbose=verbose
    )

    if len(chunks) == 0:
        raise ValueError("No chunks were generated from the document. Check if the file has content.")

    print(f"Document split into {len(chunks)} chunks with topic context.")

    # Generate embeddings for each chunk
    print("Generating embeddings for all chunks...")
    chunk_embeddings = model.encode(chunks, show_progress_bar=True)

    # Ensure embeddings are properly shaped (should be 2D: [n_chunks, embedding_dim])
    chunk_embeddings = np.array(chunk_embeddings).astype('float32')
    if chunk_embeddings.ndim == 1:
        # Single chunk case - reshape to 2D
        chunk_embeddings = chunk_embeddings.reshape(1, -1)

    # Create a FAISS index
    print("Creating FAISS index...")
    # Using IndexFlatL2 - a simple L2 distance (Euclidean) index
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(chunk_embeddings)

    # Save the index and the chunks
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
    Main function to run optimized index generation from command line.
    """
    parser = argparse.ArgumentParser(
        description='Generate FAISS index with optimized topic-aware chunking for RAG'
    )
    parser.add_argument(
        '--text-file',
        type=str,
        default='data/wikitext2.txt',
        help='Path to the text file to index (default: data/wikitext2.txt)'
    )
    parser.add_argument(
        '--index-path',
        type=str,
        default='index_optimized.faiss',
        help='Path to save the FAISS index (default: index_optimized.faiss)'
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
        default=5,
        help='Number of sentences per chunk (default: 5)'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=1,
        help='Number of overlapping sentences between chunks (default: 1)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print each chunk as it is generated (useful for debugging)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FAISS Index Generation (Optimized with Topic Context)")
    print("=" * 60)
    print(f"Text file:       {args.text_file}")
    print(f"Index path:      {args.index_path}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Chunk size:      {args.chunk_size}")
    print(f"Chunk overlap:   {args.chunk_overlap}")
    print(f"Verbose mode:    {args.verbose}")
    print("=" * 60)

    try:
        index, chunks, model, duration = create_index_optimized(
            text_file_path=args.text_file,
            faiss_index_path=args.index_path,
            embedding_model_name=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            verbose=args.verbose
        )

        print("\n" + "=" * 60)
        print("Index Generation Complete!")
        print("=" * 60)
        print(f"Total chunks:    {len(chunks)}")
        print(f"Total duration:  {duration:.2f} seconds")
        print(f"Index saved to:  {args.index_path}")
        print(f"Chunks saved to: {args.index_path}.json")
        print("=" * 60)

        # Show a sample chunk
        if chunks:
            print("\nSample chunk (first):")
            print("-" * 60)
            print(chunks[0][:300] + "..." if len(chunks[0]) > 300 else chunks[0])
            print("-" * 60)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
