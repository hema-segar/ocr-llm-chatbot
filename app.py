from doc_intel import extract_text_from_image, init_reader
from chunker import chunk_text
from embedder import embed_chunks
from indexer import save_index
from prompt_engineering import generate_answer
import numpy as np
import sys


def main():
    # Initialize OCR reader (disable GPU by default)
    init_reader(gpu=False)

    image_path = "SDLC-methodology.png"

    # Extract text from image via EasyOCR
    text = extract_text_from_image(image_path)
    if not text:
        print(f"No text extracted from {image_path}")
        # If you want to abort when extraction fails, uncomment the next line
        # sys.exit(1)

    # Chunk and embed
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    print(chunks)
    print(embeddings)

    # Save index
    save_index(np.array(embeddings), chunks)

    # Example QA loop
    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break
        answer = generate_answer(query)
        print("Answer:", answer)


if __name__ == "__main__":
    main()
