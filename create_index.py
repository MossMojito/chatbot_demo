import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json

def create_document_from_clean_json(file_path):
    """
    Reads the cleaned JSON file and converts it into a single,
    human-readable text document for the RAG pipeline.
    This function preserves all information.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Make sure it's in the same folder.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return None

    fields = data.get("fields", {})
    document_parts = []

    for key, value in fields.items():
        key_cleaned = key.strip()
        
        if isinstance(value, list):
            value_str = ", ".join(str(v).strip() for v in value if v)
        else:
            value_str = str(value).strip()
        
        if value_str:
            document_parts.append(f"{key_cleaned}: {value_str}")

    full_document_text = "\n".join(document_parts)
    return [full_document_text] # Return as a list with one item for the RAG pipeline

# --- Main part of the script ---
if __name__ == "__main__":
    
    # 1. Load the clean document from the JSON file
    json_file = "KM1139171_clean.json"
    documents = create_document_from_clean_json(json_file)

    if documents:
        # 2. Load the embedding model
        print("Loading sentence transformer model...")
        embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        # 3. Create embeddings for the document
        print("Creating embeddings for the document...")
        embeddings = embedding_model.encode(documents)
        embeddings = np.array(embeddings).astype('float32')

        # 4. Build the FAISS index
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # 5. Save the index to a file
        index_file_name = "km1139171_index.faiss"
        faiss.write_index(index, index_file_name)

        print(f"\nSUCCESS! FAISS index file '{index_file_name}' has been created.")
    else:
        print("Could not create documents from JSON. Please check the file.")
