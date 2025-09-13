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

    # Get the main dictionary of information
    fields = data.get("fields", {})
    
    # This list will hold each line of our final document
    document_parts = []

    # Process each key-value pair from the JSON
    for key, value in fields.items():
        # Clean up the key text
        key_cleaned = key.strip()
        
        # Check if the value is a list of strings or just a single string
        if isinstance(value, list):
            # If it's a list, join all items with a comma and space
            value_str = ", ".join(str(v).strip() for v in value if v)
        else:
            # If it's a single string, just use it
            value_str = str(value).strip()
        
        # Combine them into a readable format, e.g., "Topic: Some text"
        if value_str: # Only add if there is content
            document_parts.append(f"{key_cleaned}: {value_str}")

    # Join all the parts into one single block of text
    full_document_text = "\n".join(document_parts)
    
    # Format the final text into a Python list string that you can copy
    documents_list_for_code = f'documents = [\n    """\n{full_document_text}\n"""\n]'
    
    return documents_list_for_code

# --- This is the main part that runs when you execute the file ---
if __name__ == "__main__":
    # Ensure this filename matches the one you saved
    json_file_to_process = "KM1139171_clean.json" 
    
    formatted_documents = create_document_from_clean_json(json_file_to_process)
    
    if formatted_documents:
        print("--- SUCCESS! COPY THE TEXT BELOW ---")
        print("--- Paste this into both app.py and create_index.py, replacing the old 'documents' list ---")
        print("\n" + "="*50 + "\n")
        print(formatted_documents)
        print("\n" + "="*50 + "\n")
