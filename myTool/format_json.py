import json
import sys
import os

def format_json_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    print(f"Reading {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Construct new filename
        base, ext = os.path.splitext(file_path)
        new_file_path = f"{base}_formatted{ext}"
        
        print(f"Writing formatted data to {new_file_path}...")
        with open(new_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        print("Done!")
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    target_file = "/home/ykw/projects/GraphRAG/HippoRAG/outputs/2wikimultihopqa/openie_results_ner_meta-llama_Llama-3.1-8B-Instruct.json"
    format_json_file(target_file)
