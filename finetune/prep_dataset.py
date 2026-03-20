import json
import os

def prepare_local_dataset():
    # Define relative paths based on the project structure
    input_path = r"C:\Projects\Code Generation RAG\data\training_data.json"
    output_path = r"C:\Projects\Code Generation RAG\data\training_data.jsonl"
    
    # Verify the input file actually exists before starting
    if not os.path.exists(input_path):
        print(f"Error: Could not find the file at {input_path}")
        print("Make sure your JSON file is named exactly 'training_data.json' and is inside the 'data' folder.")
        return

    print(f"Reading local JSON from {input_path}...")
    
    # Load the Alpaca JSON data into memory
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Found {len(data)} records. Converting to JSONL format...")

    # Open the JSONL file for writing
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in data:
            # Safely get the instruction and input
            user_content = row.get("instruction", "")
            input_text = row.get("input", "")
            
            # Merge input into the instruction if it exists
            if input_text and input_text.strip() != "":
                user_content += "\n\n" + input_text
                
            # Restructure data into the ChatML 'messages' format Unsloth expects
            formatted_row = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": row.get("output", "")}
                ]
            }
            
            # Write the formatted row as a single string line
            f.write(json.dumps(formatted_row) + "\n")
            
    print(f"Success! Your Unsloth-ready dataset is saved to {output_path}")

if __name__ == "__main__":
    prepare_local_dataset()