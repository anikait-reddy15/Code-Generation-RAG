import json
import os

dataset_path = os.path.join("data", "training_data.jsonl")

print(f"Validating dataset pathway: {dataset_path}")

if not os.path.exists(dataset_path):
    print("Error: The specified dataset file path does not exist.")
    exit(1)

with open(dataset_path, "r", encoding="utf-8") as file:
    row_zero = file.readline()

print("\n--- Raw Text Extracted from Row 0 ---")
print(row_zero)
print("--------------------------------------\n")

try:
    parsed_json = json.loads(row_zero)
    print("Success: Native JSON module successfully parsed row 0.")
    print("This indicates a structural mismatch or trailing items on downstream lines.")
except json.JSONDecodeError as exception:
    print("Syntax Validation Error Detected:")
    print(f"Reason: {exception.msg}")
    print(f"Character Column Location: {exception.colno}")
    print(f"Global Index Position: {exception.pos}")
    
    if len(row_zero) > 0:
        error_pointer = " " * (exception.colno - 1) + "^"
        print("\nVisual Error Alignment Map:")
        print(row_zero.rstrip())
        print(error_pointer)