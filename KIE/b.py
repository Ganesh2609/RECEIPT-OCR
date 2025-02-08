import json
import csv
import os
from pathlib import Path

def json_to_two_column_csv(input_file, output_file):
    """
    Convert a JSON file to a two-column CSV format with 'label' and 'value' columns.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output CSV file
    """
    try:
        # Read JSON file
        with open(input_file, 'r') as file:
            data = json.load(file)
        
        # Write to CSV with two columns
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            
            # Write header
            writer.writerow(['label', 'value'])
            
            # Write data
            for key, value in data.items():
                writer.writerow([key, value])
        
        return True
    
    except Exception as e:
        print(f"Error converting {input_file}: {str(e)}")
        return False

def process_folder(input_folder, output_folder):
    """
    Convert all JSON files in input folder to two-column CSV format
    
    Args:
        input_folder (str): Path to folder containing JSON files
        output_folder (str): Path to folder where CSV files will be saved
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all JSON files in the input folder
    json_files = list(Path(input_folder).glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return
    
    print(f"Found {len(json_files)} JSON files to convert...")
    
    successful = 0
    failed = 0
    
    for json_file in json_files:
        input_path = str(json_file)
        output_path = os.path.join(output_folder, json_file.stem + '.csv')
        
        print(f"Converting: {json_file.name}...")
        
        if json_to_two_column_csv(input_path, output_path):
            successful += 1
        else:
            failed += 1
    
    print("\nConversion complete!")
    print(f"Successfully converted: {successful} files")
    print(f"Failed to convert: {failed} files")
    print(f"\nCSV files have been saved to: {output_folder}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python script.py input_folder output_folder")
    else:
        process_folder(sys.argv[1], sys.argv[2])