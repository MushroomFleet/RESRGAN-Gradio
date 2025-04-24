import os
import re

def rename_files_in_directory(directory):
    """Rename all files in a directory, replacing spaces with underscores."""
    # Get list of files in directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Create a mapping of old to new filenames
    filename_mapping = {}
    
    # Rename files
    for old_filename in files:
        # Replace spaces with underscores
        new_filename = old_filename.replace(' ', '_')
        
        if old_filename != new_filename:
            old_path = os.path.join(directory, old_filename)
            new_path = os.path.join(directory, new_filename)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_filename} -> {new_filename}")
                filename_mapping[old_filename] = new_filename
            except Exception as e:
                print(f"Error renaming {old_filename}: {e}")
    
    return filename_mapping

def update_meta_info_file(meta_info_path, input_mapping, processed_mapping):
    """Update meta_info file with new filenames."""
    try:
        # Read the meta_info file
        with open(meta_info_path, 'r') as f:
            lines = f.readlines()
        
        # Update filenames in the meta_info file
        new_lines = []
        for line in lines:
            new_line = line.strip()
            
            # Check if line starts with input_images or processed_images
            if new_line.startswith('input_images/'):
                # Extract the filename
                filename = new_line.replace('input_images/', '')
                # Find the new filename if it exists in the mapping
                for old_name, new_name in input_mapping.items():
                    if filename == old_name:
                        new_line = f"input_images/{new_name}"
                        break
            elif new_line.startswith('processed_images/'):
                # Extract the filename
                filename = new_line.replace('processed_images/', '')
                # Find the new filename if it exists in the mapping
                for old_name, new_name in processed_mapping.items():
                    if filename == old_name:
                        new_line = f"processed_images/{new_name}"
                        break
            
            new_lines.append(new_line)
        
        # Write the updated meta_info file
        with open(meta_info_path, 'w') as f:
            f.write('\n'.join(new_lines))
        
        print(f"Updated meta_info file: {meta_info_path}")
    except Exception as e:
        print(f"Error updating meta_info file: {e}")

if __name__ == "__main__":
    # Base directory
    base_dir = "datasets/evalset"
    
    # Rename files in input_images directory
    input_dir = os.path.join(base_dir, "input_images")
    print(f"Renaming files in {input_dir}...")
    input_mapping = rename_files_in_directory(input_dir)
    
    # Rename files in processed_images directory
    processed_dir = os.path.join(base_dir, "processed_images")
    print(f"Renaming files in {processed_dir}...")
    processed_mapping = rename_files_in_directory(processed_dir)
    
    # Update meta_info file
    meta_info_path = os.path.join(base_dir, "meta_info", "meta_info_evalset_fixed.txt")
    print(f"Updating meta_info file: {meta_info_path}...")
    update_meta_info_file(meta_info_path, input_mapping, processed_mapping)
    
    print("Done!")
