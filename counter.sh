#!/bin/bash

# Define the target directory (default is current directory)
TARGET_DIR=${1:-.}

# Temporary file to store results
temp_file=$(mktemp)

# Find all files in the directory and subdirectories
find "$TARGET_DIR" -type f | while read -r file; do
    # Extract the file extension
    ext="${file##*.}"
    
    # If there's no extension, label it as "no_extension"
    if [ "$ext" = "$file" ]; then
        ext="no_extension"
    fi

    # Get the file size in bytes and append to temporary file
    size=$(stat --printf="%s" "$file")
    echo "$ext $size" >> "$temp_file"

done

# Summarize sizes by extension
awk '{sizes[$1]+=$2} END {for (ext in sizes) printf "%-15s %10.2f MB\n", ext, sizes[ext]/(1024*1024)}' "$temp_file" | sort -k2 -nr

# Clean up temporary file
rm -f "$temp_file"
