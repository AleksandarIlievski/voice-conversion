#!/bin/bash

# Set the base directory
base_dir="/content/vctk/wav48_silence_trimmed"

# Change to the base directory
cd "$base_dir" || exit

# Loop through all subfolders in the wav48 directory
for subfolder in */; do
	subfolder="${subfolder%/}"  # Remove trailing slash
	cd "$subfolder" || continue  # Change to the subfolder

		    # Delete files ending with mic2.flac
	find . -name "*mic2.flac" -type f -exec rm {} +

			    # Change back to the base directory
	cd "$base_dir" || exit
done

echo "Deletion complete."

