#!/bin/bash
# Double-click this file in Finder to start tensorMerger in a Terminal window.
cd "$(dirname "$0")" || exit 1
python3 merge_safetensors.py
echo
read -n 1 -s -r -p "Press any key to close..."
echo
