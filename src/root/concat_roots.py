"""
concat_roots.py

Script to concatenate multiple ROOT files into a single ROOT file.

Usage:
    python concat_roots.py input_dir [outfile.root]
"""

import argparse

import ROOT
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input_dir', help='Base path containing the ROOT files')
parser.add_argument('output', nargs='?', default='combined_files.root',
                    help='Output file (default: <input_path>/combined_files.root)')
args = parser.parse_args()

# Convert input_path to properly formatted path
input_dir = os.path.abspath(args.input_dir)
# Same for output, if provided
if args.output: output_path = os.path.abspath(args.output)

# Make sure the input path exists
if not os.path.exists(input_dir):
    print(f'[ERROR] Input path {input_dir} does not exist.')
    exit(1)
# Output file should be written to the input path if not specified
output_path = os.path.join(input_dir, os.path.dirname(output_path))
if not os.path.exists(output_path):
    print(f'[INFO] Creating output directory {output_path}.')
    os.makedirs(output_path, exist_ok=True)

# Now find all ROOT files in this path and its subdirectories
infiles = []
print(f'[INFO] Searching for ROOT files...')
for root, _, files in os.walk(input_dir):
    for file in files:
        # Only add ROOT files that are not the output file (to avoid recursion)
        if file.endswith('.root') and not file.startswith('combined_files'):
            infiles.append(os.path.join(root, file))

print(f'[INFO] Concatenating files:')
for infile in infiles:
    print(f'[INFO]  - {infile}')
print(f'[INFO] Into single output file: {output_path}')

# Create a TFileMerger to merge the files
merger = ROOT.TFileMerger()
merger.OutputFile(output_path)
# Open each input file and add it to the list
for infile in infiles:
    print(f'[INFO] Adding {infile}...')
    # Add file to merger by name; TFileMerger manages opening/closing it
    merger.AddFile(infile)
# Use ROOT's TFileMerger to merge the files
merger.Merge()
print(f'[INFO] Successfully created {output_path}.')
