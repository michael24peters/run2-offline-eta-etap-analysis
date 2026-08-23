"""
concat_roots.py

Script to concatenate multiple ROOT files into a single ROOT file.

Usage:
    python concat_roots.py input_path [output.root]
"""

import argparse

import ROOT
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input_path', help='Base path containing the ROOT files')
parser.add_argument('output', nargs='?', default=None,
                    help='Output file (default: <input_path>/combined_files.root)')
args = parser.parse_args()

# Convert input_path to properly formatted path
input_path = os.path.abspath(args.input_path)
# Same for output, if provided
if args.output:
    output_path = os.path.abspath(args.output)

# Make sure the input path exists
if not os.path.exists(input_path):
    print(f'[ERROR] Input path {input_path} does not exist.')
    exit(1)
# Create output path if it doesn't exist
if args.output:
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        print(f'[INFO] Creating output directory {output_dir}.')
        os.makedirs(output_dir)

# Now find all ROOT files in this path and its subdirectories
infiles = []
print(f'[INFO] Searching for ROOT files...')
for root, _, files in os.walk(input_path):
    for file in files:
        if file.endswith('.root'):
            infiles.append(os.path.join(root, file))
# Write outfile to base path
outfile = os.path.join(input_path, 'combined_files.root')

print(f'[INFO] Concatenating files:')
for infile in infiles:
    print(f'[INFO]  - {infile}')
print(f'[INFO] Into single output file: {outfile}')

# Create a TFileMerger to merge the files
merger = ROOT.TFileMerger()
merger.OutputFile(outfile)
# Open each input file and add it to the list
for infile in infiles:
    print(f'[INFO] Adding {infile}...')
    # Add file to merger by name; TFileMerger manages opening/closing it
    merger.AddFile(infile)
# Use ROOT's TFileMerger to merge the files
merger.Merge()
print(f'[INFO] Successfully created {outfile}.')
