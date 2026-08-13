"""
concat_roots.py

Script to concatenate multiple ROOT files into a single ROOT file.

TODO: Remove user input and instead accept a command-line argument for the input
path. Also add optional output file name and path. By default the output file
should go into the shared directory and be named "combined_files.root".

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
parser.add_argument('--subdirs', nargs='*', default=['magup', 'magdown'],
                    help='Subdirectories to search (default: magup magdown)')
args = parser.parse_args()

# Convert input_path to properly formatted path
input_path = os.path.abspath(args.input_path)
# Same for output, if provided
if args.output:
    output_path = os.path.abspath(args.output)
subdirs = args.subdirs

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
# Search the magup and magdown subdirectories within the provided path
print(f'[INFO] Searching for ROOT files...')
for subdir in ['magup', 'magdown']:
    dir = os.path.join(input_path, subdir)
    if not os.path.exists(dir):
        print(f'[WARNING] Subdirectory {dir} does not exist. Skipping.')
        continue
    for root, _, files in os.walk(dir):
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
