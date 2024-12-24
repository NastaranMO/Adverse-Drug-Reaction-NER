import os
import csv
from pathlib import Path

# Define paths
text_folder = '/Users/nastaran/Downloads/cadec 2/text'
annotation_folder = '/Users/nastaran/Downloads/cadec 2/original'
csv_output_folder = '/Users/nastaran/Documents/Dev/TM-A3/csv_output'

# Create output directory if it doesn't exist
os.makedirs(csv_output_folder, exist_ok=True)

# Function to read text files and save as CSV
def process_text_files():
    for text_file in Path(text_folder).glob('*.txt'):
        with open(text_file, 'r') as file:
            content = file.read().strip()
        csv_file_path = Path(csv_output_folder) / f'{text_file.stem}.csv'
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Symptom'])
            writer.writerow([content])

# Function to convert text to IOB format
def convert_to_iob(text, annotations):
    words = text.split()
    iob_tags = ['O'] * len(words)
    for annotation in annotations:
        parts = annotation.strip().split('\t')
        if len(parts) < 3:
            continue
        _, tag, span = parts
        span_info = span.split()
        if len(span_info) < 2:
            continue
        try:
            start, end = map(int, span_info[0].split(';'))
        except ValueError:
            continue
        annotation_text = text[start:end]
        annotation_words = annotation_text.split()
        if not annotation_words:
            continue
        start_idx = len(text[:start].split())
        if start_idx < len(words):
            iob_tags[start_idx] = f'B-{tag}'
            for i in range(1, len(annotation_words)):
                if start_idx + i < len(iob_tags):
                    iob_tags[start_idx + i] = f'I-{tag}'
    return iob_tags

# Function to read annotation files and convert to IOB format
def process_annotation_files():
    for ann_file in Path(annotation_folder).glob('*.ann'):
        text_file_path = Path(text_folder) / f'{ann_file.stem}.txt'
        if not text_file_path.exists():
            continue
        with open(text_file_path, 'r') as file:
            text = file.read()
        with open(ann_file, 'r') as file:
            annotations = file.readlines()
        iob_tags = convert_to_iob(text, annotations)
        csv_file_path = Path(csv_output_folder) / f'{ann_file.stem}_iob.csv'
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Word', 'IOB'])
            for word, tag in zip(text.split(), iob_tags):
                writer.writerow([word, tag])

# Process files
process_text_files()
process_annotation_files()
