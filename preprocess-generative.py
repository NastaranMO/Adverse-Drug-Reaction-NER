import os
import csv
from nltk.tokenize import word_tokenize

# Input text and annotations
# read text file form cadec folder
folder_path = "./cadec 2/text"

filtered_files = []

for file in os.listdir("./cadec 2/original"):
    file_name = os.path.splitext(file)[0].split()[0]
    # check if there is semicolumn 
    with open(os.path.join("./cadec 2/original", file), 'r') as f:
        content = f.read()
        # skip if there is semicolon
        if ";" in content:
            continue
        else:
            filtered_files.append(file_name)

            

# List all .txt files in the folder
# text_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
texts = []
count = 0

# Iterate through each text file and its corresponding annotation file
for text_file in filtered_files:
    # Read the text file
    text_file = f'{text_file}.txt'
    with open(os.path.join(folder_path, text_file), "r") as file:
        text = file.read().replace("\n", " ") 
        example = {
            "text": text,
            "file": os.path.splitext(text_file)[0].split()[0]
        }
        texts.append(example)
# print(texts[1])
# read annotations
tags = ["Symptom", "ADR", "Finding", "Disease", "Drug"]
annotation_all = []
for text in texts:
    file_name = text["file"]
    annotations_file_path = f'/Users/nastaran/Downloads/cadec 2/original/{file_name}.ann'
    # print("********************")
    # print("file_name:",file_name)
    # print(annotations_file_path)
    with open(annotations_file_path, 'r') as file:
        has_semicolon = False
        annotations = file.readlines()
        # if count == 0:
        #     print("annotations:",annotations)
            # print("file_name:",file_name)
        annotations_list = []
        for annotation in annotations:
            parts = annotation.strip().split()
            # print(parts)
            if ";" in parts[3]:
                has_semicolon = True
                break
            if (parts[1] in tags):
                annotations_list.append({
                    "label": parts[1],
                    "start": int(parts[2]),
                    "end": int(parts[3]),
                    "text": " ".join(parts[4:])
                })
        if has_semicolon == False:
            annotation_all.append(annotations_list)


# print(annotation_all[0])
# print(texts[0])

def iob_tagging(text, annotations):
    tokens = word_tokenize(text)
    token_starts = []
    current_pos = 0
    
    # Map token start positions
    for token in tokens:
        start_pos = text.find(token, current_pos)
        token_starts.append(start_pos)
        current_pos = start_pos + len(token)
    
    iob_tags = ["O"] * len(tokens)
    
    for annotation in annotations:
        label = annotation['label']
        start, end = annotation['start'], annotation['end']
        entity_tokens = word_tokenize(text[start:end])
        for i, token in enumerate(entity_tokens):
            # Find the token index in original token list
            try:
                token_index = token_starts.index(start)
                if i == 0:
                    iob_tags[token_index] = f"B-{label}"
                else:
                    iob_tags[token_index] = f"I-{label}"
                start += len(token) + 1  # Update start position for next token
            except ValueError:
                continue  # Skip if token not found (rare case)
    
    # return list(zip(tokens, iob_tags))
    return iob_tags

# Generate IOB tags
count = 0
for text, annotations in zip(texts, annotation_all):
    iob_data = iob_tagging(text["text"], annotations)
    # save to a csv file
    # create a file with text and IOBs in the txt file
    if count == 0:
        print(" ".join(iob_data))
        print(text["text"])
    count += 1
    # create a folder if not exist
    if not os.path.exists(f'./text_annotations'):
        os.makedirs(f'./text_annotations')
    csv_file_path = f'./text_annotations/{text["file"]}.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sentence', 'IOB'])
        writer.writerow([text["text"], " ".join(iob_data)])


