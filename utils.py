import matplotlib.pyplot as plt
import ast
import pandas as pd
from collections import defaultdict, Counter
from wordcloud import WordCloud
import argparse
VOCABULARY = {}


def parse_arguments():
    """
    Available arguments:

    """
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--mode",
        default="train",
        help="Choose the mode to run"
    )
    # parser.add_argument(
    #     "--mode",
    #     default="average",
    #     help="Choose the mode to run"
    # )
    parser.add_argument(
        "--model",
        default="bert-base-cased",
        help="Choose the model to run"
    )
    parser.add_argument(
        "--analysis",
        default=False,
        help="Run data analysis"
    )
    return parser.parse_args()

def get_all_tokens(dataset):
    all_tokens = {
        "train": [],
        "test": [],
        "validation": []
    }
    for split in all_tokens.keys():
        for item in dataset[split]:
            for token in item["tokens"]:
                all_tokens[split].append(token)
    return all_tokens

def get_vocabulary_size(dataset):
    # Get all tokens from the dataset
    all_tokens = get_all_tokens(dataset)
    token_split_size = {split: len(tokens) for split, tokens in all_tokens.items()}
    for split, tokens in all_tokens.items():
        VOCABULARY[split] = set(tokens)
    split_size = {split: len(tokens) for split, tokens in VOCABULARY.items()}
    return split_size, token_split_size

def get_ner_tags_count_per_split(NER_TAGS, dataset):
    splits = ["train", "test", "validation"]
    ner_tags_count_per_split = {split: {tag: 0 for tag in NER_TAGS} for split in splits}
    ner_tags_count = {tag: 0 for tag in NER_TAGS}
    
    for split in splits:
        for item in dataset[split]:
            for ner in item["ner_tags"]:
                ner_tag = NER_TAGS[ner]
                ner_tags_count_per_split[split][ner_tag] += 1
                
    return ner_tags_count_per_split

def plot_distribution(df, verbose=False):
    if verbose:
        ner_tags_count = df
        dist_df = pd.DataFrame(ner_tags_count)
        dist_df.to_csv("ner_tags_count_per_split.csv", index=True)
        dist_df.plot(kind="bar", width=0.8)
        plt.title(
            "Distribution of Named Entity Tag Across Train, Validation, and Test Sets"
        )
        plt.xlabel("Named Entity Tag")
        plt.ylabel("Percentage")
        plt.xticks(rotation=45, ha='right', fontsize=10)  
        plt.legend(title="Dataset")
        plt.tight_layout()
        plt.savefig("ner_distribution.png")

def get_top_entities(dataset, top_n=5):
    categorized_entities = {split: defaultdict(Counter) for split in ["train", "test", "validation"]}
    
    for split in ["train", "test", "validation"]:
        for item in dataset[split]:
            info = item["info"]
            for entity_info in info:
                try:
                    entity_dict = ast.literal_eval(entity_info)
                    original_text = entity_dict.get("original", "Unknown")
                    ner_tag = entity_dict.get("ner_tag", "Unknown")
                    categorized_entities[split][ner_tag][original_text] += 1
                except (ValueError, SyntaxError) as e:
                    print(f"Error: {entity_info} - {e}")
    
    for split, tag_dict in categorized_entities.items():
        rows = []
        for ner_tag, entity_counter in tag_dict.items():
            top_entities = entity_counter.most_common(top_n)
            for entity, count in top_entities:
                rows.append({"Split": split, "NER Tag": ner_tag, "Entity": entity, "Count": count})
        df = pd.DataFrame(rows)
        df.to_csv(f"top_entities_by_category_{split}.csv", index=False)
    
    return categorized_entities

def create_wordcloud():
    df = pd.read_csv("./top_entities_by_category_train.csv")
    filtered_df = df[df["NER Tag"] == "ADR"]
    term_frequencies = dict(zip(filtered_df["Entity"], filtered_df["Count"]))

    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        max_words=100,
        prefer_horizontal=0.7
    ).generate_from_frequencies(term_frequencies)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of Most Frequent ADR-Related Terms", fontsize=16)
    plt.tight_layout(pad=0)
    plt.savefig("wordcloud.png", dpi=300, bbox_inches='tight')
