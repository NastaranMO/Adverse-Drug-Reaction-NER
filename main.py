from datasets import load_dataset
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForTokenClassification
import evaluate
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import pprint
import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report
from utils import plot_distribution, get_all_tokens, get_vocabulary_size, get_ner_tags_count_per_split,get_top_entities, create_wordcloud, parse_arguments

args = parse_arguments()
tokenizer = AutoTokenizer.from_pretrained(args.model)
dataset = load_dataset("mireiaplalis/processed_cadec")

np.random.seed(42)
metric = evaluate.load("seqeval")

NER_TAGS = dataset["train"].features["ner_tags"].feature.names # ['O', 'B-ADR', 'I-ADR', 'B-Drug', 'I-Drug', 'B-Disease', 'I-Disease', 'B-Symptom', 'I-Symptom', 'B-Finding', 'I-Finding']


def get_ner_tags_percentages(ner_tags_count_per_split):
    """
    Calculate the percentage of each named entity tag in each split of the dataset.

    Args:
        ner_tags_count_per_split (dict): A dictionary containing the count of each named entity tag in each split of the dataset.

    Returns:
        dict: A dictionary containing the percentage of each named entity tag in each split of the dataset.
    """
    for split in ner_tags_count_per_split.keys():
        total = sum(ner_tags_count_per_split[split].values())
        for tag in ner_tags_count_per_split[split].keys():
            ner_tags_count_per_split[split][tag] = round((ner_tags_count_per_split[split][tag] / total) * 100, 2)
    return ner_tags_count_per_split

def get_statistics():
    """
    Calculate the vocabulary size and token size of the dataset.

    Args:
        dataset (dict): A dictionary containing the train, test, and validation splits of the dataset.

    Returns:
        None
    """
    vocabulary_split_size, token_split_size = get_vocabulary_size(dataset) # [6089, 1880, 1861], {'train': 85824, 'test': 9629, 'validation': 10211}
    vocabulary_size = sum(size for split, size in vocabulary_split_size.items()) #9830
    token_size = sum(size for split, size in token_split_size.items()) #105,664
    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Token size: {token_size}")

def data_analysis():
    """
    Perform data analysis on the dataset.

    Args:
        dataset (dict): A dictionary containing the train, test, and validation splits of the dataset.

    Returns:
        None
    """
    get_statistics()
    ner_tags_count_df = get_ner_tags_count_per_split(NER_TAGS, dataset)
    plot_distribution(ner_tags_count_df, True)
    # print(ner_tags_count_df)
    ner_tags_percentages_df = get_ner_tags_percentages(ner_tags_count_df)
    # pprint.pprint(ner_tags_percentages_df)
    top_entities = get_top_entities(dataset, top_n=10)
    create_wordcloud()

def align_labels_with_tokens(labels, word_ids):
    """
    For a given set of labels, align the labels with the tokens they correspond to.

    Args:
        labels (list): A list of labels, where each label is a token.
        word_ids (list): A list of word ids, where each word id corresponds to a token.

    Returns:
        list: A list of aligned labels.
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    """
    Tokenize the input text and align the labels with the tokens.

    Args:
        examples (dict): A dictionary containing the input text and labels.

    Returns:
        dict: A dictionary containing the tokenized input text and aligned labels.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=512
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])

def compute_metrics(eval_preds, dataset_split="validation"):
    """
    Compute metrics for the evaluation of a model.

    Args:
        eval_preds (_type_): _description_
        dataset_split (str, optional): _description_. Defaults to "validation".

    Returns:
        _type_: _description_
    """
    print("Computing metrics...")
    label_names = NER_TAGS
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    classification_details_sklearn = classification_report(
        [label for sublist in true_labels for label in sublist],
        [pred for sublist in true_predictions for pred in sublist],
        output_dict=True,
        zero_division=0
    )
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    metrics_dic =  {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
        "dataset": dataset_split
    }

    metrics_df = pd.DataFrame(metrics_dic, index=[0])
    sklearn_metrics_df = pd.DataFrame(classification_details_sklearn).transpose()
    metrics_df = pd.concat([metrics_df, sklearn_metrics_df], axis=1)
    csv_path = f'./metrics/{args.mode.replace("/", "_")}_metrics.csv'
    metrics_df.to_csv(csv_path, mode='a', header=True)

    return metrics_dic


id2label = {i: label for i, label in enumerate(NER_TAGS)}
label2id = {v: k for k, v in id2label.items()}

def evaluate_test_set(model_path):
    """
    Evaluate the model on the test set.

    Args:
        model_path (_type_): _description_

    Returns:
        _type_: _description_  
    """
    final_model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        id2label=id2label,
        label2id=label2id,
    )
    
    # Create trainer for evaluation
    test_trainer = Trainer(
        model=final_model,
        args=TrainingArguments(
            output_dir="./results",
            do_train=False,
            do_eval=True,
        ),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, dataset_split="test")
    )
    
    # Evaluate on test set
    test_results = test_trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print(f"Final test set results: {test_results}")
    
    # Save test results with special marker
    # metrics_df = pd.DataFrame([test_results])
    # os.makedirs('metrics', exist_ok=True)
    # csv_path = f'./metrics/{args.model.replace("/", "_")}_test_results.csv'
    # metrics_df.to_csv(csv_path, index=False)

def train_baseline_model(model):
    """
    Train a baseline model on the training set.

    Args:
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    print("Training baseline model...")
    args = TrainingArguments(
        output_dir="./results",
        eval_strategy="no",
        do_train=True
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model("./baseline_model")

def train_model(model_name,run=False, baseline=False):
    """
    Train a model on the training set.

    Args:
        model_name (str): The name of the model to train.
        run (bool, optional): Whether to run the model. Defaults to False.
        baseline (bool, optional): Whether to train a baseline model. Defaults to False.

    Returns:
        _type_: _description_
    """
    if run:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        # Creat a baseline model and train the model on the training set and don't evaluate on the validation set
        if baseline:
            train_baseline_model(model)
            print("Training baseline model...")
        # Training the model
        print("Training model..."+model_name)

        args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            do_train=True,
            lr_scheduler_type="linear",
            # gradient_accumulation_steps=2,
            # per_device_train_batch_size=16,
            # per_device_eval_batch_size=16,
            # learning_rate=2e-5,
            warmup_steps=100,
            seed=42,
            num_train_epochs=10,
            weight_decay=0.01,
            load_best_model_at_end = True
        )
        # default I used
        #         args = TrainingArguments(
        #     output_dir="./results",
        #     eval_strategy="epoch",
        #     save_strategy="epoch",
        #     do_train=True,
        #     lr_scheduler_type="linear",
        #     # warmup_steps=100,
        #     # learning_rate=1e-05,
        #     seed=42,
        #     num_train_epochs=10,
        #     weight_decay=0.01,
        #     load_best_model_at_end = True
        # )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_preds: compute_metrics(eval_preds, dataset_split="validation"),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)]
        )
        trainer.train()
        trainer.save_model(f"./{model_name}_model")
        evaluate_test_set(f"./{model_name}_model")

if __name__ == "__main__":
    run=False
    if args.analysis:
        data_analysis() # Run data analysis
    if args.mode == "train":
        run = True
        train_model(args.model, run, baseline=False)
    # Average over 5 runs to ensure that the model is stable and the result is consistent
    if args.mode == "average":
        all_metrics = []
        for seed in range(5):
            print(f"Running with seed {seed}...")
            metrics = train_model(model_name="bert-base-uncased", run=True, random_seed=seed)
            all_metrics.append(metrics)
        metrics_df = pd.DataFrame(all_metrics)
        # save the average metrics in csv file
        metrics_df.to_csv("average_metrics.csv", index=False)
        average_metrics = metrics_df.drop(columns=["seed"]).mean().to_dict()
    




