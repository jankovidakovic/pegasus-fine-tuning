import nltk
import numpy as np
import torch
from datasets import load_metric, load_dataset
from transformers import Seq2SeqTrainingArguments

from constants import DEFAULT_TRAINING_ARGS
from pegasus_dataset import PegasusDataset


def get_tokenization_function(tokenizer):
    def tokenize_data(articles, summaries):
        article_encodings = tokenizer(articles, truncation=True)
        summary_encodings = tokenizer(summaries, truncation=True)
        dataset_tokenized = PegasusDataset(article_encodings, summary_encodings)
        return dataset_tokenized

    return tokenize_data


def get_metric_function(tokenizer):
    metric = load_metric("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                                use_stemmer=True)  # not sure if using stemmer is correct here

        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # combine rouge scores for model comparison
        result["rougecomb"] = result["rouge1"] + 2 * result["rouge2"] + result["rougeLsum"]
        return result

    return compute_metrics


def get_data(config):
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    train_articles, train_summaries = dataset['train']['article'], dataset['train']['highlights']
    eval_articles, eval_summaries = dataset["validation"]["article"], dataset["validation"]["highlights"]

    if "n_train_examples" in config:
        train_cutoff = config["n_train_examples"]
        train_articles, train_summaries = train_articles[:train_cutoff], train_summaries[:train_cutoff]

    if "n_eval_examples" in config:
        eval_cutoff = config["n_eval_examples"]
        eval_articles, eval_summaries = eval_articles[:eval_cutoff], eval_summaries[:eval_cutoff]

    test_articles, test_summaries = dataset["test"]["article"], dataset["test"]["highlights"]

    return train_articles, train_summaries, eval_articles, eval_summaries, test_articles, test_summaries


def get_training_args(config):
    training_args_config = {k: config.get(k, v) for k, v in DEFAULT_TRAINING_ARGS.items()}

    device_count = torch.cuda.device_count()
    training_args_config["gradient_accumulation_steps"] = \
        int(config["total_train_batch_size"] / config["per_device_train_batch_size"] / device_count)
    training_args = Seq2SeqTrainingArguments(**training_args_config)
    return training_args
