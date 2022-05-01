import argparse
import json
import os.path

import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import Dataset
from transformers import PegasusTokenizerFast, AutoConfig, PegasusForConditionalGeneration, DataCollatorForSeq2Seq, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer

from default_training_args import DEFAULT_TRAINING_ARGUMENTS


class PegasusDataset(Dataset):
    # TODO - extract to another file
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="path/to/checkpoint")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, required=True)
    parser.add_argument("--eval_accumulation_steps", type=int, default=16)

    return parser


def main():
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = create_parser()
    args = parser.parse_args()
    cli_config = vars(args)

    checkpoint_path = cli_config["checkpoint_path"]
    tokenizer_path = os.path.join(checkpoint_path, "tokenizer.json")
    tokenizer_config_path = os.path.join(checkpoint_path, "tokenizer_config.json")

    model_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    # Load the tokenizer
    tokenizer = PegasusTokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)

    # Load the tokenizer config
    config = AutoConfig.from_pretrained(tokenizer_config_path, local_files_only=True)
    tokenizer.config = config

    # Load the model
    model = PegasusForConditionalGeneration\
        .from_pretrained(model_path, local_files_only=True)\
        .to(torch_device)

    dataset = load_dataset("cnn_dailymail", "3.0.0")
    metric = load_metric("rouge")

    test_articles, test_summaries = dataset["test"]["article"], dataset["test"]["highlights"]

    def tokenize_data(articles, summaries):
        article_encodings = tokenizer(articles, truncation=True)
        summary_encodings = tokenizer(summaries, truncation=True)
        dataset_tokenized = PegasusDataset(article_encodings, summary_encodings)
        return dataset_tokenized

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

    test_dataset = tokenize_data(test_articles, test_summaries)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    final_training_args = {k: cli_config.get(k, v) for k, v in DEFAULT_TRAINING_ARGUMENTS.items()}

    training_args = Seq2SeqTrainingArguments(**final_training_args)

    trainer = Seq2SeqTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        # optimizers=(optimizer, scheduler),
        args=training_args,  # training arguments, defined above
        # train_dataset=train_dataset,  # training dataset
        # eval_dataset=test_dataset,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate(
        test_dataset,
        max_length=512,
        num_beams=1
    )

    with open(os.path.join(checkpoint_path, "metrics.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
