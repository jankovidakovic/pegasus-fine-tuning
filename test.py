import argparse
import json
import os.path

import torch
from transformers import PegasusTokenizerFast, PegasusForConditionalGeneration, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer

from utils import get_tokenization_function, get_data, get_training_args, get_metric_function


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, required=True)
    parser.add_argument("--eval_accumulation_steps", type=int, default=16)

    return parser


def main():
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = create_parser()
    args = parser.parse_args()
    config = vars(args)

    if config.get("zero_shot", False):
        checkpoint_path = config["checkpoint_path"]
        model_args = {
            "pretrained_model_name_or_path": checkpoint_path,
            "local_files_only": True
        }
        save_path = checkpoint_path
    else:
        model_args = {
            "pretrained_model_name_or_path": "google/pegasus-large"
        }
        save_path = "./zero_shot"

    # tokenizer = PegasusTokenizerFast.from_pretrained(checkpoint_path, local_files_only=True)
    tokenizer = PegasusTokenizerFast.from_pretrained(**model_args)

    model = PegasusForConditionalGeneration\
        .from_pretrained(**model_args) \
        .to(torch_device)

    _, _, _, _, test_articles, test_summaries = get_data(config)
    tokenize_data = get_tokenization_function(tokenizer)
    test_dataset = tokenize_data(test_articles, test_summaries)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = get_training_args(config)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_metric_function(tokenizer),
    )

    metrics = trainer.evaluate(
        test_dataset,
        max_length=512,
        num_beams=1
    )

    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
