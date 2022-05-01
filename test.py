import argparse
import json
import os.path

import torch
from transformers import PegasusTokenizerFast, PegasusForConditionalGeneration, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer

from utils import get_tokenization_function, get_data, get_training_args, get_metric_function


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
    config = vars(args)

    checkpoint_path = config["checkpoint_path"]

    tokenizer = PegasusTokenizerFast.from_pretrained(checkpoint_path, local_files_only=True)

    model = PegasusForConditionalGeneration\
        .from_pretrained(checkpoint_path, local_files_only=True) \
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

    with open(os.path.join(checkpoint_path, "metrics.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
