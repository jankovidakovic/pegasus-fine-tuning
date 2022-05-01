import argparse

import torch
import wandb
from transformers import PegasusTokenizerFast, PegasusForConditionalGeneration, DataCollatorForSeq2Seq, Adafactor, \
    Seq2SeqTrainer, EarlyStoppingCallback
from transformers.optimization import AdafactorSchedule

from utils import get_tokenization_function, get_data, get_training_args, get_metric_function


def create_parser():
    parser = argparse.ArgumentParser(description="Low-resource summarization fine-tuning configuration")

    parser.add_argument("--n_train_examples", type=int)
    parser.add_argument("--n_eval_examples", type=int)

    parser.add_argument("--per_device_train_batch_size", type=int, required=True)
    parser.add_argument("--per_device_eval_batch_size", type=int, required=True)
    parser.add_argument("--total_train_batch_size", type=int, default=256, required=True)

    parser.add_argument("--eval_accumulation_steps", type=int)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--early_stopping", action="store_true", default=False)
    parser.add_argument("--early_stopping_patience", type=int, default=3, required=False)

    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--save_total_limit", type=int)
    parser.add_argument("--eval_steps", type=int)

    parser.add_argument("--output_dir")
    parser.add_argument("--logging_dir")
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--wandb_project")

    return parser


def main():
    # parse arguments
    parser = create_parser()
    args = parser.parse_args()
    config = vars(args)

    wandb.init(project=config["wandb_project"], entity="jankovidakovic")

    model_name = 'google/pegasus-large'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = PegasusTokenizerFast.from_pretrained(model_name)

    train_articles, train_summaries, eval_articles, eval_summaries, _, _ = get_data(config)
    tokenize_data = get_tokenization_function(tokenizer)
    train_dataset = tokenize_data(train_articles, train_summaries)
    eval_dataset = tokenize_data(eval_articles, eval_summaries)

    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model)
    optimizer = Adafactor(
        model.parameters(),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=5e-4
    )
    scheduler = AdafactorSchedule(optimizer, initial_lr=5e-4)

    training_args = get_training_args(config)
    trainer = Seq2SeqTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        optimizers=(optimizer, scheduler),
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_metric_function(tokenizer)
    )

    if config["early_stopping"]:
        early_stopping_patience = config["early_stopping_patience"]
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience))

    trainer.train()


if __name__ == "__main__":
    main()
