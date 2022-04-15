import torch
import transformers
from transformers import PegasusForConditionalGeneration, TrainingArguments, IntervalStrategy, Trainer, PegasusTokenizer

from transformers.optimization import Adafactor, AdafactorSchedule
from datasets import load_metric, load_dataset
import wandb

from pegasus_dataset import PegasusDataset


def prepare_model(*, device: str = "cuda"):
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large").to(device)

    # replace AdamW with Adafactor
    optimizer = Adafactor(
        model.parameters(),
        # scale_parameter=False,
        relative_step=True,
        scale_parameter=False,
        # warmup_init=True,
        warmup_init=False,
    )
    scheduler = AdafactorSchedule(optimizer, initial_lr=5e-5)
    return model, optimizer, scheduler


def prepare_dataset(articles, summaries, *, tokenizer):
    article_encodings = tokenizer(articles, truncation=True, padding=True)
    summary_encodings = tokenizer(summaries, truncation=True, padding=True)

    dataset = PegasusDataset(article_encodings, summary_encodings)
    return dataset


def main():

    # prepare data



    dataset = load_dataset("cnn_dailymail", "3.0.0")  # ["train", "validation", "test"]

    model_name = "google/pegasus-large"

    tokenizer = PegasusTokenizer.from_pretrained(model_name)

    train_dataset = prepare_dataset(
        dataset["train"]["article"],
        dataset["train"]["highlights"],
        tokenizer=tokenizer
    )

    eval_dataset = prepare_dataset(
        dataset["validation"]["article"],
        dataset["validation"]["highlights"],
        tokenizer=tokenizer
    )

    wandb.init(project="pegasus-cnn-fine-tuning", entity="jankovidakovic")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, optimizer, scheduler = prepare_model(device=device)

    output_dir = "./results"

    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        per_device_train_batch_size=1,  # batch size per device during training, can increase if memory allows
        per_device_eval_batch_size=1,  # batch size for evaluation, can increase if memory allows
        save_steps=500,  # number of updates steps before checkpoint saves
        save_total_limit=3,  # limit the total amount of checkpoints and deletes the older checkpoints
        evaluation_strategy=IntervalStrategy.STEPS,  # evaluation strategy to adopt during training
        eval_steps=100,  # number of update steps before evaluation
        # warmup_steps=500,  # number of warmup steps for learning rate scheduler
        # weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
        #####
        label_smoothing_factor=0.1,
        max_steps=170000,
        per_gpu_train_batch_size=1,  # total batch size = 2
        gradient_accumulation_steps=128,   # batch_size_orig / batch_size_possible
        report_to=["wandb"],
    )

    trainer = Trainer(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        model=model,
        # compute_metrics="hehe"  # i suppose computing rouge on every step is overkill
        optimizers=(optimizer, scheduler),
        args=training_args
    )

    trainer.train()


if __name__ == "__main__":
    main()
