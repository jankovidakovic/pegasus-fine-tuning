import nltk
import wandb
import torch
import argparse

from torch.utils.data import Dataset
from transformers import PegasusForConditionalGeneration, Adafactor, IntervalStrategy, \
    PegasusTokenizerFast, SchedulerType, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers.optimization import AdafactorSchedule
from datasets import load_dataset, load_metric
from transformers.training_args import OptimizerNames


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


"""
        output_dir=output_dir,  # output directory
        per_device_train_batch_size=1,  # batch size per device during training, can increase if memory allows
        per_device_eval_batch_size=1,  # batch size for evaluation, can increase if memory allows
        gradient_accumulation_steps=128,
        eval_accumulation_steps=128,
        save_steps=1,
        save_total_limit=1,  # limit the total amount of checkpoints and deletes the older checkpoint
        eval_steps=1,
        logging_dir='./test-logs',  # directory for storing logs
        logging_steps=1,
    )

    trainer = Seq2SeqTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        optimizers=(optimizer, scheduler),
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
"""


def create_parser():
    parser = argparse.ArgumentParser(description="Low-resource summarization fine-tuning configuration")

    parser.add_argument("--n_train_examples", type=int)
    parser.add_argument("--n_eval_examples", type=int)
    parser.add_argument("--per_device_train_batch_size", type=int, required=True)
    parser.add_argument("--per_device_eval_batch_size", type=int, required=True)
    parser.add_argument("--total_train_batch_size", type=int, default=256, required=True)
    parser.add_argument("--eval_accumulation_steps", type=int, default=256, required=True)
    parser.add_argument("--gradient_checkpointing", type=bool, action="store_true", required=False, default=False)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--save_total_limit", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--logging_dir")
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--wandb_project")
    parser.add_argument("--max_steps", type=int)

    return parser


DEFAULT_TRAINING_ARGUMENTS = {
    "output_dir": "./results",  # output directory
    "max_steps": 2000,  # max number of gradient updates

    "per_device_train_batch_size": 1,  # batch size per device during training, can increase if memory allows
    "gradient_accumulation_steps": 128,
    "gradient_checkpointing": False,

    "per_device_eval_batch_size": 1,  # batch size for evaluation, can increase if memory allows
    "eval_accumulation_steps": 128,

    "evaluation_strategy": IntervalStrategy.STEPS,  # evaluation strategy to adopt during training
    "eval_steps": 100,
    "predict_with_generate": True,
    "generation_num_beams": 1,

    "save_steps": 100,
    "save_total_limit": 3,  # limit the total amount of checkpoints and deletes the older checkpoint
    "load_best_model_at_end": True,
    "metric_for_best_model": "rougecomb",

    "logging_dir": './logs',  # directory for storing logs
    "logging_steps": 10,

    "adafactor": True,
    "optim": OptimizerNames.ADAFACTOR,
    "lr_scheduler_type": SchedulerType.CONSTANT,
    "label_smoothing_factor": 0.1
}


def main():
    parser = create_parser()
    args = parser.parse_args()
    cli_config = vars(args)

    wandb.init(project=cli_config["wandb_project"], entity="jankovidakovic")

    dataset = load_dataset("cnn_dailymail", "3.0.0")
    metric = load_metric("rouge")

    train_articles, train_summaries = dataset['train']['article'], dataset['train']['highlights']
    eval_articles, eval_summaries = dataset["validation"]["article"], dataset["validation"]["highlights"]

    train_cutoff, eval_cutoff = cli_config["n_train_examples"], cli_config["n_eval_examples"]
    train_articles, train_summaries = train_articles[:train_cutoff], train_summaries[:train_cutoff]
    eval_articles, eval_summaries = eval_articles[:eval_cutoff], eval_summaries[:eval_cutoff]

    model_name = 'google/pegasus-large'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # prepare training data
    tokenizer = PegasusTokenizerFast.from_pretrained(model_name)

    def tokenize_data(articles, summaries):
        article_encodings = tokenizer(articles, truncation=True)
        summary_encodings = tokenizer(summaries, truncation=True)
        dataset_tokenized = PegasusDataset(article_encodings, summary_encodings)
        return dataset_tokenized

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
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

    train_dataset = tokenize_data(train_articles, train_summaries)
    eval_dataset = tokenize_data(eval_articles, eval_summaries)

    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    optimizer = Adafactor(
        model.parameters(),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=5e-4
    )

    scheduler = AdafactorSchedule(optimizer, initial_lr=5e-4)

    final_training_args = {k: cli_config.get(k, v) for k, v in DEFAULT_TRAINING_ARGUMENTS.items()}

    device_count = torch.cuda.device_count()
    final_training_args["gradient_accumulation_steps"] = \
        int(cli_config["total_train_batch_size"] / cli_config["per_device_train_batch_size"] / device_count)
    # final_training_args["eval_gradient_accumulation_steps"] = \
    #    cli_config["total_train_batch_size"] / cli_config["per_device_train_batch_size"] / device_count

    training_args = Seq2SeqTrainingArguments(**final_training_args)

    trainer = Seq2SeqTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        optimizers=(optimizer, scheduler),
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # return trainer
    trainer.train()


if __name__ == "__main__":
    main()
