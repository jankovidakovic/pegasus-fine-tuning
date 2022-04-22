import nltk
import wandb
import torch

from torch.utils.data import Dataset
from transformers import PegasusForConditionalGeneration, Trainer, TrainingArguments, Adafactor, IntervalStrategy, \
    PegasusTokenizerFast, SchedulerType, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from transformers.optimization import AdafactorSchedule
from datasets import load_dataset, load_metric
from transformers.training_args import OptimizerNames


class PegasusDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)


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
    result["rougecomb"] = result["rouge1"] + 2 * result["rouge2"] + result["rougeLsum"]
    return result


if __name__ == "__main__":
    wandb.init(project="pegasus-cnn-fine-tuning", entity="jankovidakovic")

    dataset = load_dataset("cnn_dailymail", "3.0.0")
    metric = load_metric("rouge")

    train_texts, train_labels = dataset['train']['article'][:1000], dataset['train']['highlights'][:1000]
    val_texts, val_labels = dataset["validation"]["article"], dataset["validation"]["highlights"]

    model_name = 'google/pegasus-large'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = "test-results"

    # prepare training data
    tokenizer = PegasusTokenizerFast.from_pretrained(model_name)


    def tokenize_data(texts, labels):
        encodings = tokenizer(texts, truncation=True)  # removed padding=True
        decodings = tokenizer(labels, truncation=True)
        dataset_tokenized = PegasusDataset(encodings, decodings)
        return dataset_tokenized


    train_dataset = tokenize_data(train_texts, train_labels)
    eval_dataset = tokenize_data(val_texts, val_labels)

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

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,  # output directory
        max_steps=2000,
        per_device_train_batch_size=1,  # batch size per device during training, can increase if memory allows
        per_device_eval_batch_size=1,  # batch size for evaluation, can increase if memory allows
        gradient_accumulation_steps=128,
        eval_accumulation_steps=128,
        # save_steps=1000,  # number of updates steps before checkpoint saves
        save_steps=1,
        save_total_limit=1,  # limit the total amount of checkpoints and deletes the older checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="rougecomb",
        evaluation_strategy=IntervalStrategy.STEPS,  # evaluation strategy to adopt during training
        # eval_steps=10000,  # number of update steps before evaluation
        eval_steps=1,
        predict_with_generate=True,
        logging_dir='./test-logs',  # directory for storing logs
        logging_steps=1,
        adafactor=True,
        # learning_rate=5e-4,
        optim=OptimizerNames.ADAFACTOR,
        lr_scheduler_type=SchedulerType.CONSTANT
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
    )

    # return trainer
    trainer.train()
