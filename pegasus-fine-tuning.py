import wandb
import torch

from torch.utils.data import Dataset
from transformers import PegasusForConditionalGeneration, Trainer, TrainingArguments, Adafactor, IntervalStrategy, \
    PegasusTokenizerFast, PegasusTokenizer
from transformers.optimization import AdafactorSchedule
from datasets import load_dataset


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


def prepare_data(model_name,
                 train_texts, train_labels,
                 val_texts=None, val_labels=None):
    """
    Prepare input data for model fine-tuning
    """
    tokenizer = PegasusTokenizerFast.from_pretrained(model_name)
    # tokenizer = PegasusTokenizer.from_pretrained(model_name)

    prepare_val = False if val_texts is None or val_labels is None else True

    def tokenize_data(texts, labels):
        encodings = tokenizer(texts, truncation=True, padding=True)
        decodings = tokenizer(labels, truncation=True, padding=True)
        dataset_tokenized = PegasusDataset(encodings, decodings)
        return dataset_tokenized

    train_dataset = tokenize_data(train_texts, train_labels)
    val_dataset = tokenize_data(val_texts, val_labels) if prepare_val else None

    return train_dataset, val_dataset, tokenizer


def prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset=None,
                        output_dir='./results'):
    """
    Prepare configurations and base model for fine-tuning
    """
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    optimizer = Adafactor(
        model.parameters(),
        # scale_parameter=False,
        relative_step=True,
        scale_parameter=False,
        # warmup_init=True,
        warmup_init=False,
        # lr=5e-5
    )
    scheduler = AdafactorSchedule(optimizer, initial_lr=1e-2)

    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        max_steps=170000,
        per_device_train_batch_size=1,  # batch size per device during training, can increase if memory allows
        per_device_eval_batch_size=1,  # batch size for evaluation, can increase if memory allows
        gradient_accumulation_steps=128,
        eval_accumulation_steps=128,
        save_steps=1000,  # number of updates steps before checkpoint saves
        save_total_limit=5,  # limit the total amount of checkpoints and deletes the older checkpoints
        evaluation_strategy=IntervalStrategy.STEPS,  # evaluation strategy to adopt during training
        eval_steps=10000,  # number of update steps before evaluation
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
        # optim=OptimizerNames.ADAFACTOR
        # lr_scheduler_type=SchedulerType.POLYNOMIAL
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        optimizers=(optimizer, scheduler),
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        tokenizer=tokenizer
    )

    return trainer


if __name__ == "__main__":
    wandb.init(project="pegasus-cnn-fine-tuning", entity="jankovidakovic")

    dataset = load_dataset("cnn_dailymail", "3.0.0")
    train_texts, train_labels = dataset['train']['article'], dataset['train']['highlights']
    val_texts, val_labels = dataset["validation"]["article"], dataset["validation"]["highlights"]

    model_name = 'google/pegasus-large'
    train_dataset, eval_dataset, tokenizer = prepare_data(model_name, train_texts, train_labels, val_texts, val_labels)

    trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset, eval_dataset)
    trainer.train()
