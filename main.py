"""Script for fine-tuning Pegasus
Example usage:
  # use XSum dataset as example, with first 1000 docs as training data
  from datasets import load_dataset
  dataset = load_dataset("xsum")
  train_texts, train_labels = dataset['train']['document'][:1000], dataset['train']['summary'][:1000]

  # use Pegasus Large model as base for fine-tuning
  model_name = 'google/pegasus-large'
  train_dataset, _, _, tokenizer = prepare_data(model_name, train_texts, train_labels)
  trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset)
  trainer.train()

Reference:
  https://huggingface.co/transformers/master/custom_datasets.html

"""


from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments, IntervalStrategy
import torch
from transformers.trainer_utils import EvaluationStrategy

from pegasus_dataset import PegasusDataset
from datasets import load_dataset


def prepare_data(model_name: str,
                 train_articles, train_summaries,
                 val_articles=None, val_summaries=None,
                 test_articles=None, test_summaries=None):
    # TODO - refactor, cleanup, docs

    # this should be a sentencepiece tokenizer, right?
    tokenizer = PegasusTokenizer.from_pretrained(model_name)

    def tokenize_data(articles, summaries):
        article_encodings = tokenizer(articles, truncation=True, padding=True)
        summary_encodings = tokenizer(summaries, truncation=True, padding=True)
        dataset_tokenized = PegasusDataset(article_encodings, summary_encodings)
        return dataset_tokenized

    train_dataset = tokenize_data(train_articles, train_summaries)
    if val_articles is not None:
        val_dataset = tokenize_data(val_articles, val_summaries)

    if test_articles is not None:
        test_dataset = tokenize_data(test_articles, test_summaries)

    return train_dataset, val_dataset, test_dataset, tokenizer


def prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset=None, freeze_encoder=False,
                        output_dir='./results'):
    """
    Prepare configurations and base model for fine-tuning
    """
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    # TODO - do i really need this for now?
    if freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        per_device_train_batch_size=1,  # batch size per device during training, can increase if memory allows
        per_device_eval_batch_size=1,  # batch size for evaluation, can increase if memory allows
        save_steps=500,  # number of updates steps before checkpoint saves
        save_total_limit=2,  # limit the total amount of checkpoints and deletes the older checkpoints
        evaluation_strategy=IntervalStrategy.STEPS,  # evaluation strategy to adopt during training
        eval_steps=100,  # number of update steps before evaluation
        # warmup_steps=500,  # number of warmup steps for learning rate scheduler
        # weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
        #####
        label_smoothing_factor=0.1,
        max_steps=170000,
        per_gpu_train_batch_size=2,  # total batch size = 4
        gradient_accumulation_steps=64,   # batch_size_orig / batch_size_possible
        report_to=["wandb"],
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        tokenizer=tokenizer,
    )

    return trainer


if __name__ == '__main__':
    # use XSum dataset as example, with first 1000 docs as training data

    # dataset = load_dataset("xsum")
    # dataset = load_dataset("")
    train_documents, train_summaries = dataset['train']['document'][:1000], dataset['train']['summary'][:1000]

    # use Pegasus Large model as base for fine-tuning
    model_name = 'google/pegasus-large'
    train_dataset, _, _, tokenizer = prepare_data(model_name, train_documents, train_summaries)
    trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset)
    trainer.train()