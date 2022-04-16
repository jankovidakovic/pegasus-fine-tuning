import torch
from torch.utils.data import Dataset
from transformers import PegasusForConditionalGeneration, Adafactor
from transformers.optimization import AdafactorSchedule


class PegasusDataset(Dataset):
    def __init__(self, encodings, labels):
        # TODO - check what this is
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)


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
