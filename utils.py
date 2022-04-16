from transformers import PegasusForConditionalGeneration, Adafactor
from transformers.optimization import AdafactorSchedule

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
