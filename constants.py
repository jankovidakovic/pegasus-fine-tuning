from transformers import IntervalStrategy, SchedulerType
from transformers.training_args import OptimizerNames

DEFAULT_TRAINING_ARGS = {
    "output_dir": "./results",  # output directory
    "max_steps": 2000,  # max number of gradient updates

    "per_device_train_batch_size": 4,  # batch size per device during training, can increase if memory allows
    "gradient_accumulation_steps": 32,
    "gradient_checkpointing": False,

    "per_device_eval_batch_size": 16,  # batch size for evaluation, can increase if memory allows
    "dataloader_num_workers": 0,
    "eval_accumulation_steps": None,

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
