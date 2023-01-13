# coding=utf-8

"""
训练脚本 
"""

import pandas as pd
import logging
from seq2seq_model import Seq2SeqModel
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_data = pd.read_csv("./data/train.csv", sep=',').values.tolist()
train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

eval_data = pd.read_csv("./data/dev.csv", sep=',').values.tolist()
eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 50,
    "train_batch_size": 100,
    "num_train_epochs": 20,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "evaluate_during_training": True,
    "evaluate_generated_text": True,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "max_length": 25,
    "manual_seed": 4,
    "save_steps": 11898,
    "gradient_accumulation_steps": 1,
    "output_dir": "./exp/template",
}

model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-large",
    args=model_args,
)

model.train_model(train_df, eval_data=eval_df)

results = model.eval_model(eval_df)