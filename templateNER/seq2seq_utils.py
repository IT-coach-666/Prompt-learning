import logging
import os
import pickle
from multiprocessing import Pool
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

def preprocess_data_bart(data):
    """ 将数据处理成 encode 和 decode 形式。
    
    Args:
        data: 包含 [input_text, target_text, tokenizer, args] 四元组。
            input_text: Australia will defend the Ashes
            target_text: Australia is a location entity
    
    Returns:
        键值对
            source_ids
            source_mask
            target_ids
    """
    input_text, target_text, tokenizer, args = data

    input_ids = tokenizer.batch_encode_plus(
        [input_text], max_length=args.max_seq_length, padding='max_length', truncation=True, return_tensors="pt",
    )

    target_ids = tokenizer.batch_encode_plus(
        [target_text], max_length=args.max_seq_length, padding='max_length', truncation=True, return_tensors="pt"
    )

    return {
        "source_ids": input_ids["input_ids"].squeeze(),
        "source_mask": input_ids["attention_mask"].squeeze(),
        "target_ids": target_ids["input_ids"].squeeze(),
    }


class SimpleSummarizationDataset(Dataset):
    """ Simple Summarization Dataset，适应 Encoder-Decoder 网络结构的输入输出形式
    """
    def __init__(self, tokenizer, args, data, mode):
        self.tokenizer = tokenizer

        cached_features_file = os.path.join(
            args.cache_dir, args.model_name + "_cached_" + str(args.max_seq_length) + str(len(data))
        )

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not args.no_cache)
            or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)
        data = [
            (input_text, target_text, tokenizer, args)
            for input_text, target_text in zip(data["input_text"], data["target_text"])
        ]
        
        self.examples = [preprocess_data_bart(d) for d in tqdm(data, disable=args.silent)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
