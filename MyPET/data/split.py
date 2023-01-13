# coding=utf-8

import json
import random

with open('train.json', 'r') as in_file:
    lines = in_file.readlines()
    items = [json.loads(line.strip()) for line in lines]
    random.shuffle(items)
    
    train_items = items[:100]
    eval_items = items[100:500]

    with open('train.tsv', 'w') as out_file:
        for item in train_items:
            out_file.write(item['question'] + '\t' + str(item['label']) + '\n')

    with open('dev.tsv', 'w') as out_file:
        for item in eval_items:
            out_file.write(item['question'] + '\t' + str(item['label']) + '\n')