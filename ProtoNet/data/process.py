# coding=utf-8

from collections import defaultdict

with open('./thunews/class.txt', 'r') as in_file:
    lines = in_file.readlines()
    classes = [line.strip() for line in lines]

class2sent = defaultdict(list)
with open('./thunews/train.txt', 'r') as in_file:
    lines = in_file.readlines()
    for line in lines:
        sent, class_idx = line.split('\t')
        class_name = classes[int(class_idx.strip())]
        class2sent[class_name].append(sent)

with open('./thunews/dev.txt', 'r') as in_file:
    lines = in_file.readlines()
    for line in lines:
        sent, class_idx = line.split('\t')
        class_name = classes[int(class_idx.strip())]
        class2sent[class_name].append(sent)

with open('./thunews/test.txt', 'r') as in_file:
    lines = in_file.readlines()
    for line in lines:
        sent, class_idx = line.split('\t')
        class_name = classes[int(class_idx.strip())]
        class2sent[class_name].append(sent)



source_domain_classes = list(class2sent.keys())[:6]
target_domain_classes = list(class2sent.keys())[6:]

# 写入训练数据集
with open('train.txt', 'w') as out_file:
    for class_name in source_domain_classes:
        sents = class2sent[class_name]
        for sent in sents:
            out_file.write(sent + '\t' + class_name + '\n')

# 写入测试数据集
with open('test.txt', 'w') as out_file:
    for class_name in target_domain_classes:
        sents = class2sent[class_name]
        for sent in sents:
            out_file.write(sent + '\t' + class_name + '\n')

    