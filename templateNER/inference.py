# coding=utf-8

import math
import time
import torch
from utils_metrics import get_entities_bio, f1_score, classification_report
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig

# 模板
template_list = [" is a location entity .", " is a person entity .", " is an organization entity .",
                     " is an other entity .", " is not a named entity ."]

entity_dict = {0: 'LOC', 1: 'PER', 2: 'ORG', 3: 'MISC', 4: 'O'}

# 依赖于具体的业务场景，其值等价于 entity_dict 中键的数量
num_labels = 5

# 
max_n_grams = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class InputExample():
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels

def template_entity(model, tokenizer, n_grams, input_TXT, start):
    """ 计算候选 n_grams 是各个实体类别的分数
    Args:
        n_grams:
        input_TXT:
        start:
    
    Returns:

    """
    words_length = len(n_grams)
    words_length_list = [len(i) for i in n_grams]
    input_TXT = [input_TXT]*(num_labels*words_length)

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']

    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(n_grams[i]+template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    
    # 替换成 decoder 端的输入起始符
    output_ids[:, 0] = 2
    output_length_list = [0]*num_labels*words_length

    # 记录每个类别模板分词后的长度：从样本开头一直到 is a xx 的长度
    for i in range(len(temp_list)//5):
        base_length = ((tokenizer(temp_list[i * num_labels], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[1] - 4
        output_length_list[i*num_labels:i*num_labels+ num_labels] = [base_length]*num_labels
        output_length_list[i*num_labels+4] += 1

    score = [1]*num_labels*words_length
    with torch.no_grad():
        output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[0]
        for i in range(output_ids.shape[1] - 3):
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            logits = logits.to('cpu').numpy()

            # 累乘计算最终输出序列的“合理性”得分
            for j in range(0, num_labels*words_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    end = start+(score.index(max(score))//num_labels)
    return [start, end, entity_dict[(score.index(max(score))%num_labels)], max(score)] # [start_index,end_index,label,score]


def prediction(model, tokenizer, input_text):
    """ 对于输入文本做实体识别预测
    Args:
        input_text: 
    input_text: ACL will be held in Bangkok
    """
    input_text_tokens = input_text.split(' ')

    entity_list = []

    # 在第i个位置，遍历 n-gram （从 1-gram 到 max_n-gram）
    # acl will be hold in bangkok
    # acl, acl will, acl will be, acl will be hold, acl will be hold in, acl will be hold in bangkok
    for i in range(len(input_text_tokens)):
        words = []
        for j in range(1, min(max_n_grams+1 , len(input_text_tokens) - i + 1)):
            word = (' ').join(input_text_tokens[i:i+j])
            words.append(word)

        # 将第 i 个位置开始的所有候选 gram 和 原始文本送去做预测
        entity = template_entity(model, tokenizer, words, input_text, i)   # [start_index,end_index,label,score]
        if entity[1] >= len(input_text_tokens):
            entity[1] = len(input_text_tokens)-1
        if entity[2] != 'O':
            entity_list.append(entity)
    
    # 将结果合并，嵌套实体则选择最大的值。
    i = 0
    if len(entity_list) > 1:
        while i < len(entity_list):
            j = i+1
            while j < len(entity_list):
                if (entity_list[i][1] < entity_list[j][0]) or (entity_list[i][0] > entity_list[j][1]):
                    j += 1
                else:
                    if entity_list[i][3] < entity_list[j][3]:
                        entity_list[i], entity_list[j] = entity_list[j], entity_list[i]
                        entity_list.pop(j)
                    else:
                        entity_list.pop(j)
            i += 1
    label_list = ['O'] * len(input_text_tokens)

    for entity in entity_list:
        label_list[entity[0]:entity[1]+1] = ["I-"+entity[2]]*(entity[1]-entity[0]+1)
        label_list[entity[0]] = "B-"+entity[2]
    return label_list

def cal_time(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def main():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartForConditionalGeneration.from_pretrained('./exp/template/')
    model.eval()
    model.config.use_cache = False
    model.to(device)

    score_list = []
    file_path = './data/conll2003/test.txt'
    guid_index = 1
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(words=words, labels=labels))
                    words = []
                    labels = []
            else:
                splits = line.strip().split("\t")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(words=words, labels=labels))

    
    trues_list = []
    preds_list = []
    str = ' '
    num_01 = len(examples)
    num_point = 0
    start = time.time()
    for example in examples:
        sources = str.join(example.words)
        preds_list.append(prediction(model, tokenizer, sources))
        trues_list.append(example.labels)
        print('%d/%d (%s)'%(num_point+1, num_01, cal_time(start)))
        print('Pred:', preds_list[num_point])
        print('Gold:', trues_list[num_point])
        num_point += 1


    true_entities = get_entities_bio(trues_list)
    pred_entities = get_entities_bio(preds_list)
    results = {
        "f1": f1_score(true_entities, pred_entities)
    }
    print(results["f1"])
    for num_point in range(len(preds_list)):
        preds_list[num_point] = ' '.join(preds_list[num_point]) + '\n'
        trues_list[num_point] = ' '.join(trues_list[num_point]) + '\n'
    with open('./pred.txt', 'w') as f0:
        f0.writelines(preds_list)
    with open('./gold.txt', 'w') as f0:
        f0.writelines(trues_list)

if __name__ == '__main__':
    main()
