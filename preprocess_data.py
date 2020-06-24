# -*-coding:UTF-8 -*-
from transformers import BertTokenizer
import json
import random
import torch
from torch.utils.data import TensorDataset

# 讀取數據
def LoadJson(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        AllData = json.load(f)
    return AllData

def convert_data_to_feature(filepath):
    DRCD = LoadJson(filepath)
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')

    token_embeddings = []
    segement_embeddings = []
    attention_mask = []
    masked_lm_labels = []
    max_seq_len = 0
    Context_count = 0

    # BertForMaskedLM的訓練需要特殊符號('[MASK]')以及被mask掉的詞的id
    # context最大長度450，question最大長度42，answer最大長度16，以及4個特殊符號(1個[CLS]，3個[SEP])，加起來最大不超過512
    for data in DRCD["data"]:
        for paragraph in data["paragraphs"]:
            context = paragraph["context"]
            word_piece_list = tokenizer.tokenize(context)
            if len(word_piece_list) <= 450:
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    word_piece_list = tokenizer.tokenize(question)
                    if len(word_piece_list) <= 42:
                        answer = qa["answers"][0]["text"]
                        answer = answer + "[SEP]"
                        word_piece_list = tokenizer.tokenize(answer)
                        if len(word_piece_list) <= 16:
                            max_seq_len = create_input_features(tokenizer, context, question, answer, token_embeddings, segement_embeddings, attention_mask, masked_lm_labels, max_seq_len)
                            Context_count += 1
    
    print("最大長度:",max_seq_len)
    print("符合條件的context有" + str(Context_count) + "筆資料")
    print("總共產生" + str(len(token_embeddings)) + "筆資料")
    assert max_seq_len <= 512 # 小於BERT-base長度限制
    max_seq_len = 512         # 將長度統一補齊至512(避免traindata和testdata最大長度不一致)

    # 補齊長度
    for c in token_embeddings:
        while len(c)<max_seq_len:
            c.append(0)

    for c in segement_embeddings:
        while len(c)<max_seq_len:
            c.append(0)

    for c in attention_mask:
        while len(c)<max_seq_len:
            c.append(0)

    for c_l in masked_lm_labels:
        while len(c_l)<max_seq_len:
            c_l.append(-1)
    
    # BERT input embedding
    assert len(token_embeddings) == len(segement_embeddings) and len(token_embeddings) == len(attention_mask) and len(token_embeddings) == len(masked_lm_labels)
    data_features = {'token_embeddings':token_embeddings,
                    'segement_embeddings':segement_embeddings,
                    'attention_mask':attention_mask,
                    'masked_lm_labels':masked_lm_labels}

    return data_features

# input_token範例:
# C:C1C2
# Q:Q1Q2
# A:A1A2[SEP]
# 由於A有3個token(包含[SEP]，因為要預測答案何時該停下)，所以input_data有3筆
# [cls]C1C2[sep]Q1Q2[sep][mask]
# [cls]C1C2[sep]Q1Q2[sep]A1[mask]
# [cls]C1C2[sep]Q1Q2[sep]A1A2[mask]
# segement_embeddings的type有3種(0,1,2)，但經過實測發現即使只用兩種，loss的結果也沒太大差別，因此該程式主要目的是為了練習而存在
# 產生input_features
def create_input_features(tokenizer, context, question, answer, token_embeddings, segement_embeddings, attention_mask, masked_lm_labels, max_seq_len):
    segement_id = []
    attention_id = []
    masked_lm_id = []

    input_context_string = "[CLS]" + context + "[SEP]"
    input_context_word_piece_list = tokenizer.tokenize(input_context_string)
    for i in range(len(input_context_word_piece_list)):
        segement_id.append(0)
        attention_id.append(1)
        masked_lm_id.append(-1)

    input_question_string = question + "[SEP]"
    input_question_word_piece_list = tokenizer.tokenize(input_question_string)
    for i in range(len(input_question_word_piece_list)):
        segement_id.append(1)
        attention_id.append(1)
        masked_lm_id.append(-1)

    # 將答案替換成'[MASK]'，但一次只有1個token
    answer_word_piece_list = []
    answer_word_piece_list = tokenizer.tokenize(answer)
    for index, ans in enumerate(answer_word_piece_list):
        input_word_piece_list = input_context_word_piece_list + input_question_word_piece_list
        token_id = []
        input_segement_id = segement_id + []
        input_attention_id = attention_id + []
        input_masked_lm_id = masked_lm_id + []
        
        for i in range(index):
            input_word_piece_list.append(answer_word_piece_list[i])
            input_segement_id.append(2)
            input_attention_id.append(1)
            input_masked_lm_id.append(-1)

        input_word_piece_list.append('[MASK]')
        input_segement_id.append(2)
        input_attention_id.append(1)
        input_masked_lm_id.append(tokenizer.convert_tokens_to_ids(ans))      #被mask掉的詞的id

        token_id = tokenizer.convert_tokens_to_ids(input_word_piece_list)
        # print(input_word_piece_list)
        # print(token_id)
        # print(input_segement_id)
        # print(input_attention_id)
        # print(input_masked_lm_id)

        # 將各id的結果存起來
        token_embeddings.append(token_id)
        segement_embeddings.append(input_segement_id)
        attention_mask.append(input_attention_id)
        masked_lm_labels.append(input_masked_lm_id)

        # 更新最大長度
        if len(token_id) > max_seq_len:
            max_seq_len = len(token_id)

    return max_seq_len

def makeDataset(token_embeddings, segement_embeddings, attention_mask, masked_lm_labels):
    all_token_embeddings = torch.tensor([token_id for token_id in token_embeddings], dtype=torch.long)
    all_segement_embeddings = torch.tensor([segement_id for segement_id in segement_embeddings], dtype=torch.long)
    all_attention_mask = torch.tensor([attention_id for attention_id in attention_mask], dtype=torch.long)
    all_masked_lm_labels = torch.tensor([masked_lm_id for masked_lm_id in masked_lm_labels], dtype=torch.long)

    return TensorDataset(all_token_embeddings, all_segement_embeddings, all_attention_mask, all_masked_lm_labels)
        
if __name__ == "__main__":
    
    data_features = convert_data_to_feature('DRCD_test.json')
    Dataset = makeDataset(data_features['token_embeddings'], data_features['segement_embeddings'], data_features['attention_mask'], data_features['masked_lm_labels'])
    print(Dataset[0])

    # 如果要測試下面的例子，就將125~129的註解拿掉-----
    # st = []
    # M = 0
    # tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')
    # create_input_features(tokenizer, "要探討從梨俱吠", "夜柔吠陀", "因國督", st, st, st ,st, M)

   
