# -*-coding:UTF-8 -*-
import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
import torch.nn.functional as F     # 激励函数都在这

def to_input_id(sentence_input):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence_input))

if __name__ == "__main__":

    # load and init
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')
    # 第2個資料夾的loss最低，因此使用來生成答案
    config = BertConfig.from_pretrained('trained_model/1/config.json')
    model = BertForMaskedLM.from_pretrained('trained_model/1/pytorch_model.bin', from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config)
    model.eval()

    print("請輸入context")
    context = input()
    print("請輸入question")
    question = input()

    input_id = to_input_id("[CLS] " + context + " [SEP] " + question + " [SEP]")

    count = 0
    answer = ""
    maskpos = len(input_id)                     # 標出要預測答案的位置
    input_id.append(103)
    # 補齊長度
    while len(input_id)<512:
        input_id.append(0)

    # 限制答案最大長度為10
    while(count < 10):
        input_id_tensor = torch.LongTensor([input_id])
        outputs = model(input_id_tensor)
        predictions = outputs[0]
        predicted_index = torch.argmax(predictions[0, maskpos]).item()      # 生出最有可能的token_id
        predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)  # id轉token

        # 當預測為[SEP]的時候，就結束生成答案
        if predicted_token == '[SEP]':
            break

        answer = answer + predicted_token       # 將生成的token連接起來
        input_id[maskpos] = predicted_index     # 用生成的token_id取代當前的[MASK]的id
        maskpos += 1
        if maskpos < 512:
            input_id[maskpos] = 103             # 標出下一個預測的[MASK]的id
        else:
            break

        count += 1

    print("答案:" + answer)