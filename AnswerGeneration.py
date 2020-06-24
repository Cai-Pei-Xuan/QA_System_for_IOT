# -*-coding:UTF-8 -*-
import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
import torch.nn.functional as F     # 激励函数都在这

class testAnswerGeneration():
    def __init__(self):
        self.tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')
        self.config = BertConfig.from_pretrained('trained_model/1/config.json')
        self.model = BertForMaskedLM.from_pretrained('trained_model/1/pytorch_model.bin', from_tf=bool('.ckpt' in 'bert-base-chinese'), config=self.config)
        self.model.eval()

    def to_input_id(self, sentence_input):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence_input))

    def getAnswer(self, context, question):
        input_id = self.to_input_id("[CLS] " + context + " [SEP] " + question + " [SEP]")

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
            outputs = self.model(input_id_tensor)
            predictions = outputs[0]
            predicted_index = torch.argmax(predictions[0, maskpos]).item()      # 生出最有可能的token_id
            predicted_token = self.tokenizer.convert_ids_to_tokens(predicted_index)  # id轉token

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

        return answer

    