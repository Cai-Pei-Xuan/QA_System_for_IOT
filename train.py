# -*-coding:UTF-8 -*-
from preprocess_data import convert_data_to_feature, makeDataset
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForMaskedLM, AdamW
import torch
import os

# 動態調整學習率，參考網站:http://www.spytensor.com/index.php/archives/32/
def adjust_learning_rate(optimizer, Learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = Learning_rate

if __name__ == "__main__":

    # 设置使用的GPU用法來源:https://www.cnblogs.com/darkknightzh/p/6591923.html
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # set device(使用gpu)
    device = torch.device("cuda")

    # PreprocessData
    train_data_feature = convert_data_to_feature('DRCD_training.json')
    test_data_feature = convert_data_to_feature('DRCD_test.json')
    train_dataset = makeDataset(token_embeddings = train_data_feature['token_embeddings'], segement_embeddings = train_data_feature['segement_embeddings'], attention_mask = train_data_feature['attention_mask'], masked_lm_labels = train_data_feature['masked_lm_labels'])
    test_dataset = makeDataset(token_embeddings = test_data_feature['token_embeddings'], segement_embeddings = test_data_feature['segement_embeddings'], attention_mask = test_data_feature['attention_mask'], masked_lm_labels = test_data_feature['masked_lm_labels'])
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    # 設定model參數
    # type_vocab_size:如果token_type_ids有用到2種以上時，則需要做修改
    # 另外還有在本地的transformers/modeling_utils.py中第469~471行需要註解掉，因為pytorch會拋出錯誤訊息(RuntimeError: Error(s) in loading state_dict for BertForMaskedLM : size mismatch for bert.embeddings.token_type_embeddings.weight: copying a param with shape torch.Size([2, 768]) from checkpoint, the shape in current model is torch.Size([3, 768]).)，但原始bert就有16維可以用，所以我用3維應該是沒問題的
    config = BertConfig.from_pretrained('bert-base-chinese', type_vocab_size = 3)
    model = BertForMaskedLM.from_pretrained('bert-base-chinese', from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config)
    model.to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    Learning_rate = 5e-6       # 學習率
    training_epoch = 5         # 訓練回合
    optimizer = AdamW(optimizer_grouped_parameters, lr=Learning_rate, eps=1e-8)

    for epoch in range(training_epoch):
        # 訓練模式
        model.train()
        if epoch % 5 == 0 and epoch != 0:
            Learning_rate = Learning_rate * 0.5
            adjust_learning_rate(optimizer, Learning_rate)

        AllTrainLoss = 0.0
        count = 0
        for batch_dict in train_dataloader:
            batch_dict = tuple(t.to(device) for t in batch_dict)
            outputs = model(
                input_ids = batch_dict[0],                  # token_embeddings
                token_type_ids = batch_dict[1],             # segement_embeddings
                attention_mask = batch_dict[2],
                masked_lm_labels = batch_dict[3]
                )
            
            loss, logits = outputs[:2]
            AllTrainLoss += loss.item()
            count += 1

            model.zero_grad()
            loss.backward()
            optimizer.step()
            
        Average_train_loss = round(AllTrainLoss/count, 3)

        # 測試模式
        model.eval()
        AllTestLoss = 0.0
        count = 0
        for batch_dict in test_dataloader:
            batch_dict = tuple(t.to(device) for t in batch_dict)
            outputs = model(
                input_ids = batch_dict[0],                  # token_embeddings
                token_type_ids = batch_dict[1],             # segement_embeddings
                attention_mask = batch_dict[2],
                masked_lm_labels = batch_dict[3]
                )

            loss, logits = outputs[:2]
            AllTestLoss += loss.item()
            count += 1
        
        Average_test_loss = round(AllTestLoss/count, 3)

        print('第' + str(epoch+1) + '次' + '訓練模式，loss為:' + str(Average_train_loss) + '，測試模式，loss為:' + str(Average_test_loss))
        # 檢查並創建資料夾
        folder = os.path.exists('trained_model/'+ str(epoch))
        if not folder:
            os.makedirs('trained_model/'+ str(epoch))
        
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model  
        model_to_save.save_pretrained('trained_model/'+str(epoch))