# -*-coding:UTF-8 -*-
import jieba
import jieba.posseg
import os

class testJieba():
    def __init__(self):
        self.path = os.getcwd()  # 當前路徑
        self.Jieba_test = self.LoadJieba()          # 載入jieba的詞庫
        self.CommonWordTable = self.LoadCommonTable(os.path.join(self.path, 'CommonTable/CommonWordTable'))      # 自訂常見詞

    def LoadJieba(self):
        # 引入繁體中文詞庫
        jieba.initialize(os.path.join(self.path, 'dict/dict.txt.big'))
        # 載入自訂的詞庫(可以多個載入，前面的資料會保留
        jieba.load_userdict(os.path.join(self.path, 'dict/mydict'))
        jieba.load_userdict(os.path.join(self.path, 'dict/dict.txt.big.txt'))
        jieba.load_userdict(os.path.join(self.path, 'dict/ptt.txt'))
        jieba.load_userdict(os.path.join(self.path, 'dict/wiki.dict.txt'))
        jieba.load_userdict(os.path.join(self.path, 'dict/attractions.dict.txt'))
        jieba.load_userdict(os.path.join(self.path, 'dict/dcard.dict.txt'))
        jieba.load_userdict(os.path.join(self.path, 'dict/zh_translate_en.dict'))

        return jieba
        
    # 讀取常見詞或詞性列
    def LoadCommonTable(self, filepath):
        CommonTable = []
        for i in open(filepath, 'r', encoding='UTF-8'):
            # 移除換行
            CommonTable.append(i.replace('\n', ''))
        return CommonTable
    