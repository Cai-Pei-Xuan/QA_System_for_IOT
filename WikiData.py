# -*-coding:UTF-8 -*-
import json

class testWikiData():
    def __init__(self):
        self.AllWikiDataDict = self.LoadAllWikiDataDict()          # 讀取AllWikiDataDict資料  

    # 讀取數據
    def LoadJson(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            AllData = json.load(f)
        return AllData

    # 將字典存檔，字典快速保存與讀取，參考網站:https://blog.csdn.net/u012155582/article/details/78077180
    def SaveJson(self, filepath, Dict):
        f = open(filepath, 'w', encoding='UTF-8')
        json.dump(Dict,f, ensure_ascii=False)
        f.close()

    # 將所有wiki資料以單詞為key，內容以\n做分割後存成list
    def DealWithWikiData(self, filepath, AllWikiDataDict):
        AllWikiData = self.LoadJson(filepath)
        for key in AllWikiData.keys():
            key_Data_list = AllWikiData[key].split('\n')            # 以\n做分割
            if key_Data_list[0] not in AllWikiDataDict.keys():
                AllWikiDataDict[key_Data_list[0]] = []
                for index, key_Data in enumerate(key_Data_list):
                    if index != 0:
                        if key_Data != '':
                            AllWikiDataDict[key_Data_list[0]].append(key_Data)

        self.SaveJson('AllWikiDataDict.json', AllWikiDataDict)

    # 讀取AllWikiDataDict資料
    def LoadAllWikiDataDict(self):
        AllWikiDataDict = {}
        try:
            AllWikiDataDict = self.LoadJson('AllWikiDataDict.json')
        except:
            DealWithWikiData('wiki20180805_fullText.json', AllWikiDataDict)

        return AllWikiDataDict
