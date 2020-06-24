# -*-coding:UTF-8 -*-
import Jieba
import WikiData

class testSearchContext():
    def __init__(self):
        self.SearchContext_test = Jieba.testJieba()          # 載入Jieba
        self.WikiData_test = WikiData.testWikiData()          # 載入WikiData

    # 擷取是名詞的單詞，且不是常見的單詞(如:的)
    def ExtractNoun(self, sentence):
        seg_list = self.SearchContext_test.Jieba_test.posseg.lcut(sentence)
        word_list = []
        for i in seg_list:
            if i.flag == 'n':
                if i.word not in word_list:
                    if i.word not in self.SearchContext_test.CommonWordTable:
                        word_list.append(i.word)

        return word_list

    # 計算有多少單詞出現在內容中
    def CountWordNum(self, data, word_list):
        count = 0
        for word in word_list:
            if word in data:
                count = count + 1

        return count

    # 找出跟問句有關的context
    def FindContext(self, word_list):
        count = 0
        count_list = []
        alldata_list = []
        context_list = []
        for word in word_list:
            try:
                data_list = self.WikiData_test.AllWikiDataDict[word]
                for data in data_list:
                    CountWord = self.CountWordNum(data, word_list)
                    alldata_list.append(data)
                    count_list.append(CountWord)
                    if CountWord > count:
                        count = CountWord
            except:
                continue

        for index, value in enumerate(count_list):
            if value == count:
                context_list.append(alldata_list[index])

        return context_list

if __name__ == '__main__':
    SearchContext = testSearchContext()
    sentence = "紅茶比綠茶多出了甚麼步驟?"
    word_list = SearchContext.ExtractNoun(sentence)
    context = SearchContext.FindContext(word_list)
    print(context)