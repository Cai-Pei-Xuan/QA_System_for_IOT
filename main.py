# -*-coding:UTF-8 -*-
import SearchContext
import AnswerGeneration
from flask import Flask, request
import os
import json

SearchContext_test = SearchContext.testSearchContext()
AnswerGeneration_test = AnswerGeneration.testAnswerGeneration()

# Flask應用來源:https://www.cnblogs.com/lsdb/p/10488448.html
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"

@app.route('/up_question', methods=['post'])
def get_ContextAndAnswer():
    question = request.form["question"]
    word_list = SearchContext_test.ExtractNoun(question)
    context_list = SearchContext_test.FindContext(word_list)
    current_context = ''
    for context in context_list:
        answer = AnswerGeneration_test.getAnswer(context, question)
        if answer != '':
            if '[UNK]' not in answer:
                current_context = context
                break


    All_result = {}
    if current_context != '':
        All_result["context"] = current_context         # 代表context
        All_result["answer"] = answer           # 代表answer
    else:
        All_result["context"] = "沒有符合的內容"         # 代表context
        All_result["answer"] = ""           # 代表answer

    return json.dumps(All_result, ensure_ascii=False)


if __name__ == "__main__":
    # try:
    #     SearchContext_test = SearchContext.testSearchContext()
    #     AnswerGeneration_test = AnswerGeneration.testAnswerGeneration()
    # except Exception as e:
    #     print(e)

    # question = "數學是利用什麼來研究數量、結構、變化以及空間等概念的一門學科，從某種角度看屬於形式科學的一種?"
    # word_list = SearchContext_test.ExtractNoun(question)
    # context = SearchContext_test.FindContext(word_list)
    # print("找到的context : " + context)
    # answer = AnswerGeneration_test.getAnswer(context, question)
    # print("找到的answer : " + answer)
    app.run(host="0.0.0.0", port="3000", debug=False)

