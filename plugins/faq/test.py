import os
import ast
import time
import helper
import pandas as pd
import numpy as np
import openai
import openai.embeddings_utils as em_utils
from openai.embeddings_utils import get_embedding, cosine_similarity

def load_faq_from_embedding_file(file) -> list:
    #read csv file
    df = pd.read_csv(file, encoding='utf-8')
    df['emb'] = df['emb'].apply(ast.literal_eval)
    vec = []
    for i in range(0, df.shape[0]):
        q = df.iloc[i, 0]
        a = df.iloc[i, 1]
        emb = df.iloc[i, 2]
        
        dict = {"question": q, "answer": a, "emb": emb}
        vec.append(dict)

    return vec

def get_answer_from_embedding(question, qa_list, topk=1) ->list:
    #get embedding of question
    emb = em_utils.get_embedding(question, engine="text-embedding-ada-002")
    #calculate cosine similarity
    cos_sim = []
    for i in range(0, len(qa_list)):
        cos_sim.append(em_utils.cosine_similarity(emb, qa_list[i]["emb"]))
    #get topk and cos_sim greater than threshold
    topk_idx = np.argsort(cos_sim)[-topk:]
    answer = []
    for i in range(0, len(topk_idx)):
        cos_sim = cos_sim[topk_idx[i]]
        #print cos_sim
        print("cos_sim: {}, answer: {}".format(cos_sim, qa_list[topk_idx[i]]["answer"]))
        if cos_sim > 0.93:
            answer.append(qa_list[topk_idx[i]]["answer"])
        else:
            #如果相似度小于0.92，用chatgpt计算相似度
            #生成用于计算query和question的相似度的chatgpt prompt,返回相似度
            gpt_sim = get_chatgpt_style_sim(question, qa_list[topk_idx[i]]["question"])
            if float(gpt_sim) > 0.9:
                answer.append(qa_list[topk_idx[i]]["answer"])
                #print gpt_sim
                print("gpt_sim: {}, answer: {}", gpt_sim, qa_list[topk_idx[i]]["answer"])
    #如果 answer列表为空，用question请求openai chatcomplition
    if len(answer) <= 0:
       answer.append(get_answer_from_openai(question))
    return answer

def get_answer_from_openai(question) ->str:
    #get answer from openai chatcomplition
    messages = [{"role": "user", "content": question}]
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.0,
        max_tokens=2000
    )
    #如何将str转成float

    return res.choices[0]["message"]["content"]

def get_chatgpt_style_sim(query, question) -> float:
    #生成一个chatgpt prompt，用于计算query和question的相似度,返回float
    prompt = f"计算下面两段话的相似度\n{query} \n{question}，相似度用[0,1]之间的小数表示，数值越大，相似度越高,只需要给我返回这个相似度值即可"
    messages = [{"role": "user", "content": prompt}]
    #调用openai chatcomplition api 计算query和question的相似度
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.0,
        max_tokens=2000
    )

    return res.choices[0]["message"]["content"]

#write a function from csv file read quesion and answer 
#then get question embedding and save to csv file
def generate_embedding_from_csv_file(file):
   #read csv file
   df = pd.read_csv(file, encoding='utf-8')
   vec = []
   for i in range(0, df.shape[0]):  
       question = df.iloc[i, 0]
       answer = df.iloc[i, 1]
       emb = get_embedding(question, engine="text-embedding-ada-002")
       touched_answer = helper.touch_up_the_text(answer, 0)
       if len(touched_answer) == 0:
           touched_answer = answer
       print("[FAQ] answer: {}, touched_answer: {}".format(answer, touched_answer))
       dict = {"question": question, "answer": touched_answer.strip(), "emb": emb}
       vec.append(dict)
   #save to csv file
   df = pd.DataFrame(vec)
   #get file path1
   curdir = os.path.dirname(file)
   df.to_csv(os.path.join(curdir, "qa_embedding.csv"), index=False)

def generate_relevant_queries(question:str)->list:
    queries_input = f"""
    生成一系列跟此问题相关的问题。这些问题应该是一些你认为用户可能会从不同角度提出的同一个问题。
    在问题中使用关键词的变体，尽可能的概括，包括你能想到的尽可能多的提问。
    比如, 包含的查询就想这样 ['keyword_1 keyword_2', 'keyword_1', 'keyword_2']

    User question: {question}

    Format: {{"queries": ["query_1", "query_2", "query_3"]}}
    """
    queries = helper.json_gpt(queries_input)
    return queries["queries"]

#write termal command to test
if __name__ == "__main__":
    print("test")
    openai.api_key = "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"
    curdir = os.path.dirname(__file__)
    #给出菜单选项，选择1，2，3。1是生成qa_embedding.csv文件，2是从qa_embedding.csv文件中读取数据，3是退出
    #如果选择1，则调用generate_embedding_from_csv_file函数生成qa_embedding.csv文件
    #如果选择2，则调用load_faq_from_embedding_file函数从qa_embedding.csv文件中读取数据
    #如果选择3，则调用generate_relevant_queries函数生成相关的查询，需要输入一个问题
    #如果选择4，则退出
    #生成代码如下   
    while True:
        print("1. generate embedding file")
        print("2. get answer from embedding file")
        print("3. generate relevant queries")
        print("4. words generate story")
        option = input("option: ")
        if option == "1":
            generate_embedding_from_csv_file(os.path.join(curdir, "qa_test.csv"))
        elif option == "2":
            #遍历当前目录下是否存在qa_embedding.csv文件
            #如果不存在则打印提示信息
            if not os.path.exists(os.path.join(curdir, "qa_embedding.csv")):
                generate_embedding_from_csv_file(os.path.join(curdir, "qa.csv"))
            else:
                qa_list = load_faq_from_embedding_file(os.path.join(curdir, "qa_embedding.csv"))
                while True:
                    question = input("question: ")
                    if question == "exit":
                        break
                    reply = get_answer_from_embedding(question, qa_list, topk=1)
                    print(reply[0])
                    while True:
                        question = input("question: ")
                        if question == "exit":
                            break
                        reply = get_answer_from_embedding(question, qa_list, topk=1)
                        print(reply[0])
        elif option == "3":
            #需要输入问题
            question = input("question: ")
            sim_question_list = generate_relevant_queries(question)
            emb_sim_question_list = em_utils.get_embeddings(sim_question_list, engine="text-embedding-ada-002")
            emb_question = em_utils.get_embeddings([question], engine="text-embedding-ada-002")
            for i in range(0, len(emb_sim_question_list)):
                sim = em_utils.cosine_similarity(emb_question, emb_sim_question_list[i])
                print(f"question: {sim_question_list[i]}, sim: {sim}")

        elif option == "4":
            #循环10次，每次生成一个故事
            for i in range(0, 10):

                prompt = """You as a story writer, I give you a set of words, you need to choose 10 words from them and generate an article on the topic of culture and art， keeping it to 200 words or less. The list of words is as follows：["divide", "preliminary", "expand", "interior", "process", "reside", "determine", "oppose", "rural", "arrange", "lodging", "broadcast", "project", "punctual", "wealth", "motive", "construct", "latitude", "explode", "imperative", "incidence", "dedicate", "commerce", "displace", "fuse", "impair", "boundary", "democratic", "persuade", "radical", "renovate", "undertake", "prohibit", "receipt", "topic", "delight", "illustrate", "economic", "allocate", "expose", "subtle", "emphasize", "aware", "strategy", "intermediate", "estimate", "genuine", "prevail", "nurture", "isolate", "configure", "represent", "clinic", "constrain", "orchestra", "rebel", "oblige", "disrupt", "withdraw", "contribution", "interpret", "amplitude", "mechanical", "divert", "pension", "dynamic", "appreciate", "ignite", "petrol", "wreck"]
json format:{'story':{'english': 'story in english', 'chinese': 'story in simplified chinese', 'selected_words': ['word1', 'word2']}}"""
                result = helper.json_gpt(prompt)
                print(result)
                #休息一会
                time.sleep(2)
            break
        else:
            print("invalid option")




