import os
import sys
import ast
import time
import pandas as pd
import numpy as np
import openai
import openai.embeddings_utils as em_utils
from openai.embeddings_utils import get_embedding, cosine_similarity

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from helper import json_gpt, touch_up_the_text, remove_no_chinese

OPENAI_API_KEY = "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"
if sys.platform == 'win32':
    CHROMA_DB_DIR = ".\\plugins\\faq\\vectordb\\chroma_db\\"
    PROMPT_DIR = "prompt\\"
else:
    CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectordb/chroma_db")
    PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt")
CHROMA_COLLECTION_NAME = "sponger_bot"
SIMILAR_KEYWORDS_FILE = "similar_keywords.csv"

similar_keywords = []

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

def get_answer_from_chroma(question, topk=1) ->list:
    embedding = OpenAIEmbeddings()
    db = Chroma(CHROMA_COLLECTION_NAME, embedding, persist_directory=CHROMA_DB_DIR)
    if db is None:
        print("db is None")
        return []
    
    result = db.similarity_search_with_relevance_scores(question, k=topk)
    answers = []
    for i in range(0, len(result)):
        meta_data = result[i][0].metadata
        #如果meta_data没有数据，说明不是faq的qa对
        if len(meta_data) <= 0:
            continue
        meta_question = meta_data["question"]
        answer = meta_data["answer"]
        score = result[i][1]
        print("score: {}, question: {}, meta_question: {}".format(score, question, meta_question))
        if score > 0.93:
            answers.append(answer)
    #如果 answer列表为空，生成相似问题，从相似问题中查找是否有满足阈值的答案
    if len(answers) <= 0:
        rel_questions = generate_relevant_queries(question)
        total_result = []
        for i in range(0, len(rel_questions)):
            result = db.similarity_search_with_relevance_scores(rel_questions[i], k=topk)
            total_result.extend(result)
        #根据score对total_result进行排序
        sorted_result = sorted(total_result, key=lambda x: x[1], reverse=True)
        #遍历total_result，如果score大于0.9，将answer加入到answers中
        for i in range(0, len(sorted_result)):
            meta_question = sorted_result[i][0].metadata["question"]
            score = sorted_result[i][1]
            print("score: {}, meta_question: {}".format(score, meta_question))
            if score > 0.88:
                answers.append(sorted_result[i][0].metadata["answer"])
    return answers

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
       question = df.iloc[i, 0].strip()
       answer = df.iloc[i, 1].strip()
       if len(question) == 0 or len(answer) == 0:
            continue
       
       emb = get_embedding(question, engine="text-embedding-ada-002")
       touched_answer = touch_up_the_text(answer, 0)
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

def generate_embedding_from_csv_file_chroma(file):
    #read csv file
    df = pd.read_csv(file, encoding='utf-8')
    meta_datas = []
    texts = []
    for i in range(0, df.shape[0]):
        question = str(df.iloc[i, 0]).strip()
        answer = str(df.iloc[i, 1]).strip()
        #将question利用split分成多个句子
        question_list = question.split("；")
        for j in range(0, len(question_list)):
            question = question_list[j]
            if len(question) == 0 or len(answer) == 0 or question == "nan" or answer == "nan":
                continue
            dict = {"question": question, "answer": answer}
            meta_datas.append(dict)
            texts.append(question)
            print(dict)
    
    embedding = OpenAIEmbeddings()
    chroma = Chroma(CHROMA_COLLECTION_NAME, embedding, persist_directory=CHROMA_DB_DIR)
    if len(texts) > 0:
        chroma.add_texts(texts, meta_datas)

def load_affirmations(file) -> list:
    curdir = os.path.dirname(__file__)
    affirmations_path = os.path.join(curdir, file)
    with open(affirmations_path, "r", encoding="utf-8") as f:
        affirmations = []
        for line in f:
            affirmation = line.strip()
            if affirmation:
                affirmations.append(affirmation)
    return affirmations
def generate_embedding_from_txt_file_chroma(file):
    affirmations = load_affirmations(file)
    embedding = OpenAIEmbeddings()
    chroma = Chroma(CHROMA_COLLECTION_NAME, embedding, persist_directory=CHROMA_DB_DIR)
    chroma.add_texts(affirmations, meta_datas=None)
 #chain test
from langchain.prompts import PromptTemplate, FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import json
def generate_relevant_queries(question:str)->list:
    example = [{"query":"清华是不是很看重学历？", 
                "similar_keywords":["学历", "背景"],
                "similar_queries": ["清华大学对学历要求严格吗？", 
                                    "清华对申请人的学历有什么具体要求？",
                                    "在清华大学的招生中，学历是否是决定性因素之一？",
                                    "清华大学录取时是否会对学历进行严格的筛选？",
                                    "清华的招生政策是否存在对学历的硬性要求？",
                                    "清华是不是很看重背景？",
                                    "学历对于申请清华的影响有多大？",
                                    "清华对于申请者背景会有什么要求？",
                                    "学历在清华的招生评审中扮演着怎样的角色？",
                                    "清华是不是很看重背景？"]},
                {"query":"人大mba提面流程？", 
                 "similar_keywords":["提面", "提前面试", "面试", "提前批", "提前面", "提前复试", "预面试"],
                 "similar_queries": ["人大mba提前面试流程是怎样的？", 
                                    "人民大学mba的面试流程是怎么样的？",
                                    "人大mba提前批的流程是什么样的？",
                                    "人大mba有提前面这个环节吗？可以告诉我一下相关流程吗？",
                                    "人大mba提前复试的步骤是怎样的？",
                                    "人大mba的预面试是什么意思？有什么准备工作吗？",
                                    "人大mba提前面试一般在什么时间进行？",
                                    "人大mba提前批的录取标准是什么？",
                                    "人大mba预面试需要准备些什么？",
                                    "请问人大mba提前复试的内容有哪些？"]},
                    {"query":"法大报考条件是什么？",
                     "similar_keywords": ['中国政法大学', '中政', '法大', '中法大'],
                     "similar_queries": ["中国政法大学提面条件是什么？",
                                         "中政提面条件是什么？",
                                         "中法大提面条件是什么？",
                                         "中国政法大学的提面要求有哪些？",
                                         "中政的提面标准是什么？",
                                         "中法大的提面条件具体包括哪些？"]}]
    example_prompt = ChatPromptTemplate.from_messages(
            [('human', '{query}'), ('ai', '{similar_queries}')]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(examples=example,
                                                      example_prompt=example_prompt)

    sys_prompt = f"""
    你是一个相似问题生成器，生成相似问题的时候，可以分步骤来做：
    第一步，根据每一组相似关键词列表所有可能进行排列组合替换{similar_keywords}；
    第二步，替换完成的问题，生成可能的不同角度的相似问题；
    第三步，根据生成的相似问题，生成最终的相似问题列表。
    """
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            few_shot_prompt,
            ("human", "{query}"),
        ]
    )
    final_prompt.format(query=question)
    print(final_prompt)
    chain = final_prompt | ChatOpenAI(model_name="gpt-4", temperature=0.0)
    output = chain.invoke({"query": question})
    content_str = output.content.replace("'", '"')
    try:
        json_output = json.loads(content_str)
    except json.JSONDecodeError as e:
        print("json decode error: {}".format(e))
        return {}
    return  json_output

import math
def load_similar_keywords()->list:
    df = pd.read_csv(os.path.join(curdir, "similar_keywords.csv"), encoding='utf-8')
    for i in range(0, df.shape[0]):
        #去掉list中的空字符串元素
        #keywords = list(filter(lambda x: x != "" and not math.isnan(x), list(df.iloc[i, :])))
        keywords = [x for x in list(df.iloc[i, :]) 
                    if (isinstance(x, str) and x != "") 
                    or (isinstance(x, float) and not math.isnan(x))]
        similar_keywords.append(keywords)
    return similar_keywords
#langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA, VectorDBQAWithSourcesChain
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
def langchain_text_qa(file_path:str, question:str):
    #load qa file
    loader = PyPDFLoader(file_path)
    document = loader.load()
    #split and get embedding
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    split_docs = splitter.split_documents(document)
    if len(split_docs) == 0:
        split_docs = document
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    vs = Chroma.from_documents(split_docs, embedding)
    qa = RetrievalQAWithSourcesChain.from_chain_type(llm=OpenAI(model_name="gpt-3.5-turbo-16k"), 
                                                     chain_type="stuff", 
                                                     retriever=vs.as_retriever(),
                                                     reduce_k_below_max_tokens=True)
    result = qa({"question": question})
    print(result)


def langchain_long_text_summary(file_path:str):
    loader = PyPDFLoader(file_path)
    document = loader.load()
    print("doc len: {}".format(len(document)))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    split_docs = text_splitter.split_documents(document)
    print("split_docs len: {}".format(len(split_docs)))
    if len(split_docs) == 0:
        split_docs = document
    llm = OpenAI(max_tokens=500)
    chain = load_summarize_chain(llm=llm, chain_type="refine", verbose=True)
    result = chain.run(split_docs)
    print(result)

from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
def generate_qustion():
    loader = TextLoader(os.path.join(os.path.dirname(__file__), "qa_test.txt"), encoding="utf-8")
    print("doc length:{}".format(len(loader.load())))
    doc = loader.load()[0]
    print("doc page content:{}".format(doc.page_content))
    prompt = PromptTemplate.from_template("生成的问题跟答案只能用简体中文，并且不能出现文中已经有的问题")
    chain = QAGenerationChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613", temperature=0.0))
    qa = chain.run(doc.page_content)
    #print all qa list
    for i in range(0, len(qa)):
        print(qa[i])

from langchain.chains import LLMChain
from langchain.evaluation.qa import QAEvalChain
def evalution_qa():
    prompt = PromptTemplate(template="Question: {question}\nAnswer:", input_variables=["question"])
    llm = OpenAI(model_name="text-davinci-003", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    examples = [
    {
        "question": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
        "answer": "11"
    },
    {
        "question": 'Is the following sentence plausible? "Joao Moutinho caught the screen pass in the NFC championship."',
        "answer": "No"
    }]

    predictions = chain.apply(examples)
    print(predictions)

    eval_chain = QAEvalChain.from_llm(OpenAI(temperature=0.0))
    graded_outputs = eval_chain.evaluate(examples, predictions, question_key="question", prediction_key="text")

    for i, eg in enumerate(examples):
        print(f"Example {i}:")
        print("Question: " + eg['question'])
        print("Real Answer: " + eg['answer'])
        print("Predicted Answer: " + predictions[i]['text'])
        print("Predicted Grade: " + graded_outputs[i]['text'])
        print()

###作文批改 
##段落结构
#段落关联词
def correct_paragraph_correlative() -> list:
    #段落关联词
    paragraph_correlative_words = ["首先", "其次", "最后", "总之", "总而言之", "总的来说", "总的说来", "总的来看", "总的来讲", "总的说", "总的来", "总的"]
    example = [{"query":"清华是不是很看重学历？", 
                "correlative_keywords":paragraph_correlative_words,
                "similar_queries": ["清华大学对学历要求严格吗？", 
                                    "清华对申请人的学历有什么具体要求？",
                                    "在清华大学的招生中，学历是否是决定性因素之一？",
                                    "清华大学录取时是否会对学历进行严格的筛选？",
                                    "清华的招生政策是否存在对学历的硬性要求？",
                                    "清华是不是很看重背景？",
                                    "学历对于申请清华的影响有多大？",
                                    "清华对于申请者背景会有什么要求？",
                                    "学历在清华的招生评审中扮演着怎样的角色？",
                                    "清华是不是很看重背景？"]}]
    example_prompt = ChatPromptTemplate.from_messages(
            [('human', '{query}'), ('ai', '{similar_queries}')]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(examples=example,
                                                      example_prompt=example_prompt)
    sys_prompt = f"""
    你是一个相似问题生成器，生成相似问题的时候，可以分步骤来做：
    第一步，根据每一组相似关键词列表所有可能进行排列组合替换{similar_keywords}；
    第二步，替换完成的问题，生成可能的不同角度的相似问题；
    第三步，根据生成的相似问题，生成最终的相似问题列表。
    """
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            few_shot_prompt,
            ("human", "{query}"),
        ]
    )
    final_prompt.format(query=question)
    print(final_prompt)
    chain = final_prompt | ChatOpenAI(model_name="gpt-4", temperature=0.0)
    output = chain.invoke({"query": question})
    content_str = output.content.replace("'", '"')
    try:
        json_output = json.loads(content_str)
    except json.JSONDecodeError as e:
        print("json decode error: {}".format(e))
        return {}
    return  json_output
 
 #标点符号
def correct_article_punctuation():
    pass

#从excel文件中加载学校提示列表
def load_select_school_prompt(file_path: str) -> list:
    df = pd.read_excel(file_path)
    prompt_list = []
    last_university = ""
    for i in range(0, df.shape[0]):
        university = df.iloc[i, 0]
        prompt = df.iloc[i, 1]
        few_shot_question = df.iloc[i]["few_shot_question"]
        few_shot_answer = df.iloc[i]["few_shot_answer"]
        if isinstance(university, str) and len(university) > 0:
            last_university = university
            entry = {
                "university": university,
                "prompt": prompt,
                "few_shot_list": [
                    {
                        "question": few_shot_question,
                        "answer": few_shot_answer
                    }
                ]
            }     
            prompt_list.append(entry)
        else:
            existing_university = next((item for item in prompt_list if item["university"] == last_university), None)
            if existing_university is not None:
                existing_university["few_shot_list"].append({
                    "question": few_shot_question,
                    "answer": few_shot_answer
                })
        
    return prompt_list

def get_university_match_result(university: str, user_info: str) -> str:
    university = remove_no_chinese(university)
    prompt_list = load_select_school_prompt(os.path.join(os.path.join(curdir, PROMPT_DIR), "select_school_prompt.xlsx"))
    for i in range(0, len(prompt_list)):
        if university == prompt_list[i]["university"]:
            prompt = prompt_list[i]["prompt"]
            few_shot_list = prompt_list[i]["few_shot_list"]
            break
    if len(few_shot_list) <= 0:
        return '还没有这个院校的择校规则信息，因此无法提供该院校的择校服务'
    example_prompt = ChatPromptTemplate.from_messages(
            [('human', '{question}'), ('ai', '{answer}')]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(examples=few_shot_list,
                                                      example_prompt=example_prompt)
    sys_prompt = f"{prompt}\n{user_info}"
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            few_shot_prompt,
            ("human", "{question}"),
        ]
    )
    final_prompt.format(question=user_info)
    chain = final_prompt | ChatOpenAI(model_name="gpt-4", temperature=0.0)
    output = chain.invoke({"question": user_info})
    content_str = output.content.replace("'", '"')
    print(content_str)
    return content_str


#write termal command to test
if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    openai.api_key = OPENAI_API_KEY
    curdir = os.path.dirname(__file__)
    load_similar_keywords()
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
        print("5. langchain text qa")
        print("6. langchain long text summary")
        print("7. get answer from chroma")
        print("8. generate chroma from csv file")
        print("9. generate question answer pair from txt file")
        print("10. generate chroma from txt file")
        print("11. evalution qa")
        print("12. select school from prompt")
        print("13. exit")
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
                result = json_gpt(prompt)
                print(result)
                #休息一会
                time.sleep(2)
            break
        elif option == "5":
            question = input("question: ")
            langchain_text_qa(os.path.join(curdir, "summary_test.pdf"), question)
        elif option == "6":
            langchain_long_text_summary(os.path.join(curdir, "summary_test.pdf"))
        elif option == "7":
            load_select_school_prompt(os.path.join(curdir, "select_school_prompt.xlsx"))
            question = input("question: ")
            answers = get_answer_from_chroma(question, 4)
            print(answers)
        elif option == "8":
            generate_embedding_from_csv_file_chroma(os.path.join(os.path.join(curdir, "knowledge_base", "interview_services"), "merged.csv"))
        elif option == "9":
            generate_qustion()
        elif option == "10":
            generate_embedding_from_txt_file_chroma("affirmations.txt")
        elif option == "11":
            evalution_qa()
        elif option == "12":
            university = input("university: ")
            user_info = input("user_info: ")
            get_university_match_result(university, user_info)
        elif option == "13":
            break
        else:
            print("invalid option")




