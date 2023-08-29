#encoding utf-8

import time
import os
import sys
import ast
import json
import math
import pandas as pd
import numpy as np
import openai
import openai.embeddings_utils as em_utils
from openai.embeddings_utils import get_embedding, cosine_similarity

import plugins
from bridge.bridge import Bridge
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
import config
from .helper import json_gpt, touch_up_the_text
from plugins import *

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate, FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

if sys.platform == 'win32':
    CHROMA_DB_DIR = ".\\plugins\\faq\\vectordb\\chroma_db\\"
else:
    CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectordb/chroma_db")
CHROMA_COLLECTION_NAME = "sponger_bot"
SIMILAR_KEYWORDS_FILE = "similar_keywords.csv"

@plugins.register(
    name="FAQ",
    desire_priority=0,
    hidden=True,
    desc="FAQ system",
    version="0.1",
    author="kennylee",
)

class FAQ(Plugin):
    def load_faq_from_embedding_file(self, file) -> list:
        pass
    def get_answer_from_openai(self, question) -> list:
        pass
    def generate_embedding_from_csv_file(self, file):
        pass
    def generate_embedding_from_csv_file_chroma(self, file):
        pass
    def get_chatgpt_style_sim(self, quary, question, retry_count=0) -> json:
        pass
    def generate_relevant_queries(self, question)->list:
        pass
    def generate_relevant_queries_langchain(self, question)->list:
        pass
    def get_answer_from_chroma(self, question, topk=1) ->list:
        pass
    def load_similar_keywords(self, file):
        pass

    def __init__(self):
        super().__init__()
        try:
            # init openai
            openai.api_key = config.conf().get("open_ai_api_key")
            os.environ["OPENAI_API_KEY"] = config.conf().get("open_ai_api_key")
            # load config
            curdir = os.path.dirname(__file__)
            config_path = os.path.join(curdir, "faq_config.json")
            conf = None
            if not os.path.exists(config_path):
                conf = {"need_context": "true", "sim_threshold": 0.9}
                with open(config_path, "w") as f:
                    json.dump(conf, f, indent=4)
            else:
                with open(config_path, "r") as f:
                    conf = json.load(f)
            self.need_context = conf["need_context"]
            self.embedding_model = conf["embedding_model"]
            self.cos_sim_threshold_max = conf["cos_sim_threshold_max"]
            self.cos_sim_threshold_mid = conf["cos_sim_threshold_mid"]
            self.cos_sim_threshold_min = conf["cos_sim_threshold_min"]
            self.similar_keywords = []
            self.load_similar_keywords(os.path.join(curdir, SIMILAR_KEYWORDS_FILE))
            # load faq
            #self.qa_list = self.load_faq_from_embedding_file(os.path.join(curdir, "qa_embedding.csv"))
            #logger.info("[FAQ] faq loaded. qa_list size: %d" % len(self.qa_list))

            # register event handler
            self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
            logger.info("[FAQ] inited")
        except Exception as e:
            logger.warn("[FAQ] init failed")
            raise e
    def is_valid_json_string(self, s):
        try:
            json.loads(s)
            return True
        except json.JSONDecodeError:
            return False
        
    def on_handle_context(self, e_context: EventContext):
        if e_context["context"].type != ContextType.TEXT:
            return

        content = e_context["context"].content
        logger.debug("[FAQ] on_handle_context. content: %s" % content)
        reply = self.get_faq(content)
        if reply:
            logger.debug("[FAQ] FAQ reply= %s", reply)
            e_context["reply"] = reply
            # if need context, break the event, and wait for next context
            # if not need context, break pass the event, and continue to handle context
            # normally, we should not break pass the event, otherwise, the context will be missed
            session_id = e_context["context"]["session_id"]
            bot = Bridge().get_bot("chat")
            session = bot.sessions.build_session(session_id)
            session.add_query(content)
            session.add_reply(reply.content)
            logger.debug("[FAQ] session messages: {}".format(session.messages))
            e_context.action = EventAction.BREAK_PASS
        else:
            logger.debug("[FAQ] FAQ reply is None, request gpt")
            e_context.action = EventAction.BREAK
        
    
    def load_faq_from_embedding_file(self, file) -> list:
        #if file not exist, use gererate_faq_embedding.py to generate
        if not os.path.exists(file):
            self.generate_embedding_from_csv_file(os.path.join(os.path.dirname(__file__), "qa.csv"))
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
    
    def get_answer_from_embedding(self, question, qa_list, topk=1) ->list:
        #get embedding of question
        emb = em_utils.get_embedding(question, engine=self.embedding_model)
        #calculate cosine similarity
        cos_sim = []
        for i in range(0, len(qa_list)):
            cos_sim.append(em_utils.cosine_similarity(emb, qa_list[i]["emb"]))
        #get topk and cos_sim greater than threshold
        topk_idx = np.argsort(cos_sim)[-topk:]
        answer = []
        for i in range(0, len(topk_idx)):
            cos_sim = cos_sim[topk_idx[i]]
            logger.debug("[FAQ] cos_sim: {}, answer: {}".format(cos_sim, qa_list[topk_idx[i]]["answer"]))
            if cos_sim >= self.cos_sim_threshold_max:
                answer.append(qa_list[topk_idx[i]]["answer"])
            else:
                sim_question_list = self.generate_relevant_queries(question)
                emb_list = em_utils.get_embeddings(sim_question_list, engine=self.embedding_model)
                for j in range(0, len(emb_list)):
                    cos_sim = em_utils.cosine_similarity(qa_list[topk_idx[i]]["emb"], emb_list[j])
                    top_question = qa_list[topk_idx[i]]["question"]
                    logger.debug(f"question: {top_question}, rel_question: {sim_question_list[j]}, sim: {cos_sim}")
                    if cos_sim >= self.cos_sim_threshold_max:
                        answer.append(qa_list[topk_idx[i]]["answer"])
                        break
        return answer
    
    def get_answer_from_chroma(self, question, topk=1) ->list:
        embedding = OpenAIEmbeddings()
        db = Chroma(CHROMA_COLLECTION_NAME, embedding, persist_directory=CHROMA_DB_DIR)
        if db is None:
            logger.error("[FAQ] chroma db is None")
            return []
        
        ori_result = db.similarity_search_with_score(question, k=topk)
        #logger.debug("[FAQ] question sim result: {}".format(ori_result))
        answers = []
        for i in range(0, len(ori_result)):
            meta_question = ori_result[i][0].metadata["question"]
            answer = ori_result[i][0].metadata["answer"]
            score = ori_result[i][1]
            logger.debug("[FAQ] origin score: {}, question: {}, meta_question: {}".format(score, question, meta_question))
            if self.cos_sim_threshold_max + score <= 1:
                answers.append(answer)
        #如果 answer列表为空，生成相似问题，从相似问题中查找是否有满足阈值的答案
        if len(answers) <= 0:
            total_result = []
            #先把原始问题匹配结果放进总结果列表
            total_result.extend(ori_result)
            rele_questions = self.generate_relevant_queries_langchain(question)
            for i in range(0, len(rele_questions)):
                result = db.similarity_search_with_score(rele_questions[i], k=topk)
                #logger.debug("[FAQ] relevant question result: {}".format(result))
                total_result.extend(result)
            #根据score对total_result进行排序
            sorted_result = sorted(total_result, key=lambda x: x[1])
            #遍历total_result，如果score大于0.9，将answer加入到answers中
            for i in range(0, len(sorted_result)):
                meta_question = sorted_result[i][0].metadata["question"]
                score = sorted_result[i][1]
                logger.debug("[FAQ] relevant score: {}, meta_question: {}".format(score, meta_question))
                if self.cos_sim_threshold_mid + score <= 1:
                    answers.append(sorted_result[i][0].metadata["answer"])
            #如果依赖没有召回答案，找到最接近的问题，提示用户是否在找这个问题
            if len(answers) <= 0 and len(total_result) > 0:
                score = total_result[0][1]
                question = total_result[0][0].metadata["question"]
                if self.cos_sim_threshold_min + score <= 1:
                    answers.append("我猜你是想问这个问题吧。({})".format(question))
        return answers

    def get_faq(self, question) -> Reply:
        #get answer
        #answer = self.get_answer_from_embedding(question, self.qa_list, topk=1)
        answer = self.get_answer_from_chroma(question, topk=1)
        #return reply
        if len(answer) > 0:
            return Reply(content=answer[0], type=ReplyType.TEXT)
        else:
            return None

    #write a function from csv file read quesion and answer 
    #then get question embedding and save to csv file
    #then touch up the text
    def generate_embedding_from_csv_file(self, file):
       #read csv file
       df = pd.read_csv(file, encoding='utf-8')
       vec = []
       for i in range(0, df.shape[0]):  
           question = df.iloc[i, 0]
           answer = df.iloc[i, 1]
           emb = get_embedding(question, engine=self.embedding_model)
           touched_answer = touch_up_the_text(answer, 0)
           if len(touched_answer) == 0:
               touched_answer = answer
           logger.debug("[FAQ] answer: {}, touched_answer: {}".format(answer, touched_answer))
           dict = {"question": question, "answer": touched_answer.strip(), "emb": emb}
           vec.append(dict)
       #save to csv file
       df = pd.DataFrame(vec)
       #get file path
       curdir = os.path.dirname(file)
       df.to_csv(os.path.join(curdir, "qa_embedding.csv"), index=False)
       logger.info("[FAQ] generate qa_embedding.csv success")

    def generate_embedding_from_csv_file_chroma(self, file):
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
        
        embedding = OpenAIEmbeddings()
        chroma = Chroma(CHROMA_COLLECTION_NAME, embedding, persist_directory=CHROMA_DB_DIR)
        chroma.add_texts(texts, meta_datas)

    def get_answer_from_openai(self, question) ->str:
        #get answer from openai chatcomplition
        messages = [{"role": "user", "content": question}]
        res = openai.ChatCompletion.create(
            model=config.conf().get("model"),
            messages=messages,
            temperature=0.0,
            max_tokens=2000
        )
        return res.choices[0]["message"]["content"]
    
    def get_chatgpt_style_sim(self, query, question, retry_count) -> json:
        #生成一个chatgpt prompt，用于计算query和question的相似度,返回float
        prompt = f"对比下面两段话的相似度\n{query} \n{question}，相似度用[0,1]之间的小数表示，数值越大，相似度越高。提供JSON格式的输出，key为sim，value为相似度。"
        messages = [{"role": "system", "content": "Output only valid JSON"},{"role": "user", "content": prompt}]
        #调用openai chatcomplition api 计算query和question的相似度
        try:
            res = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.0,
                max_tokens=2000
            )
            
            res_content = res.choices[0]["message"]["content"]
            logger.debug("[OPEN_AI] res: {}".format(res_content))
            if self.is_valid_json_string(res_content):
                return json.loads(res_content)
            else:
                logger.warn("[OPEN_AI] res_content is not valid json string: {}".format(res_content))
                return None
        except Exception as e:
            need_retry = retry_count < 2
            result = {"sim":0.0}
            if isinstance(e, openai.error.RateLimitError):
                logger.warn("[OPEN_AI] RateLimitError: {}".format(e))
                if need_retry:
                    time.sleep(20)
            elif isinstance(e, openai.error.Timeout):
                logger.warn("[OPEN_AI] Timeout: {}".format(e))
                if need_retry:
                    time.sleep(5)
            elif isinstance(e, openai.error.APIConnectionError):
                logger.warn("[OPEN_AI] APIConnectionError: {}".format(e))
                need_retry = False
            else:
                logger.warn("[OPEN_AI] Exception: {}".format(e))
                need_retry = False

            if need_retry:
                logger.warn("[OPEN_AI] 第{}次重试".format(retry_count + 1))
                return self.get_chatgpt_style_sim(query, question, retry_count + 1)
            else:
                return result
    def generate_relevant_queries(self, question)->list:
        queries_input = f"""
        生成一系列跟此问题相关的问题。这些问题应该是一些你认为用户可能会从不同角度提出的同一个问题。
        在问题中使用关键词的变体，尽可能的概括，包括你能想到的尽可能多的提问。
        比如, 包含的查询就想这样 ['keyword_1 keyword_2', 'keyword_1', 'keyword_2']

        User question: {question}

        Format: {{"queries": ["query_1", "query_2", "query_3"]}}
        """
        queries = json_gpt(queries_input)
        return queries["queries"]
    def generate_relevant_queries_langchain(self, question)->list:
        example = [{"query":"清华是不是很看重学历？", 
                    "similar_queries": ["清华大学对学历要求严格吗？", 
                                        "清华对申请人的学历有什么具体要求？",
                                        "在清华大学的招生中，学历是否是决定性因素之一？",
                                        "清华大学录取时是否会对学历进行严格的筛选？",
                                        "清华对学历要求很高？"]},
                    {"query":"人大提面考什么？", 
                    "similar_queries": ["人民大学提面考啥？", 
                                        "人大提前面试考什么？",
                                        "人民大学提前面试考哪些？",
                                        "人大提前面试会问些什么问题？",
                                        "人大提前面试会问些什么问题？"]},    ]
        example_prompt = ChatPromptTemplate.from_messages(
                [('human', '{query}'), ('ai', '{similar_queries}')]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(examples=example,
                                                          example_prompt=example_prompt)
        # similar_keywords = [["清华大学", "清华"], 
        #                     ["北京大学", "北大"],
        #                     ["光华管理学院", "光华管院", "光华"],
        #                     ["人民大学", "人大"], 
        #                     ["提面", "提前面试", "提前面"],
        #                     ["北京师范大学", "北师大", "北师"],
        #                     ["非全日制", "非全"],
        #                     ["北京理工", "北理"]]
        sys_prompt = f"""
        你是一个相似问题生成器，生成相似问题的时候，可以分步骤来做：
        第一步，根据每一组相似关键词列表所有可能进行排列组合替换{self.similar_keywords}；
        第二步，根据替换完成的问题，生成可能的不同角度的相似问题；
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
            logger.error("[FAQ] json decode error: {}".format(e))
            return {}
        logger.debug("[FAQ] relevant question: {}".format(json_output))
        return  json_output
    
    def load_similar_keywords(self, file):
        self.similar_keywords.clear()
        df = pd.read_csv(file, encoding='utf-8')
        for i in range(0, df.shape[0]):
            #去掉list中的空字符串元素
            keywords = [x for x in list(df.iloc[i, :]) 
                        if (isinstance(x, str) and x != "") 
                        or (isinstance(x, float) and not math.isnan(x))]
            self.similar_keywords.append(keywords)
    
#write a terminal command to test
if __name__ == "__main__":
    curdir = os.path.dirname(__file__)
    #遍历当前目录下是否存在qa_embedding.csv文件
    #如果不存在则生成qa_embedding.csv文件
    faq = FAQ()
    if not os.path.exists(os.path.join(curdir, "qa_embedding.csv")):
        qa_file = os.path.join(curdir, "qa.csv")
        faq.generate_embedding_from_csv_file(qa_file)
    while True:
        question = input("question: ")
        if question == "exit":
            break
        reply = faq.get_faq(question)
        print(reply)

