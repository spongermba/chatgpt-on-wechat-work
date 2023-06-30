#encoding utf-8

import time
import os
import ast
import json
import requests
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
from plugins import *

#write a function 请求openai chatcomplition ，只返回正确的json
def json_gpt(input:str)->json:
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Output only valid JSON"},
            {"role": "user", "content": input},
        ],
        temperature=0.5,
    )

    text = completion.choices[0].message.content
    parsed = json.loads(text)

    return parsed


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
    def get_chatgpt_style_sim(self, quary, question, retry_count=0) -> json:
        pass
    def generate_relevant_queries(self, question)->list:
        pass

    def __init__(self):
        super().__init__()
        try:
            # init openai
            openai.api_key = config.conf().get("open_ai_api_key")
            
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
            self.cos_sim_threshold = conf["cos_sim_threshold"]
            self.gpt_sim_threshold = conf["gpt_sim_threshold"]
            logger.info("[FAQ] config loaded. need_context: %s" % self.need_context)

            # load faq
            self.qa_list = self.load_faq_from_embedding_file(os.path.join(curdir, "qa_embedding.csv"))
            logger.info("[FAQ] faq loaded. qa_list size: %d" % len(self.qa_list))

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
            if cos_sim >= self.cos_sim_threshold:
                answer.append(qa_list[topk_idx[i]]["answer"])
            else:
                sim_question_list = self.generate_relevant_queries(question)
                emb_list = em_utils.get_embeddings(sim_question_list, engine=self.embedding_model)
                for j in range(0, len(emb_list)):
                    cos_sim = em_utils.cosine_similarity(qa_list[topk_idx[i]]["emb"], emb_list[j])
                    top_question = qa_list[topk_idx[i]]["question"]
                    logger.debug(f"question: {top_question}, rel_question: {sim_question_list[j]}, sim: {cos_sim}")
                    if cos_sim >= self.cos_sim_threshold:
                        answer.append(qa_list[topk_idx[i]]["answer"])
                        break

                #如果相似度小于cos_threshold，用chatgpt计算相似度
                # gpt_sim_json = self.get_chatgpt_style_sim(question, qa_list[topk_idx[i]]["question"], 0)
                # #判断返回的json是否正确，并且包含sim字段
                # if gpt_sim_json is None or "sim" not in gpt_sim_json:
                #     logger.warn("[FAQ] get_chatgpt_style_sim return json is None or not contains sim field")
                #     continue

                # gpt_sim = gpt_sim_json["sim"]
                # if isinstance(gpt_sim, str):
                #     gpt_sim = float(gpt_sim)
                # logger.debug("[FAQ] gpt_sim: {}, answer: {}".format(gpt_sim, qa_list[topk_idx[i]]["answer"]))
                # if gpt_sim >= self.gpt_sim_threshold:
                #     answer.append(qa_list[topk_idx[i]]["answer"])
                    
        #即使没有找到答案，也不再请求openai chatcomplition，插件处理完了，还会交给chat_channel处理
        return answer
    
        #如果 answer列表为空，用question请求openai chatcomplition
        # if len(answer) <= 0:
        #    answer.append(self.get_answer_from_openai(question))
        
    
    def get_faq(self, question) -> Reply:
        #get answer
        answer = self.get_answer_from_embedding(question, self.qa_list, topk=1)
        #return reply
        if len(answer) > 0:
            return Reply(content=answer[0], type=ReplyType.TEXT)
        else:
            return None

    #write a function from csv file read quesion and answer 
    #then get question embedding and save to csv file
    def generate_embedding_from_csv_file(self, file):
       #read csv file
       df = pd.read_csv(file, encoding='utf-8')
       vec = []
       for i in range(0, df.shape[0]):  
           q = df.iloc[i, 0]
           a = df.iloc[i, 1]
           emb = get_embedding(q, engine=self.embedding_model)
           dict = {"question": q, "answer": a.strip(), "emb": emb}
           vec.append(dict)
       #save to csv file
       df = pd.DataFrame(vec)
       #get file path
       curdir = os.path.dirname(file)
       df.to_csv(os.path.join(curdir, "qa_embedding.csv"), index=False)
       logger.info("[FAQ] generate qa_embedding.csv success")

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

