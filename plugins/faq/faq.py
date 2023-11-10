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
from openai.embeddings_utils import get_embedding

import plugins
from bridge.bridge import Bridge
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from plugins import *
from plugins.faq.helper import json_gpt, touch_up_the_text, remove_no_chinese
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate, FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI

import config
#from banwords.lib.WordsSearch import WordsSearch

if sys.platform == 'win32':
    CHROMA_DB_DIR = ".\\plugins\\faq\\vectordb\\chroma_db\\"
    PROMPT_DIR = ".\\plugins\\faq\\prompt\\"
else:
    CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectordb/chroma_db")
    PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt")
CHROMA_COLLECTION_NAME = "sponger_bot"
SIMILAR_KEYWORDS_FILE = "similar_keywords.csv"
SELECT_SCHOOL_PROMPT_FILE = "select_school_prompt.xlsx"

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
    def load_affirmations(self, file) -> list:
        pass
    def load_similar_keywords(self, file):
        pass
    def load_select_school_prompt(self, file_path: str) -> list:
        pass
    def get_university_full_name(self, name: str) -> str:
        pass
    def get_answer_from_openai(self, question) -> list:
        pass
    def generate_embedding_from_csv_file(self, file):
        pass
    def generate_embedding_from_csv_file_chroma(self, file):
        pass
    def generate_embedding_from_txt_file_chroma(self, file):
        pass
    def get_chatgpt_style_sim(self, quary, question, retry_count=0) -> json:
        pass
    def generate_relevant_queries(self, question)->list:
        pass
    def generate_relevant_queries_langchain(self, question)->list:
        pass
    def get_answer_from_chroma(self, session_id, question, topk=1) ->list:
        pass
    def get_faq(self, session_id, question) -> Reply:
        pass
    def get_university_match_result(self, university: str, user_info: str) -> Reply:
        pass

    def __init__(self):
        super().__init__()
        try:
            # init openai
            openai.api_key = config.conf().get("open_ai_api_key")
            os.environ["OPENAI_API_KEY"] = config.conf().get("open_ai_api_key")
            self.model = config.conf().get("model")
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
            self.previous_not_hit_questions = {}
            self.prompt_list = []
            self.load_similar_keywords(os.path.join(curdir, SIMILAR_KEYWORDS_FILE))
            self.load_select_school_prompt(os.path.join(os.path.join(curdir, PROMPT_DIR), SELECT_SCHOOL_PROMPT_FILE))

            # register event handler
            self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
            logger.info("[FAQ] inited")
        except Exception as e:
            logger.warn("[FAQ] init failed")
            raise e
    def load_affirmations(self, file) -> list:
        curdir = os.path.dirname(__file__)
        affirmations_path = os.path.join(curdir, file)
        with open(affirmations_path, "r", encoding="utf-8") as f:
            affirmations = []
            for line in f:
                affirmation = line.strip()
                if affirmation:
                    affirmations.append(affirmation)
        logger.info("[FAQ] affirmations loaded. affirmations size: %d" % len(self.affirmations))
        return affirmations

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
        # 特殊的择校指令，走择校流程
        if content.startswith("*"):
            if len(content) > 1:
                content = content[1:]
                results = content.split(" ", 1)
                if len(results) < 2:
                    e_context["reply"] = Reply(content="请按照格式输入：*学校名称 用户背景信息，或者 *择校 用户背景信息", type=ReplyType.TEXT)
                    e_context.action = EventAction.BREAK_PASS
                    return
                
                university = results[0]
                user_info = results[1]
                # 通用择校指令，根据用户背景信息，返回匹配的学校
                if university == "择校":
                    e_context["reply"] = Reply(content="开发中...", type=ReplyType.TEXT)
                    e_context.action = EventAction.BREAK_PASS
                # 指定学校择校指令，根据用户背景信息，返回期望院校的匹配结果
                else:
                    e_context["reply"] = self.get_university_match_result(university, user_info)
                    e_context.action = EventAction.BREAK_PASS
                logger.debug("[FAQ] university: {}, user_info: {}".format(university, user_info))
            else:
                return
        else:
            session_id = e_context["context"]["session_id"]
            reply = self.get_faq(session_id, content)
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
    def get_university_full_name(self, name: str) -> str:
        name = remove_no_chinese(name)
        for names in self.similar_keywords:
            if name in names:
                return names[0]
        return "" 
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
    
    def get_answer_from_chroma(self, session_id, question, topk=1) ->list:
        embedding = OpenAIEmbeddings()
        db = Chroma(CHROMA_COLLECTION_NAME, embedding, persist_directory=CHROMA_DB_DIR)
        if db is None:
            logger.error("[FAQ] chroma db is None")
            return []
        
        
        ori_result = db.similarity_search_with_relevance_scores(question, k=topk)
        
        #logger.debug("[FAQ] question sim result: {}".format(ori_result))
        answers = []
        for i in range(0, len(ori_result)):
            meta_data = ori_result[i][0].metadata
            score = ori_result[i][1]
            #如果meta_data没有数据，说明不是faq的qa对
            if len(meta_data) <= 0:
                if self.cos_sim_threshold_max <= score :
                    pre_question = self.previous_not_hit_questions.get(session_id)
                    if pre_question is None:
                        continue
                    #递归召回
                    logger.debug("[FAQ] recursive call get_answer_from_chroma. session_id: {}, pre_question: {}".format(session_id, pre_question))
                    self.previous_not_hit_questions.pop(session_id)
                    return self.get_answer_from_chroma(session_id, pre_question, topk=1)
            else:
                meta_question = meta_data["question"]
                answer = meta_data["answer"]
                logger.debug("[FAQ] origin score: {}, question: {}, meta_question: {}".format(score, question, meta_question))
                if self.cos_sim_threshold_max <= score:
                    answers.append(answer)
        #如果 answer列表为空，生成相似问题，从相似问题中查找是否有满足阈值的答案
        if len(answers) <= 0:
            total_result = []
            #先把原始问题匹配结果放进总结果列表
            total_result.extend(ori_result)
            rele_questions = self.generate_relevant_queries_langchain(question)
            for i in range(0, len(rele_questions)):
                result = db.similarity_search_with_relevance_scores(rele_questions[i], k=topk)
                #logger.debug("[FAQ] relevant question result: {}".format(result))
                total_result.extend(result)
            #根据score对total_result进行排序
            sorted_result = sorted(total_result, key=lambda x: x[1], reverse=True)
            for i in range(0, len(sorted_result)):
                meta_question = sorted_result[i][0].metadata["question"]
                score = sorted_result[i][1]
                logger.debug("[FAQ] relevant score: {}, meta_question: {}".format(score, meta_question))
                if self.cos_sim_threshold_mid <= score:
                    answers.append(sorted_result[i][0].metadata["answer"])
            #如果依然没有召回答案，找到最接近的问题，提示用户是否在找这个问题
            if len(answers) <= 0 and len(total_result) > 0:
                score = total_result[0][1]
                question = total_result[0][0].metadata["question"]
                if self.cos_sim_threshold_min <= score:
                    answers.append("我猜你是想问这个问题吧。({})".format(question))
                    self.previous_not_hit_questions[session_id] = question
                    logger.debug("[FAQ] insert previous_not_hit_questions. session_id: {}, question: {}".format(session_id, question))
        return answers

    def get_faq(self, session_id, question) -> Reply:
        #get answer
        #answer = self.get_answer_from_embedding(question, self.qa_list, topk=1)
        answer = self.get_answer_from_chroma(session_id, question, topk=1)
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

    def generate_embedding_from_txt_file_chroma(self, file):
        affirmations = self.load_affirmations(file)
        embedding = OpenAIEmbeddings()
        chroma = Chroma(CHROMA_COLLECTION_NAME, embedding, persist_directory=CHROMA_DB_DIR)
        chroma.add_texts(affirmations, meta_datas=None)

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
                model=self.model,
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
                                        "人大提前面试会问些什么问题？"]},
                    {"query":"法大报考条件是什么？",
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
        chain = final_prompt | ChatOpenAI(model_name=self.model, temperature=0.0)
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
    
    def load_select_school_prompt(self, file_path: str):
        df = pd.read_excel(file_path)
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
                self.prompt_list.append(entry)
            else:
                existing_university = next((item for item in self.prompt_list if item["university"] == last_university), None)
                if existing_university is not None:
                    existing_university["few_shot_list"].append({
                        "question": few_shot_question,
                        "answer": few_shot_answer
                    })
    
    def get_university_match_result(self, university: str, user_info: str) -> Reply:
        few_shot_list = []
        for i in range(0, len(self.prompt_list)):
            full_name = self.get_university_full_name(university)
            if len(full_name) <= 0:
                logger.debug("[FAQ] not found university: {}".format(university))
                return Reply(content="还没有这个院校的择校规则信息，因此无法提供该院校的择校服务", type=ReplyType.TEXT)
            if full_name == self.prompt_list[i]["university"]:
                prompt = self.prompt_list[i]["prompt"]
                few_shot_list = self.prompt_list[i]["few_shot_list"]
                break
        if len(few_shot_list) <= 0:
            return Reply(content="该院校没有配置few shots", type=ReplyType.TEXT)
        
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
        chain = final_prompt | ChatOpenAI(model_name=self.model, temperature=0.0)
        output = chain.invoke({"question": user_info})
        return Reply(content=output.content, type=ReplyType.TEXT)
    
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

