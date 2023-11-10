#encoding: utf-8
import sys
import os
import openai
import csv
import pandas as pd

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


if sys.platform == 'win32':
    CHROMA_DB_DIR = ".\\plugins\\faq\\vectordb\\chroma_db\\"
else:
    CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectordb/chroma_db")
CHROMA_COLLECTION_NAME = "sponger_bot"

class CorpusTools(object):
    def __init__(self) -> None:
        super().__init__()
        # openai.api_key = conf().get("open_ai_api_key")
        # os.environ["OPENAI_API_KEY"] = conf().get("open_ai_api_key")
        openai.api_key = "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"
        os.environ["OPENAI_API_KEY"] = "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"
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(CHROMA_COLLECTION_NAME, 
                                  self.embeddings, 
                                  persist_directory=CHROMA_DB_DIR)
        
    def generate_embedding_from_csv_file_chroma(self, file):
        #read csv file
        reader = csv.reader(open(file, encoding='utf-8'))
        self.generate_embedding_from_csv_reader(reader)

    def generate_embedding_from_csv_reader(self, reader):
        meta_datas = []
        texts = []
        csv_data = [row for row in reader]
        for row in csv_data:
            questions = row[0]
            answer = row[1]
            question_list = questions.split("；")
            for j in range(0, len(question_list)):
                question = question_list[j]
                if len(question) == 0 or len(answer) == 0 or question == "nan" or answer == "nan":
                    continue
                dict = {"question": question, "answer": answer}
                meta_datas.append(dict)
                texts.append(question)
        #add texts to vectorstore
        self.vectorstore.add_texts(texts, meta_datas)

    def update_chroma_text(self, question: str, answer: str):
        if len(question) == 0 or len(answer) == 0 or question == "nan" or answer == "nan":
            return
        
        meta_datas = [
            {"question": question, "answer": answer}
        ]
        ret_value = self.vectorstore.add_texts([question], meta_datas)
        print("add texts to vectorstore: {}".format(ret_value))

    def delete_chroma_text(self, question: str):
        #TODO delete text from vectorstore
        pass
    
    def search_chroma_text(self, question: str):
        doc_list = self.vectorstore.similarity_search(question, 1)
        if doc_list:
            return doc_list[0].metadata["answer"]
    #merge and convert
    def merge_excel_files(self, dir='.', out_file='merged.csv'):
        # 列名列表
        columns = ["prompt", "completion", "tags"]
        all_data = pd.DataFrame()
        # 遍历当前目录下的所有xlsx文件
        for file in os.listdir(dir):
            if file.endswith(".xlsx"):
                # 读取Excel文件，不使用文件中的header，而是使用自定义的列名
                df = pd.read_excel(os.path.join(dir, file), header=None, names=columns)
                # 将数据添加到all_data中
                all_data = pd.concat([all_data, df], ignore_index=True)
    
        # 将合并后的数据写入csv文件
        all_data.to_csv(os.path.join(dir, out_file), index=False, encoding='utf-8')
    
    def excel2csv(self, in_file, out_file):
        # 读取excel文件
        df = pd.read_excel(in_file)
    
        # 合并两列，用";"隔开
        df.loc[df[df.columns[2]].notna(), df.columns[0]] = df[df.columns[0]].astype(str) + "；" + df[df.columns[2]].astype(str)
        # 删除第三列
        df = df.drop(df.columns[2], axis=1)
        # 将结果写入新的csv文件
        df.to_csv(out_file, index=False, encoding='utf-8')

corpus_tools = CorpusTools()


#write main while menu test
if __name__ == '__main__':
    curdir = os.path.dirname(__file__)

    while True:
        print("1. merge excel files")
        print("2. excel to csv")
        print("3. generate embedding from csv file")
        print("4. update chroma text")
        print("5. delete chroma text")
        print("6. exit")
        choice = input("please input your choice:")
        if choice == "1":
            corpus_tools.merge_excel_files(dir=os.path.join(curdir, "knowledge_base", "interview_services"), out_file='merged.csv')
        elif choice == "2":
            in_file = input("please input excel file path:")
            out_file = input("please input csv file path:")
            corpus_tools.excel2csv(os.path.join(curdir, "knowledge_base", "interview_services", in_file), 
                                   os.path.join(curdir, "knowledge_base", "interview_services", out_file))
        elif choice == "3":
            csv_file = input("please input csv file path:")
            corpus_tools.generate_embedding_from_csv_file_chroma(os.path.join(curdir, "knowledge_base", "interview_services", csv_file))
        elif choice == "4":
            question = input("please input question:")
            answer = input("please input answer:")
            corpus_tools.update_chroma_text(question, answer)
        elif choice == "5":
            question = input("please input question:")
            corpus_tools.delete_chroma_text(question)
        elif choice == "6":
            break
        else:
            print("invalid input, please input again")
