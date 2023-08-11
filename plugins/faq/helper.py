import json
import time
import openai
#import config
import openai.embeddings_utils as em_utils
from openai.embeddings_utils import get_embedding, cosine_similarity

OPEN_AI_MODEL3 = "gpt-3.5-turbo"
OPEN_AI_MODEL4 = "gpt-4"
DEFAULT_RETRY_COUNT = 3
#write a function 请求openai chatcomplition ，只返回正确的json
def json_gpt(input:str)->json:
    completion = openai.ChatCompletion.create(
        model=OPEN_AI_MODEL3,
        messages=[
            {"role": "system", "content": "Output only valid JSON"},
            {"role": "user", "content": input},
        ],
        temperature=0.5,
        max_tokens=2000
    )

    text = completion.choices[0].message.content
    parsed = json.loads(text)

    return parsed

def touch_up_the_text(input:str, retry_count)->str:
    try:
        completion = openai.ChatCompletion.create( 
            model=OPEN_AI_MODEL4,
            messages=[
                {"role": "system", "content": "你是一名mba面试咨询老师，请你把下面一段关于面试相关的回答润色下，让它更加流畅，更加符合面试官的口语习惯。输出语言简体中文"},
                {"role": "user", "content": input},
            ],
            temperature=0.5,
            max_tokens=2000
        )
        text = completion.choices[0]["message"]["content"]
        return text
    except Exception as e:
        need_retry = retry_count < DEFAULT_RETRY_COUNT
        if isinstance(e, openai.error.RateLimitError):
            #logger.warn("[OPEN_AI] RateLimitError: {}".format(e))
            if need_retry:
                time.sleep(20)
        elif isinstance(e, openai.error.Timeout):
            #logger.warn("[OPEN_AI] Timeout: {}".format(e))
            if need_retry:
                time.sleep(5)
        elif isinstance(e, openai.error.APIConnectionError):
            #logger.warn("[OPEN_AI] APIConnectionError: {}".format(e))
            need_retry = False
        else:
            #logger.warn("[OPEN_AI] Exception: {}".format(e))
            need_retry = False

        if need_retry:
            #logger.warn("[OPEN_AI] 第{}次重试".format(retry_count + 1))
            return touch_up_the_text(input, retry_count)
        else:
            return ''
