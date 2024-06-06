from langchain.embeddings.base import Embeddings
import time
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
import logging
import configparser
from tqdm import tqdm

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s @ %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)
logger = logging.getLogger(name="YaGPT-API")


class YandexGPTEmbeddings():
    def __init__(self, iam_token=None, sleep_interval=1):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.iam_token = iam_token
        self.sleep_interval = sleep_interval
        self.api_key = self.config.get('Security', 'API-key')
        self.folder_id = self.config.get('Security', 'folder-id')
        if self.iam_token:
            self.headers = {'Authorization': 'Bearer ' + self.iam_token}
        if self.api_key:
            self.headers = {'Authorization': 'Api-key ' + self.api_key,
                            "x-folder-id": self.folder_id}

    def embed_document(self, text):
        j = {
            "modelUri": f"emb://{self.folder_id}/text-search-doc/latest",
            "text": text
        }
        res = requests.post(
            url="https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding",
            json=j,
            headers=self.headers
        )
        vec = res.json()['embedding']
        return vec

    def embed_documents(self, texts, chunk_size=0):
        res = []
        for x in tqdm(texts):
            res.append(self.embed_document(x))
            time.sleep(self.sleep_interval)
        return res

    def _generate_promt(self,
                        message: list,
                        stream: bool = False,
                        temperature: float = 0.6,
                        max_tokens: int = 100) -> dict:
        prompt = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
            "completionOptions": {
                "stream": stream,
                "temperature": temperature,
                "maxTokens": f"{max_tokens}"
            },
            "messages": message
        }
        return prompt


    def make_request(self,
                     user_message: list,
                     stream: bool = False,
                     temperature: float = 0.6,
                     max_tokens: int = 100
                     ) -> str:

        try:
            response = requests.post(
                "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                headers=self.headers,
                json=self._generate_promt(user_message,
                                          stream=stream,
                                          temperature=temperature,
                                          max_tokens=max_tokens)
            )
        except requests.RequestException as error:
            logger.error(error)
        else:
            return self._get_answer_text(response.json())
        return ""

    @staticmethod
    def _get_answer_text(answer) -> str:
        if answer:
            try:
                result = answer["result"]["alternatives"][0]["message"]["text"]
            except KeyError as error:
                logger.error(error)
            else:
                return result
        return ""
