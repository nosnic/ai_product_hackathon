from langchain.embeddings.base import Embeddings
import time
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
import langchain
import configparser
from tqdm import tqdm


class YandexGPTEmbeddings(Embeddings):
    def __init__(self, iam_token=None, api_key=None, folder_id=None, sleep_interval=1):
        self._config = configparser.ConfigParser()
        self._config.read('config.ini')
        self._request_url = self._config.get('YandexGPT', 'requests-url')
        self._model = self._config.get('YandexGPT', 'model')
        self.iam_token = iam_token
        self.sleep_interval = sleep_interval
        self.api_key = self._config.get('Security', 'API-key')
        self.folder_id = self._config.get('Security', 'folder-id')
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
        res = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding",
                            json=j, headers=self.headers)
        vec = res.json()['embedding']
        return vec

    def embed_documents(self, texts, chunk_size=0):
        res = []
        for x in tqdm(texts):
            res.append(self.embed_document(x))
            time.sleep(self.sleep_interval)
        return res

    def embed_query(self, text):
        j = {
            "model": "general:embedding",
            "embedding_type": "EMBEDDING_TYPE_QUERY",
            "text": text
        }
        res = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding",
                            json=j, headers=self.headers)
        vec = res.json()['embedding']
        time.sleep(self.sleep_interval)
        return vec
