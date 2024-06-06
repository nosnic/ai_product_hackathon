import pandas as pd
import json
from YandexGPT_API import YandexGPTEmbeddings
from chromadb import HttpClient, Settings
from time import sleep


class VectorDatabaseBuilder:
    def __init__(self, input_file, gpt_instance, chroma_host, chroma_port):
        self.df_news = None
        self.input_file = input_file
        self.gpt = gpt_instance
        self.chroma_client = HttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection("news")

    def load_and_prepare_data(self):
        self.df_news = pd.read_csv(self.input_file).head(50)
        self.df_news.drop(['title', 'topic', 'date'], axis=1, inplace=True)
        self.df_news['tags'].fillna(' ', inplace=True)

    def generate_embeddings(self):
        self.df_news['embeds'] = self.gpt.embed_documents(self.df_news['tags'].tolist())

    def upsert_to_chroma(self):
        texts = self.df_news["summary"].tolist()
        tags_embeddings = self.df_news["embeds"].tolist()
        ids = self.df_news.index.astype(str).tolist()
        self.collection.upsert(
            ids=ids,
            embeddings=tags_embeddings,
            metadatas=[{"source": "news", "summary": txt} for txt in texts],
            documents=texts
        )

    def query_similar_items(self, task, n_results=6):
        query_embedding = self.gpt.embed_document(task)
        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )
        return result

    def process(self):
        self.load_and_prepare_data()
        self.generate_embeddings()
        self.upsert_to_chroma()
