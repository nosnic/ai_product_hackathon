import pandas as pd
from time import sleep


class NewsProcessor:
    def __init__(self, input_file, output_file, gpt_instance):
        self.input_file = input_file
        self.output_file = output_file
        self.gpt = gpt_instance
        self.news = pd.read_csv(self.input_file).head(10)
        self.summaries = []
        self.tags = []

    def summarize_texts(self):
        for text in self.news['text']:
            sleep(1)
            message = [
                {
                    "role": "system",
                    "text": "Ты лучший редактор на свете, выведи краткое содержание статьи снизу"
                },
                {
                    "role": "user",
                    "text": text
                }
            ]
            try:
                answer = self.gpt.make_request(message, temperature=0.1, max_tokens=300)
            except Exception as e:
                print(f"Error occurred: {e}")
                answer = ''
            self.summaries.append(answer)

    def generate_tags(self):
        for text in self.news['text']:
            sleep(1)
            message = [
                {
                    "role": "system",
                    "text": "Ты лучший редактор на свете, выведи 5 слов через пробел которые лучше всего описывают "
                            "тему новости снизу"
                },
                {
                    "role": "user",
                    "text": text
                }
            ]
            try:
                answer = self.gpt.make_request(message, temperature=0.1, max_tokens=50)
            except Exception as e:
                print(f"Error occurred: {e}")
                answer = ''
            self.tags.append(answer)

    def process_news(self):
        self.summarize_texts()
        self.generate_tags()
        self.news['summary'] = self.summaries
        self.news['tags'] = self.tags
        self.news.to_csv(self.output_file, index=False)
