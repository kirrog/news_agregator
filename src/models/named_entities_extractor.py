import json
import re
from typing import List

from src.data_struct.news import NewsStruct, NewsStructNE, NamedEntity


class NEExtractor:
    def __init__(self, gpt_model):
        self.gpt_model = gpt_model

    def extract(self, text: str) -> List[str]:
        messages = [
            {
                "role": "system",
                "content": "Тебе дан текст новости, извлеки из него имена, "
                           "географические и политические названия, названия компаний и прочие. "
                           "Ответ дай в формате списка словарей: '[{'type': 'geo', 'text': 'Russian Federation'}]'"
            },
            {
                "role": "user",
                "content": f"Текст новости: {text}"
            }
        ]
        response_giga_model = self.gpt_model.process(messages)
        r = re.sub("'", "\"", response_giga_model)
        j = json.loads(r)
        return j

    def extract_ne_from_news(self, news_struct: NewsStruct) -> NewsStructNE:
        text = news_struct.header + news_struct.text
        ne_list = self.extract(text)
        ne_structs_list = [NamedEntity(x["type"], x["text"]) for x in ne_list]
        return NewsStructNE(news_struct, ne_structs_list)


if __name__ == "__main__":
    from src.models.gigachat_api import GIGACHAT_cstm

    giga_cstm_instance = GIGACHAT_cstm()
    neextr = NEExtractor(giga_cstm_instance)
    news_text = "В четверг утром неизвестный наехал на автомобиле, а затем напал с ножом на людей возле синагоги на улице Миддлтон-роуд в Манчестере . По последним данным, двое пострадавших умерли, еще трое находятся в критическом состоянии. Нападавший был застрелен. Британская полиция предварительно заявила, что нападение устроил 35-летний гражданин Великобритании сирийского происхождения Джихад аль-Шами. Атака произошла в самый священный для иудеев день календаря - Йом Кипур (Судный день), - когда тысячи верующих по всему миру посещают синагоги для молитв."
    result = neextr.extract(news_text)
    print(result)
