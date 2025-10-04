import json
import re
from typing import List

from src.data_struct.news import NewsStruct, NewsStructNE, NamedEntity


class SummarizatorHotness:
    def __init__(self, gpt_model):
        self.gpt_model = gpt_model

    def summarize(self, texts_list: List[str]):
        messages = [
                       {
                           "role": "system",
                           "content": "Тебе даны тексты похожих новостей. Саммаризируй эти новости в одно-два предложения."
                       }
                   ] + [
                       {
                           "role": "user",
                           "content": f"Текст новости: {text}"
                       }
                       for text in texts_list
                   ]
        response_giga_model = self.gpt_model.process(messages)
        return response_giga_model

    def hotness_extractor(self, texts_list: List[str]):
        messages = [
                       {
                           "role": "system",
                           "content": "Тебе даны тексты похожих новостей. Оцени по шкале от 1 до 100 насколько данный набор новостей актуален для финансового аналитика. Ответь, пожалуйста, ТОЛЬКО числом."
                       }
                   ] + [
                       {
                           "role": "user",
                           "content": f"Текст новости: {text}"
                       }
                       for text in texts_list
                   ]
        response_giga_model = self.gpt_model.process(messages)
        return response_giga_model


if __name__ == "__main__":
    from src.models.gigachat_api import GIGACHAT_cstm

    giga_cstm_instance = GIGACHAT_cstm()
    neextr = SummarizatorHotness(giga_cstm_instance)
    news_text = "В четверг утром неизвестный наехал на автомобиле, а затем напал с ножом на людей возле синагоги на улице Миддлтон-роуд в Манчестере . По последним данным, двое пострадавших умерли, еще трое находятся в критическом состоянии. Нападавший был застрелен. Британская полиция предварительно заявила, что нападение устроил 35-летний гражданин Великобритании сирийского происхождения Джихад аль-Шами. Атака произошла в самый священный для иудеев день календаря - Йом Кипур (Судный день), - когда тысячи верующих по всему миру посещают синагоги для молитв."
    result = neextr.hotness_extractor(news_text)
    print(result)
