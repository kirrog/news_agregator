import datetime
import json
import re
from typing import List, Dict

from src.data_struct.news import NewsStructCompany, NewsStructNE, IndustryEntity, CompaniesEntity, NamedEntity, \
    NewsStruct


class CompanyClassificator:
    def __init__(self, gpt_model, tickers: Dict[str, str]):
        self.gpt_model = gpt_model
        self.tickers_dict = tickers

    def extract_company(self, text: str) -> List[str]:
        messages = [
            {
                "role": "system",
                "content": "Тебе дан текст новости, определи, на какие компании событие новости может повлиять, "
                           "и как (позитивно или негативно)."
                           "Ответ дай в формате списка словарей: '[{'company': 'Sber', 'forecast': 'positive'}]"
            },
            {
                "role": "user",
                "content": f"Текст новости: {text}"
            }
        ]
        response_giga_model = self.gpt_model.process(messages)
        r = re.sub("'", "\"", response_giga_model)
        j = json.loads(r)
        for company in j:
            company_name = company["company"]
            ticker = ""
            if company_name in self.tickers_dict:
                ticker = self.tickers_dict[company_name]
            if company_name.lower() in self.tickers_dict:
                ticker = self.tickers_dict[company_name.lower()]
            company["ticker"] = ticker
        return j

    def extract_industry(self, text: str) -> List[str]:
        messages = [
            {
                "role": "system",
                "content": "Тебе дан текст новости, определи, на какие области экономики эта новость может повлиять, "
                           "и как (позитивно или негативно)."
                           "Ответ дай в формате списка словарей: '[{'type': 'Gas', 'forecast': 'negative'}]'"
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

    def extract(self, news_struct: NewsStructNE) -> NewsStructCompany:
        company = self.extract_company(f"{news_struct.header}\n{news_struct.text}")
        industry = self.extract_industry(f"{news_struct.header}\n{news_struct.text}")
        news_struct_result = NewsStructCompany(
            news_struct,
            [IndustryEntity(x["type"], x["forecast"]) for x in industry],
            [CompaniesEntity(x["type"], x["forecast"]) for x in company],
            [x["ticker"] for x in company]
        )
        return news_struct_result


if __name__ == "__main__":
    from src.models.gigachat_api import GIGACHAT_cstm

    cnames2tickers = dict()
    with open("./moex_ru_shares.json", "r", encoding="utf-8") as f:
        tickers2names = json.load(f)
        for ticker, cname in tickers2names.items():
            cnames2tickers[cname] = ticker
            cnames2tickers[cname.lower()] = ticker

    giga_cstm_instance = GIGACHAT_cstm()
    company_classificator = CompanyClassificator(giga_cstm_instance, cnames2tickers)
    news_text = "В четверг утром неизвестный наехал на автомобиле, а затем напал с ножом на людей возле синагоги на улице Миддлтон-роуд в Манчестере . По последним данным, двое пострадавших умерли, еще трое находятся в критическом состоянии. Нападавший был застрелен. Британская полиция предварительно заявила, что нападение устроил 35-летний гражданин Великобритании сирийского происхождения Джихад аль-Шами. Атака произошла в самый священный для иудеев день календаря - Йом Кипур (Судный день), - когда тысячи верующих по всему миру посещают синагоги для молитв."
    news_struct_ne = NewsStructNE(
        NewsStruct(datetime.datetime.now(), "", "", news_text), []
    )
    result = company_classificator.extract(news_struct_ne)
    print(result)
