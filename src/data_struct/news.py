from datetime import datetime
from typing import List, Dict

from numpy import ndarray


class NewsStruct:

    def __init__(self, date_time_: datetime,
                 source_link: str,
                 header: str,
                 text: str):
        self.date_time_ = date_time_
        self.source_link = source_link
        self.header = header
        self.text = text

    def __str__(self):
        return (
            f"date_time_: {self.date_time_}\n"
            f"source_link: {self.source_link}\n"
            f"header: {self.header}\n"
            f"text: {self.text}\n"
        )


class NamedEntity:
    def __init__(self, name_type: str, name_text: str):
        self.name_type = name_type
        self.name_text = name_text

    def __str__(self):
        return (
            f"name_type: {self.name_type}\n"
            f"name_text: {self.name_text}\n"
        )


class NewsStructNE(NewsStruct):

    def __init__(self, news_struct: NewsStruct, named_entities: List[NamedEntity]):
        self.date_time_ = news_struct.date_time_
        self.source_link = news_struct.source_link
        self.header = news_struct.header
        self.text = news_struct.text
        self.named_entities = named_entities

    def __str__(self):
        return (
            f"date_time_: {self.date_time_}\n"
            f"source_link: {self.source_link}\n"
            f"header: {self.header}\n"
            f"text: {self.text}\n"
            f"named_entities: {self.named_entities}\n"
        )


class IndustryEntity:
    def __init__(self, industry_name, industry_forecast):
        self.industry_name = industry_name
        self.industry_forecast = industry_forecast

    def __str__(self):
        return (
            f"industry_name: {self.industry_name}\n"
            f"industry_forecast: {self.industry_forecast}\n"
        )


class CompaniesEntity:
    def __init__(self, company_name, company_forecast):
        self.company_name = company_name
        self.company_forecast = company_forecast

    def __str__(self):
        return (
            f"company_name: {self.company_name}\n"
            f"company_forecast: {self.company_forecast}\n"
        )


class NewsStructCompany(NewsStructNE):

    def __init__(self, news_struct_ne: NewsStructNE, industry_list: List[IndustryEntity],
                 companies_names_list: List[CompaniesEntity],
                 companies_tickers_list: List[str]):
        self.date_time_ = news_struct_ne.date_time_
        self.source_link = news_struct_ne.source_link
        self.header = news_struct_ne.header
        self.text = news_struct_ne.text
        self.named_entities = news_struct_ne.named_entities
        self.industry_list = industry_list
        self.companies_names_list = companies_names_list
        self.companies_tickers_list = companies_tickers_list

    def __str__(self):
        return (
            f"date_time_: {self.date_time_}\n"
            f"source_link: {self.source_link}\n"
            f"header: {self.header}\n"
            f"text: {self.text}\n"
            f"named_entities: {self.named_entities}\n"
            f"industry_list: {self.industry_list}\n"
            f"companies_names_list: {self.companies_names_list}\n"
            f"companies_tickers_list: {self.companies_tickers_list}\n"
        )


class NewsStructEmbed(NewsStructCompany):

    def __init__(self, news_struct_company: NewsStructCompany, embedding: ndarray):
        self.date_time_ = news_struct_company.date_time_
        self.source_link = news_struct_company.source_link
        self.header = news_struct_company.header
        self.text = news_struct_company.text
        self.named_entities = news_struct_company.named_entities
        self.industry_list = news_struct_company.industry_list
        self.companies_names_list = news_struct_company.companies_names_list
        self.companies_tickers_list = news_struct_company.companies_tickers_list
        self.embedding = embedding

    def __str__(self):
        return (
            f"date_time_: {self.date_time_}\n"
            f"source_link: {self.source_link}\n"
            f"header: {self.header}\n"
            f"text: {self.text}\n"
            f"named_entities: {self.named_entities}\n"
            f"industry_list: {self.industry_list}\n"
            f"companies_names_list: {self.companies_names_list}\n"
            f"companies_tickers_list: {self.companies_tickers_list}\n"
            f"embedding: {self.embedding.mean()}\n"
        )
