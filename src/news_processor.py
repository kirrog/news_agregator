from collections import defaultdict

from src.data_struct.news import NewsStruct


class NewsProcessor:

    def __init__(self, clusterer):
        self.clusterer = clusterer
        self.clusters_dict = defaultdict(list)
        self.clusters_analysed_dict = defaultdict(dict)

    def process_news(self, news_struct: NewsStruct):
        return None

    def cluster_news(self, news_struct):
        cluster = self.clusterer.find_cluster(news_struct)
        self.clusters_dict[cluster].append(news_struct)
        return cluster

    # Iterate through clusters and find duplicates, narrative development and summarise
    def cluster_analysis(self):
        return None

    def ne_extraction(self, news):
        return news

    def classify_news(self, news):
        return news

    def news_influence_prediction(self, news):
        return news

    def news_cluster_influence_prediction(self, news_cluster):
        return news_cluster

    def news_cluster_summarization(self, news_cluster):
        return {
            "headline": "",
            "hotness": 0.5,
            "why_now": "",
            "entities": [],
            "sources": [],
            "timeline": "",
            "draft": "",
            "dedup_group": -1
        }
