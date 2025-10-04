import json
from collections import defaultdict
from typing import List

from sklearn.cluster import DBSCAN
from tqdm import tqdm

from src.data_struct.news import NewsStruct, NewsStructNE, NewsStructEmbed, NewsStructCompany
from src.models.company_extractor import CompanyClassificator
from src.models.embeddings_extractor import EmbeddingsExtractor
from src.models.gigachat_api import GIGACHAT_cstm
from src.models.named_entities_extractor import NEExtractor
from src.models.summurizator import SummarizatorHotness


class NewsProcessor:

    def __init__(self):
        self.clusterer = DBSCAN(eps=3, min_samples=2)
        self.clusters_dict = defaultdict(list)
        self.clusters_analysed_dict = defaultdict(dict)
        self.giga_cstm_instance = GIGACHAT_cstm()
        self.neextr = NEExtractor(self.giga_cstm_instance)
        cnames2tickers = dict()
        with open("./models/moex_ru_shares.json", "r", encoding="utf-8") as f:
            tickers2names = json.load(f)
            for ticker, cname in tickers2names.items():
                cnames2tickers[cname] = ticker
                cnames2tickers[cname.lower()] = ticker
        self.company_classificator = CompanyClassificator(self.giga_cstm_instance, cnames2tickers)
        self.emb_extr = EmbeddingsExtractor()
        self.hotness_analyser_summarizer = SummarizatorHotness(self.giga_cstm_instance)

    def process_news(self, news_structs_list: List[NewsStruct]):
        news_struct_ne_list = self.extract_ne_news(news_structs_list)
        news_struct_classified_list = self.classify_news(news_struct_ne_list)
        news_struct_embeds_list = self.extract_embeddings(news_struct_classified_list)

        self.cluster_news(news_struct_embeds_list)

        cstm_clusters_dict = dict()
        for label, cluster_list in self.clusters_dict.items():
            cstm_clusters_dict[label] = [str(x) for x in cluster_list]

        with open("clusters_summarization.json", "w", encoding="utf-8") as f:
            json.dump(cstm_clusters_dict, f, ensure_ascii=False)

        self.cluster_analysis()
        news_clusters_formated_list = self.news_clusters_formater()

        with open("news_clusters_formated_list.json", "w", encoding="utf-8") as f:
            json.dump(news_clusters_formated_list, f, ensure_ascii=False)
        return news_clusters_formated_list

    def extract_ne_news(self, news_structs_list: List[NewsStruct]) -> List[NewsStructNE]:
        news_struct_ne_list = list()
        for news_struct in tqdm(news_structs_list, desc="NE extraction"):
            news_struct_ne = self.neextr.extract_ne_from_news(news_struct)
            news_struct_ne_list.append(news_struct_ne)
        return news_struct_ne_list

    def classify_news(self, news_struct_ne_list: List[NewsStructNE]) -> List[NewsStructCompany]:
        news_struct_classified_list = list()
        for news_struct_ne in tqdm(news_struct_ne_list, desc="Company and industry Classification"):
            news_struct_classified = self.company_classificator.extract(news_struct_ne)
            news_struct_classified_list.append(news_struct_classified)
        return news_struct_classified_list

    def extract_embeddings(self, news_struct_classified_list: List[NewsStructCompany]) -> List[NewsStructEmbed]:
        news_struct_embeds_list = list()
        for news_struct_classified in tqdm(news_struct_classified_list, desc="Extract embeddings"):
            news_struct_embed = self.emb_extr.extract_from_news(news_struct_classified)
            news_struct_embeds_list.append(news_struct_embed)
        return news_struct_embeds_list

    def cluster_news(self, news_structs: List[NewsStructEmbed]):
        embeddings_list = [x.embedding for x in news_structs]
        print("Start clustering")
        labels = self.clusterer.fit_predict(embeddings_list)
        print("Clustering complete")
        self.news_structs_embed_list = news_structs
        self.news_structs_labels_list = labels
        for news_structs_embed, news_structs_labels in zip(news_structs, labels):
            self.clusters_dict[int(news_structs_labels)].append(news_structs_embed)

    def cluster_analysis(self):
        for label, cluster_list in tqdm(self.clusters_dict.items(), total=len(self.clusters_dict), desc="Summarizing"):
            texts_list = [x.header + "\n" + x.text for x in cluster_list]
            summarization = self.hotness_analyser_summarizer.summarize(texts_list)
            hottness = self.hotness_analyser_summarizer.hotness_extractor(texts_list)
            self.clusters_analysed_dict[label] = {
                "cluster_list": cluster_list,
                "summarization": summarization,
                "hotness": hottness
            }

    def news_clusters_formater(self):
        results_list = []
        for label, cluster_dict in tqdm(self.clusters_analysed_dict.items(), total=len(self.clusters_dict),
                                        desc="Summarizing"):
            etities_list = []
            for news_struct_ in cluster_dict["cluster_list"]:
                etities_list.extend(news_struct_.named_entities)

            results_list.append({
                "headline": cluster_dict["summarization"],
                "hotness": cluster_dict["hotness"],
                "why_now": "",
                "entities": list(set([str(x) for x in etities_list])),
                "sources": [x.date_time_ for x in cluster_dict["cluster_list"]],
                "timeline": [x.source_link for x in cluster_dict["cluster_list"]],
                "draft": "",
                "dedup_group": label
            })
        return results_list


if __name__ == "__main__":
    news_structs_list = []
    with open("../news_latest.json", "r", encoding="utf-8") as f:
        news_list = json.load(f)
        for news in news_list:
            news_struct = NewsStruct(news["published"], news["url"], news["title"], news["text"])
            news_structs_list.append(news_struct)
    np = NewsProcessor()
    np.process_news(news_structs_list[:10])
