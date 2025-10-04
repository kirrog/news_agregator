from typing import List

from sentence_transformers import SentenceTransformer

from src.data_struct.news import NewsStructCompany, NewsStructEmbed


# pip install -U sentence-transformers


class EmbeddingsExtractor:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def extract(self, texts_list: List[str]):
        embeds = self.model.encode(texts_list)
        return [x for x in embeds]

    def extract_from_news(self, news_struct: NewsStructCompany) -> NewsStructEmbed:
        embed = self.extract([str(news_struct)])[0]
        return NewsStructEmbed(news_struct, embed)

    def extract_from_news_list(self, news_structs_list: List[NewsStructCompany]) -> List[NewsStructEmbed]:
        news_str_list = [str(x) for x in news_structs_list]
        news_embeds_list = self.extract(news_str_list)
        news_struct_embed_list = []
        for news_struct, news_embed in zip(news_structs_list, news_embeds_list):
            news_struct_emb = NewsStructEmbed(news_struct, news_embed)
            news_struct_embed_list.append(news_struct_emb)
        return news_struct_embed_list


if __name__ == "__main__":
    sentences = ["This is an example sentence", "Each sentence is converted"]
    emb_extr = EmbeddingsExtractor()
    embeddings = emb_extr.extract(sentences)
    print(embeddings)
