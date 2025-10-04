from datetime import datetime, timedelta, timezone
from collect_news import fetch_news_sync
FEEDS = [
    "https://rssexport.rbc.ru/rbcnews/news/20/full.rss",
    "https://www.kommersant.ru/RSS/news.xml",
    "https://lenta.ru/rss/top7",
]

since = datetime.now(timezone.utc) - timedelta(days=1)
until = datetime.now(timezone.utc)

# асинхронно
# results = await fetch_news(FEEDS, since=since, until=until)

# синхронно (например, из обработчика кнопки в GUI)
results = fetch_news_sync(FEEDS, since=since, until=until, total_limit=200, article_workers=8)

# results — это list[dict], каждый dict включает текст новости в "text"
print(len(results), "items")
print(results[0])
