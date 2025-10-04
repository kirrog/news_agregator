# app.py
# pip install streamlit pandas python-dateutil

import json
import time
from datetime import datetime, timedelta, timezone
from dateutil import tz
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

from collect_news import fetch_news_sync  # функции из вашего collect_news.py

st.set_page_config(page_title="News Collector", layout="wide")

TZ = tz.gettz("Europe/Berlin")

# --- helpers ---
def _combine_local_to_utc(d, t):
    local_dt = datetime.combine(d, t).replace(tzinfo=TZ)
    return local_dt.astimezone(timezone.utc)

def _init_state():
    if "feeds" not in st.session_state:
        st.session_state.feeds = [
            "https://rssexport.rbc.ru/rbcnews/news/20/full.rss",
            "https://www.kommersant.ru/RSS/news.xml",
            "https://lenta.ru/rss/top7",
        ]
    if "results" not in st.session_state:
        st.session_state.results = []

_init_state()

st.title("News Collector GUI")

# --- SIDEBAR: настройки в сворачиваемых разделах ---
with st.sidebar:
    st.header("Настройки")

    # Источники с добавлением/удалением
    with st.expander("Источники", expanded=True):
        new_feed = st.text_input("Добавить источник", placeholder="https://example.com/rss.xml")
        cols = st.columns([1,1])
        with cols[0]:
            if st.button("Добавить", use_container_width=True, disabled=not new_feed.strip()):
                url = new_feed.strip()
                if url not in st.session_state.feeds:
                    st.session_state.feeds.append(url)
                st.experimental_rerun()
        with cols[1]:
            if st.button("Очистить все", use_container_width=True, type="secondary", disabled=not st.session_state.feeds):
                st.session_state.feeds = []
                st.experimental_rerun()

        st.caption("Текущие источники:")
        # список с кнопками удаления
        if st.session_state.feeds:
            for i, feed in enumerate(list(st.session_state.feeds)):
                c1, c2 = st.columns([8,2])
                c1.write(feed)
                if c2.button("Удалить", key=f"del_{i}"):
                    st.session_state.feeds.pop(i)
                    st.experimental_rerun()
        else:
            st.caption("Список пуст.")

        # Импорт/замена списком
        with st.expander("Массовый импорт/замена"):
            bulk = st.text_area("По одной ссылке в строке")
            cc1, cc2 = st.columns(2)
            if cc1.button("Заменить список"):
                feeds_new = [x.strip() for x in bulk.splitlines() if x.strip()]
                st.session_state.feeds = list(dict.fromkeys(feeds_new))
                st.experimental_rerun()
            if cc2.button("Добавить к списку"):
                feeds_new = [x.strip() for x in bulk.splitlines() if x.strip()]
                merged = st.session_state.feeds + feeds_new
                # уникализация с сохранением порядка
                st.session_state.feeds = list(dict.fromkeys(merged))
                st.experimental_rerun()

    # Период
    with st.expander("Период", expanded=True):
        now_local = datetime.now(TZ)
        since_date = st.date_input("С даты", value=(now_local - timedelta(days=1)).date(), key="since_d")
        since_time = st.time_input("С времени", value=(now_local - timedelta(hours=24)).time(), key="since_t")
        until_date = st.date_input("По дату", value=now_local.date(), key="until_d")
        until_time = st.time_input("По времени", value=now_local.time(), key="until_t")

    # Параметры
    with st.expander("Параметры", expanded=False):
        total_limit = st.slider("TOTAL_LIMIT", 100, 8000, 1000, step=100)
        per_feed_limit = st.slider("PER_FEED_LIMIT", 50, 2000, 500, step=50)
        article_workers = st.slider("ARTICLE_WORKERS", 1, 32, 12, step=1)
        max_per_domain = st.slider("MAX_PER_DOMAIN", 50, 2000, 800, step=50)
        title_sim_threshold = st.slider("TITLE_SIM_THRESHOLD", 70, 100, 92, step=1)

    run = st.button("Собрать новости", type="primary", use_container_width=True)

# --- Вводные данные ---
feeds = st.session_state.feeds
since_dt = _combine_local_to_utc(since_date, since_time)
until_dt = _combine_local_to_utc(until_date, until_time)

# --- Лоадер со статусом и прогрессом (работает при синхронной функции) ---
if run:
    if not feeds:
        st.error("Добавьте хотя бы один источник.")
    elif since_dt >= until_dt:
        st.error("Начало периода должно быть раньше конца.")
    else:
        status = st.empty()
        bar = st.progress(0)
        note = st.empty()

        def _task():
            return fetch_news_sync(
                feeds,
                since=since_dt,
                until=until_dt,
                per_feed_limit=per_feed_limit,
                total_limit=total_limit,
                article_workers=article_workers,
                max_per_domain=max_per_domain,
                title_sim_threshold=title_sim_threshold,
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_task)
            phase = 0
            phases = ["Получение лент", "Фильтрация и дедупликация", "Загрузка текстов статей"]
            # Индикатор пока задача не завершена
            while not future.done():
                phase = (phase + 1) % len(phases)
                status.markdown(f"**{phases[phase]}...**")
                # пульсирующий прогресс
                for p in range(0, 101, 10):
                    if future.done():
                        break
                    bar.progress(p)
                    time.sleep(0.08)
                note.caption("Работаем. Это может занять время при большом числе источников.")
            try:
                st.session_state.results = future.result()
            finally:
                bar.progress(100)
                status.markdown("**Готово**")
                note.empty()

        st.success(f"Найдено: {len(st.session_state.results)}")

# --- Новости ---
st.subheader("Новости")
results = st.session_state.results
if results:
    df = pd.DataFrame(results)
    for i, row in df.iterrows():
        title = row.get("title") or "Без заголовка"
        with st.expander(f"{i+1}. {title}", expanded=False):
            st.markdown(f"**Опубликовано:** {row.get('published')}")
            st.markdown(f"**Источник:** {row.get('source')}")
            if row.get("url"):
                st.markdown(f"[Ссылка на материал]({row.get('url')})")
            if row.get("error"):
                st.caption(f"Ошибка разбора: {row['error']}")
            st.text_area("Текст", row.get("text") or "", height=280, key=f"text_{i}")
else:
    st.caption("Нет данных. Задайте источники и период, затем нажмите «Собрать новости».")

# --- Экспорт ---
with st.expander("Экспорт", expanded=False):
    if results:
        json_bytes = json.dumps(results, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("Скачать JSON", data=json_bytes, file_name="news.json", mime="application/json")
        df = pd.DataFrame(results)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Скачать CSV (все поля)", data=csv_bytes, file_name="news.csv", mime="text/csv")
    else:
        st.caption("Экспорт недоступен. Сначала соберите новости.")
