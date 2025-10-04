# app.py
# pip install streamlit pandas python-dateutil
import json
import time
from datetime import datetime, timedelta, timezone
from dateutil import tz
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

# сбор новостей
from collect_news import fetch_news_sync
# обработка новостей
from src.news_precessor import NewsProcessor
from src.data_struct.news import NewsStruct

st.set_page_config(page_title="News Collector", layout="wide")
TZ = tz.gettz("Europe/Berlin")

# -------- helpers --------
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
    if "clusters" not in st.session_state:
        st.session_state.clusters = []
    if "processor_ready" not in st.session_state:
        st.session_state.processor_ready = False

_init_state()

st.title("News Collector GUI")

# -------- SIDEBAR: настройки (сворачиваемые разделы) --------
with st.sidebar:
    st.header("Настройки")

    # Источники
    with st.expander("Источники", expanded=True):
        new_feed = st.text_input("Добавить источник", placeholder="https://example.com/rss.xml")
        c_add, c_clear = st.columns([1,1])
        with c_add:
            if st.button("Добавить", use_container_width=True, disabled=not new_feed.strip()):
                url = new_feed.strip()
                if url and url not in st.session_state.feeds:
                    st.session_state.feeds.append(url)
                st.experimental_rerun()
        with c_clear:
            if st.button("Очистить все", use_container_width=True, type="secondary", disabled=not st.session_state.feeds):
                st.session_state.feeds = []
                st.experimental_rerun()

        st.caption("Текущие источники:")
        if st.session_state.feeds:
            for i, feed in enumerate(list(st.session_state.feeds)):
                c1, c2 = st.columns([8,2])
                c1.write(feed)
                if c2.button("Удалить", key=f"del_{i}"):
                    st.session_state.feeds.pop(i)
                    st.experimental_rerun()
        else:
            st.caption("Список пуст.")

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

    run_collect = st.button("Собрать новости", type="primary", use_container_width=True)
    run_process = st.button("Обработать новости (кластеризация)", use_container_width=True)

# -------- входные --------
feeds = st.session_state.feeds
since_dt = _combine_local_to_utc(since_date, since_time)
until_dt = _combine_local_to_utc(until_date, until_time)

# -------- сбор новостей с лоадером --------
if run_collect:
    if not feeds:
        st.error("Добавьте хотя бы один источник.")
    elif since_dt >= until_dt:
        st.error("Начало периода должно быть раньше конца.")
    else:
        status = st.empty()
        bar = st.progress(0)
        note = st.empty()

        def _collect_task():
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
            future = pool.submit(_collect_task)
            phase = 0
            phases = ["Получение лент", "Фильтрация", "Дедупликация", "Загрузка текстов"]
            while not future.done():
                phase = (phase + 1) % len(phases)
                status.markdown(f"**{phases[phase]}...**")
                for p in range(0, 101, 10):
                    if future.done():
                        break
                    bar.progress(p)
                    time.sleep(0.06)
                note.caption("Идёт сбор. При большом числе источников это дольше обычного.")
            try:
                st.session_state.results = future.result()
                st.session_state.processor_ready = False
                st.session_state.clusters = []
            finally:
                bar.progress(100)
                status.markdown("**Готово**")
                note.empty()

        st.success(f"Найдено: {len(st.session_state.results)}")

# -------- обработка новостей (кластеризация) --------
if run_process:
    if not st.session_state.results:
        st.error("Сначала соберите новости.")
    else:
        proc_status = st.empty()
        proc_bar = st.progress(0)
        proc_note = st.empty()

        def _process_task(items: list[dict]):
            # конвертация в NewsStruct
            news_structs = []
            for n in items:
                news_structs.append(
                    NewsStruct(
                        n.get("published"),
                        n.get("url"),
                        n.get("title"),
                        n.get("text"),
                    )
                )
            np = NewsProcessor()
            # процессор внутри пишет свои промежуточные файлы; возвращает список кластеров
            return np.process_news(news_structs)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_process_task, st.session_state.results)
            ticks = 0
            while not future.done():
                ticks = (ticks + 10) % 100
                proc_status.markdown("**Анализ кластеров...**")
                proc_bar.progress(ticks)
                proc_note.caption("NE, классификация компаний, эмбеддинги, DBSCAN, суммаризация…")
                time.sleep(0.2)

            try:
                st.session_state.clusters = future.result()
                st.session_state.processor_ready = True
            finally:
                proc_bar.progress(100)
                proc_status.markdown("**Готово**")
                proc_note.empty()

        st.success(f"Кластеров: {len(st.session_state.clusters)}")

# -------- раздел Новости --------
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
    st.caption("Нет данных. Сначала соберите новости.")

# -------- раздел Кластеры --------
st.subheader("Кластеры")
clusters = st.session_state.clusters
if clusters:
    # clusters: [{"headline","hotness","why_now","entities","sources","timeline","draft","dedup_group"}]
    cdf = pd.DataFrame(clusters)
    for idx, row in cdf.iterrows():
        with st.expander(f"{idx+1}. {row.get('headline') or 'Без заголовка'}", expanded=False):
            st.markdown(f"**Hotness:** {row.get('hotness')}")
            if row.get("why_now"):
                st.markdown(f"**Why now:** {row.get('why_now')}")
            entities = row.get("entities") or []
            if entities:
                st.markdown("**Сущности:** " + ", ".join(map(str, entities)))
            # Источники и таймлайн
            srcs = row.get("sources") or []
            tln = row.get("timeline") or []
            if srcs:
                st.markdown("**Даты публикаций:**")
                st.code("\n".join(map(str, srcs)), language="text")
            if tln:
                st.markdown("**Ссылки таймлайна:**")
                for url in tln:
                    st.markdown(f"- {url}")
            if row.get("draft"):
                st.markdown("**Черновик:**")
                st.code(row.get("draft"))

    with st.expander("Экспорт кластеров", expanded=False):
        clusters_json = json.dumps(clusters, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("Скачать JSON кластеров", data=clusters_json, file_name="news_clusters.json", mime="application/json")
else:
    st.caption("Кластеров нет. Нажмите «Обработать новости (кластеризация)».")

# -------- Экспорт новостей --------
with st.expander("Экспорт новостей", expanded=False):
    if results:
        json_bytes = json.dumps(results, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("Скачать JSON новостей", data=json_bytes, file_name="news.json", mime="application/json")
        df = pd.DataFrame(results)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Скачать CSV новостей", data=csv_bytes, file_name="news.csv", mime="text/csv")
    else:
        st.caption("Экспорт недоступен. Сначала соберите новости.")
