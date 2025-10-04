# pip install aiohttp feedparser newspaper3k tqdm beautifulsoup4 lxml readability-lxml rapidfuzz requests

import asyncio
import time
import re
import urllib.parse as ul
from calendar import timegm
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from newspaper import Article
from readability import Document
from rapidfuzz.fuzz import partial_ratio

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# ---------- helpers ----------

def _entry_timestamp(e):
    st = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
    return timegm(st) if st else 0

def _entry_iso(e):
    st = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", st) if st else None

def _normalize_url(u: str | None) -> str | None:
    if not u:
        return u
    try:
        pr = ul.urlsplit(u)
        q = ul.parse_qsl(pr.query, keep_blank_values=True)
        q = [(k, v) for (k, v) in q if not re.match(r'^(utm_|yclid|gclid|fbclid)', k, re.I)]
        pr = pr._replace(query=ul.urlencode(q, doseq=True), fragment="")
        return ul.urlunsplit(pr)
    except Exception:
        return u

def _domain(u: str) -> str:
    try:
        return ul.urlsplit(u).netloc.lower()
    except Exception:
        return "unknown"

async def _get(session, url, timeout=20):
    for attempt in range(3):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                content = await r.read()
                ctype = r.headers.get("Content-Type","").lower()
                return content, ctype, None
        except Exception as e:
            if attempt == 2:
                return None, None, str(e)
            await asyncio.sleep(0.6 * (2**attempt))
    return None, None, "unknown_error"

def _discover_rss_from_html(url: str, html: bytes) -> list[str]:
    try:
        soup = BeautifulSoup(html, "lxml")
        out = []
        for link in soup.find_all("link", attrs={"rel": ["alternate","ALTERNATE"]}):
            t = (link.get("type") or "").lower()
            if "rss" in t or "xml" in t or "atom" in t:
                href = link.get("href")
                if href:
                    out.append(ul.urljoin(url, href))
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r'\.(rss|xml|atom)(\?|$)', href, re.I):
                out.append(ul.urljoin(url, href))
        seen = set(); uniq = []
        for x in out:
            nx = _normalize_url(x)
            if nx and nx not in seen:
                seen.add(nx); uniq.append(nx)
        return uniq[:10]
    except Exception:
        return []

def _title_key(t: str | None) -> str:
    if not t:
        return ""
    t = t.lower()
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'["“”«»\[\]\(\)\.\,\!\?\:\;–-]', ' ', t)
    return t.strip()

def _is_similar_title(t1: str, t2: str, threshold: int) -> bool:
    if not t1 or not t2:
        return False
    if t1 in t2 or t2 in t1:
        return True
    return partial_ratio(t1, t2) >= threshold

def _to_ts(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def _entry_to_item(feed_url, e):
    return {
        "title": getattr(e, "title", None),
        "url": _normalize_url(getattr(e, "link", None)),
        "published": _entry_iso(e),
        "published_ts": _entry_timestamp(e),
        "source": feed_url,
    }

def _download_article_text(url: str, lang: str = "ru") -> tuple[str, str | None, str | None]:
    # 1) newspaper3k
    try:
        art = Article(url, language=lang)
        art.download(); art.parse()
        text = (art.text or "").strip()
        if text and len(text) > 300:
            return url, text, None
    except Exception as e:
        n_err = str(e)
    else:
        n_err = "short_or_empty"
    # 2) readability
    try:
        import requests
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        doc = Document(r.text)
        html = doc.summary()
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text("\n", strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        if text and len(text) > 300:
            return url, text, None
        return url, None, f"readability_short ({len(text) if text else 0})"
    except Exception as e2:
        return url, None, f"newspaper:{n_err}; readability:{e2}"

async def _fetch_article_texts(urls: Sequence[str], lang: str = "ru", workers: int = 12) -> dict:
    loop = asyncio.get_running_loop()
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = [loop.run_in_executor(pool, _download_article_text, url, lang) for url in urls]
        for fut in asyncio.as_completed(tasks):
            url, text, err = await fut
            results[url] = {"text": text, "error": err}
    return results

async def _fetch_feed_or_discover(session: aiohttp.ClientSession, feed_url: str, limit: int) -> list:
    content, ctype, err = await _get(session, feed_url)
    if err or not content:
        return []
    if "xml" in (ctype or "") or b"<rss" in content[:2000] or b"<feed" in content[:2000]:
        d = feedparser.parse(content)
        return [_entry_to_item(feed_url, e) for e in d.entries[:limit]]
    discovered = _discover_rss_from_html(feed_url, content)
    items = []
    for rss in discovered:
        c2, ct2, err2 = await _get(session, rss)
        if err2 or not c2:
            continue
        d2 = feedparser.parse(c2)
        if d2.entries:
            items.extend([_entry_to_item(rss, e) for e in d2.entries[:limit]])
    return items

# ---------- public API for GUI ----------

async def fetch_news(
    feeds: Iterable[str],
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    *,
    per_feed_limit: int = 1000,
    total_limit: int = 8000,
    article_workers: int = 12,
    max_per_domain: int = 800,
    title_sim_threshold: int = 92,
    lang: str = "ru",
    user_agent: str = UA
) -> list[dict]:
    """
    Асинхронная функция для GUI.
    Вход: период времени (since/until, UTC-aware или naive как UTC) и список источников (RSS/страницы).
    Выход: список объектов новостей с полями: title, url, published, source, text, error.
    """
    since_ts = _to_ts(since)
    until_ts = _to_ts(until)

    headers = {"User-Agent": user_agent}
    async with aiohttp.ClientSession(headers=headers) as session:
        feed_tasks = [_fetch_feed_or_discover(session, u, per_feed_limit) for u in feeds]
        candidates = []
        for fut in asyncio.as_completed(feed_tasks):
            items = await fut
            candidates.extend(items)

    # фильтр по времени
    if since_ts is not None or until_ts is not None:
        filtered = []
        for it in candidates:
            ts = it.get("published_ts") or 0
            if since_ts is not None and ts < since_ts:
                continue
            if until_ts is not None and ts > until_ts:
                continue
            filtered.append(it)
        candidates = filtered

    # сортировка по времени
    candidates.sort(key=lambda x: x.get("published_ts", 0), reverse=True)

    # дедуп по URL
    seen_urls = set(); dedup_url = []
    for it in candidates:
        u = it.get("url")
        if u and u not in seen_urls:
            seen_urls.add(u); dedup_url.append(it)

    # дедуп по заголовкам
    picked = []; title_bank = []
    for it in dedup_url:
        tk = _title_key(it.get("title"))
        if not tk:
            picked.append(it); title_bank.append(tk); continue
        if any(_is_similar_title(tk, tb, title_sim_threshold) for tb in title_bank if tb):
            continue
        picked.append(it); title_bank.append(tk)

    # диверсификация по доменам и общий лимит
    per_domain = defaultdict(int); diverse = []
    for it in picked:
        u = it.get("url")
        if not u:
            continue
        d = _domain(u)
        if per_domain[d] >= max_per_domain:
            continue
        per_domain[d] += 1
        diverse.append(it)
        if len(diverse) >= total_limit:
            break

    # тексты
    urls = [it["url"] for it in diverse if it.get("url")]
    url2text = await _fetch_article_texts(urls, lang=lang, workers=article_workers)

    # мерж и очистка служебного поля
    out = []
    for it in diverse:
        it.pop("published_ts", None)
        u = it.get("url")
        text_info = url2text.get(u, {"text": None, "error": "no_result"})
        out.append({
            "title": it.get("title"),
            "url": u,
            "published": it.get("published"),
            "source": it.get("source"),
            "text": text_info["text"],
            "error": text_info.get("error")
        })
    return out

# ---------- удобный синхронный враппер для GUI-потока ----------

def fetch_news_sync(
    feeds: Iterable[str],
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    **kwargs
) -> list[dict]:
    """Синхронная обертка для сред, где неудобно работать с asyncio."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # Вызов из уже работающего event loop (например, в некоторых GUI-фреймворках)
        return asyncio.run_coroutine_threadsafe(
            fetch_news(feeds, since, until, **kwargs), loop
        ).result()
    return asyncio.run(fetch_news(feeds, since, until, **kwargs))
