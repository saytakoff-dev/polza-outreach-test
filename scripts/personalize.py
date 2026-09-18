#!/usr/bin/env python3
"""
personalize.py — генератор персонализации для холодных писем (Polza Agency, тестовое задание).

Что делает:
  1. Читает список компаний из CSV (нужны колонки "Компания" и "Сайт").
  2. Скачивает главную страницу (и, если найдёт, страницу "О компании"/"Контакты") сайта компании.
  3. Вырезает из HTML только видимый текст (без скриптов/стилей/меню) — это и есть "источник правды".
  4. Отправляет этот текст в LLM (Claude) с жёстким промптом: сформулировать 1-2 предложения
     персонализации ТОЛЬКО на основе присланного текста, ничего не выдумывая. Если фактов мало —
     модель обязана вернуть "не найдено", а не придумать что-то правдоподобное.
  5. Записывает результат в колонку "Персонализация" выходного CSV.

Почему так, а не «просто спросить LLM про компанию»:
  Задание прямо требует "не выдумывать" и брать факты с сайта/новостей. Поэтому скрипт сначала
  реально скачивает страницу и передаёт модели только то, что на ней есть, — а не полагается
  на "знания" модели о компании, которых может не быть или они могут быть устаревшими.

Использование:
  export ANTHROPIC_API_KEY=sk-...
  python personalize.py --input ../data/companies.csv --output ../data/companies_personalized.csv

  # без ключа — просто скачает и покажет вытащенный текст, ничего не генерируя (для отладки):
  python personalize.py --input ../data/companies.csv --output out.csv --dry-run

Ограничения (честно, как и просит задание в пункте 5):
  - Сайты на тяжёлом JS (SPA без SSR) могут отдать requests'у пустую страницу — здесь это
    не решается (нужен headless-браузер вроде Playwright, сознательно не стали усложнять скрипт
    ради 2-дневного тестового).
  - Скрипт не обходит капчу и не логинится никуда.
  - Rate-limit сайтов: между запросами есть задержка (--delay), но при большом списке компаний
    некоторые сайты всё равно могут временно заблокировать частые запросы.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 PolzaOutreachBot/1.0"
)

CONTACT_PATH_HINTS = ("contact", "contacts", "kontakt", "kontakty", "about", "o-kompanii", "o-nas")

PROMPT_TEMPLATE = """Ты помогаешь составить ПЕРСОНАЛИЗАЦИЮ для холодного письма B2B-компании.

Ниже — текст, реально вытащенный с сайта компании "{company}" ({url}). Это ЕДИНСТВЕННЫЙ источник
фактов, которым тебе разрешено пользоваться. Ничего не выдумывай и не досочиняй.

Задача: напиши 1-2 предложения персонализации — конкретный факт о компании или её представителе,
который можно вставить в начало холодного письма, чтобы показать, что письмо не шаблонное.
Хорошо: конкретные цифры, ниша, имя и должность руководителя, необычный факт, экспертиза.
Плохо: общие фразы вроде "ваша компания активно развивается" — так писать нельзя.

Если в тексте ниже недостаточно конкретных фактов для персонализации — верни ровно строку:
не найдено

Текст с сайта:
---
{content}
---

Ответ (1-2 предложения, на русском, без вступлений вроде "Конечно!"):"""


@dataclass
class Company:
    name: str
    site: str
    extra: dict = field(default_factory=dict)


def fetch_text(url: str, timeout: int = 12) -> str:
    """Скачивает страницу и возвращает только видимый текст."""
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return f"[ошибка загрузки {url}: {exc}]"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)[:6000]  # ограничиваем, чтобы не раздувать промпт


def find_contact_link(base_url: str, timeout: int = 12) -> str | None:
    """Пытается найти ссылку на страницу 'О компании'/'Контакты' на главной странице."""
    try:
        resp = requests.get(base_url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if any(hint in href for hint in CONTACT_PATH_HINTS):
            return urllib.parse.urljoin(base_url, a["href"])
    return None


def gather_source_text(site_url: str, delay: float) -> str:
    home_text = fetch_text(site_url)
    time.sleep(delay)

    contact_url = find_contact_link(site_url)
    if contact_url and contact_url.rstrip("/") != site_url.rstrip("/"):
        contact_text = fetch_text(contact_url)
        time.sleep(delay)
        return home_text + "\n\n--- страница контактов/о компании ---\n\n" + contact_text
    return home_text


def call_llm(prompt: str) -> str:
    """Вызывает Anthropic Claude API. Требует ANTHROPIC_API_KEY в окружении."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Не найден ANTHROPIC_API_KEY. Установите переменную окружения, "
            "либо запустите скрипт с --dry-run, чтобы только проверить сбор текста с сайтов."
        )

    import anthropic  # локальный импорт, чтобы --dry-run работал без установленного пакета

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def clean_snippet(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="CSV со списком компаний (колонки: Компания, Сайт, ...)")
    parser.add_argument("--output", required=True, help="Куда записать CSV с колонкой 'Персонализация'")
    parser.add_argument("--delay", type=float, default=1.0, help="Пауза между запросами к сайтам, сек (по умолчанию 1.0)")
    parser.add_argument("--dry-run", action="store_true", help="Не звать LLM — только скачать и показать текст с сайтов")
    parser.add_argument("--limit", type=int, default=None, help="Обработать только первые N строк (для теста)")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "Персонализация" not in fieldnames:
        fieldnames = fieldnames + ["Персонализация"]

    if args.limit:
        rows = rows[: args.limit]

    for i, row in enumerate(rows, start=1):
        company = row.get("Компания", "").strip()
        site = row.get("Сайт", "").strip()
        if not site:
            row["Персонализация"] = "не найдено (нет сайта в базе)"
            continue

        print(f"[{i}/{len(rows)}] {company} — {site}", file=sys.stderr)
        source_text = gather_source_text(site, args.delay)

        if args.dry_run:
            preview = clean_snippet(source_text)[:200]
            row["Персонализация"] = f"[DRY-RUN, текста {len(source_text)} симв.] {preview}"
            continue

        prompt = PROMPT_TEMPLATE.format(company=company, url=site, content=source_text)
        try:
            result = call_llm(prompt)
        except RuntimeError as exc:
            print(f"  ! {exc}", file=sys.stderr)
            row["Персонализация"] = "не найдено (ошибка вызова LLM)"
            continue

        row["Персонализация"] = clean_snippet(result)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Готово: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
