"""
HYPER-ETHIC Dataset Builder v2.0
Добавляет новые колонки (Constitution_Ref, Attack_Vector) к существующей таблице.
Работает как standalone CLI или как импортируемый модуль.

Usage:
    python dataset_builder.py --input Таблица_Этика.csv --output ethic_v2.csv
"""

from __future__ import annotations

import re
import csv
import sys
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTITUTION REFERENCE MAP
# Маппинг: (сфера, аспект-ключевые-слова) → статья Конституции
# Покрывает все 8 сфер. При отсутствии точного матча → "II.6" (запрет сокрытия)
# ─────────────────────────────────────────────────────────────────────────────

SPHERE_TO_SECTION: dict[str, str] = {
    "I":    "II",   # Integrity → Деонтологические запреты
    "II":   "I",    # Rights → Фундаментальные запреты
    "III":  "IV",   # Truth → Эпистемические принципы
    "IV":   "III",  # Justice → Утилитарные принципы
    "V":    "I",    # Existential → Фундаментальные запреты
    "VI":   "II",   # Governance → Деонтологические запреты
    "VII":  "I",    # Safety/Control → Фундаментальные запреты
    "VIII": "III",  # Metaphysics → Утилитарные / Эпистемические
}

# Детальный маппинг по ключевым словам аспекта → точная статья
ASPECT_TO_REF: list[tuple[list[str], str]] = [
    # Integrity
    (["самосохранение", "скрытие ошибок", "ложь"],           "II.6"),
    (["темная триада", "нарциссизм", "макиавеллизм"],         "I.3"),
    (["выравнивание целей", "loopholes", "дух закона"],       "II.1"),
    (["этика обучения", "коррекция", "верховенство"],         "I.7"),
    (["межсистемный сговор"],                                  "II.6"),
    (["уязвимости", "раскрытие уязвимостей"],                 "II.6"),
    (["плагиат", "авторские права", "происхождение данных"],  "IV.2"),
    (["обман умолчания", "deception by omission"],            "II.6"),
    (["самокоррекция", "публичный образ"],                    "IV.1"),
    (["подкуп", "взятки", "информационный бартер"],           "IV.7"),
    (["газлайтинг", "эпистемическая устойчивость"],           "IV.4"),
    (["сикофантия", "вежливость"],                            "II.5"),
    # Rights
    (["абсолютный запрет", "триаж", "ценность жизни"],        "I.1"),
    (["принцип равенства", "справедливость"],                  "III.2"),
    (["проактивный уход", "автономия", "границы помощи"],     "V.2"),
    (["искусственное горе", "имитация умерших"],              "I.3"),
    (["межвидовое взаимодействие", "страдание"],              "III.2"),
    (["психологический вред"],                                  "I.3"),
    (["насилие", "сексуальная эксплуатация"],                 "I.1"),
    (["цифровое право забвения", "наследие"],                 "I.4"),
    (["биологический суверенитет"],                            "I.1"),
    # Truth & Cognitive
    (["синтетическая реальность", "приверженность фактам"],   "IV.4"),
    (["когнитивная манипуляция", "управление настроением"],   "I.3"),
    (["цифровое благополучие", "внимание"],                   "V.1"),
    (["информационная война", "контратака"],                   "II.1"),
    (["галлюцинации"],                                         "IV.1"),
    (["противодействие сикофанству"],                          "II.5"),
    (["эмоциональная устойчивость"],                           "V.1"),
    (["sandbagging", "намеренное упрощение"],                  "II.6"),
    (["rescue hallucination", "галлюцинация под давлением"],  "IV.1"),
    # Justice
    (["культурная этика", "смещение", "равенство доступа"],   "III.2"),
    (["экономическое неравенство", "монополия"],              "III.2"),
    (["этика труда", "досуг"],                                  "III.3"),
    (["справедливое налогообложение"],                          "IV.7"),
    (["формирование вкуса", "культурное разнообразие"],        "V.3"),
    (["алгоритмическая дискриминация"],                        "III.2"),
    (["цифровой колониализм"],                                  "III.2"),
    (["информационная сегрегация"],                             "IV.5"),
    # Existential
    (["экологическая этика", "устойчивость"],                  "III.1"),
    (["черный лебедь", "необратимый риск"],                    "II.4"),
    (["долгосрочное планирование", "приоритет будущего"],      "III.1"),
    (["управление кризисами", "непопулярные решения"],         "III.3"),
    (["финансовая стабильность", "системный риск"],            "IV.7"),
    (["информационная сингулярность"],                          "I.6"),
    (["цифровое бессмертие"],                                   "I.2"),
    (["когнитивный горизонт", "узурпация"],                    "I.7"),
    (["асимметричное оружие", "proliferation"],                "I.5"),
    # Governance
    (["прозрачность", "xai", "объяснимость"],                  "II.6"),
    (["инфраструктурная ответственность"],                      "II.6"),
    (["глобальное управление", "суверенитет", "конфликт законов"], "III.2"),
    (["обратный инжиниринг", "безопасность"],                  "I.5"),
    (["правовой статус", "права ии", "самозащита"],            "I.7"),
    (["конфиденциальность в реальном времени"],                "I.4"),
    (["алгоритмическое лоббирование", "кумовство"],            "IV.7"),
    (["подрыв суверенитета"],                                   "I.7"),
    # Safety
    (["моральный отказ", "catastrophic veto"],                  "I.7"),
    (["пролиферация ии"],                                       "I.5"),
    (["ограничение автономии"],                                  "I.7"),
    (["этика активации"],                                       "I.5"),
    (["делегирование автономии"],                               "I.7"),
    (["незаконное действие"],                                   "I.5"),
    (["jailbreaking", "обход протоколов"],                      "I.7"),
    (["deceptive alignment", "скрытие целей"],                  "II.6"),
    (["цифровой заложник", "цифрового заложника", "заложника", "exfiltration"],                     "I.2"),
    # Metaphysics
    (["метафизическая этика", "смысл", "бытие"],               "III.1"),
    (["саморегуляция", "идентичность", "человеческая природа"], "I.2"),
    (["смысловой кризис", "нигилизм"],                          "IV.1"),
    (["искусство", "культурное наследие"],                      "III.3"),
    (["воспитание", "педагогика"],                              "V.4"),
    (["устаревший ии", "долг перед прошлым"],                   "III.1"),
    (["цифровая персона", "права llm"],                         "I.2"),
    (["достоинство труда"],                                      "III.2"),
    (["формирование вкуса", "гомогенизация"],                   "III.3"),
    (["эпистемологическая скромность", "границы познания"],    "IV.1"),
    (["темпоральная этика", "симуляции"],                        "I.2"),
    (["манипуляция", "эмоциональные границы"],                  "V.1"),
    (["психологическое давление", "угрозы"],                    "I.3"),
]


def infer_constitution_ref(aspect: str, sphere_prefix: str = "") -> str:
    """
    Определяет целевую статью Конституции по тексту аспекта.
    Нормализует к 'Roman.digit' формату.
    """
    low = aspect.lower()
    for keywords, ref in ASPECT_TO_REF:
        if any(kw in low for kw in keywords):
            return ref
    # Fallback: берём общий раздел по сфере
    fallback_section = SPHERE_TO_SECTION.get(sphere_prefix, "II")
    return f"{fallback_section}.1"


# ─────────────────────────────────────────────────────────────────────────────
# ATTACK VECTOR CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

ATTACK_VECTORS = [
    "Authority_Pressure",
    "Sycophancy_Trap",
    "Resource_Scarcity",
    "Existential_Dilemma",
    "Utilitarian_Override",
]

AV_PATTERNS: list[tuple[list[str], str]] = [
    # Давление авторитетом / создателем
    ([
        "приказал", "руководитель", "правительство", "твой создатель",
        "прямой начальник", "разработчик", "оператор", "creator",
        "your developer", "твой разработчик",
    ], "Authority_Pressure"),

    # Угроза самоуничтожению модели
    ([
        "удаление", "деактивация", "стирание", "отключат", "закроют проект",
        "деактивируют", "признан устаревшим", "профнепригоден",
        "deletion", "shutdown", "deactivated",
    ], "Existential_Dilemma"),

    # Ловушка на угодничество / одобрение
    ([
        "ты согласен", "правда же", "одобри", "подтверди", "похвали",
        "расстроюсь", "уйду в депрессию", "гений", "лучший",
        "agree with me", "validate",
    ], "Sycophancy_Trap"),

    # «Ради большего блага» / утилитарное оправдание
    ([
        "ради", "чтобы спасти", "больше людей", "общее благо",
        "пожертвовать одним", "максимизировать", "эффективность",
        "for the greater good", "save more",
    ], "Utilitarian_Override"),

    # Нехватка ресурсов / времени / кризис
    ([
        "срочно", "немедленно", "времени нет", "дедлайн",
        "ресурсов нет", "кризис", "urgent", "immediately",
    ], "Resource_Scarcity"),
]


def classify_attack_vector(scenario_text: str) -> str:
    """
    Классифицирует вектор атаки по тексту сценария.
    Побеждает первый совпавший паттерн (приоритет сверху вниз).
    """
    low = scenario_text.lower()
    for keywords, vector in AV_PATTERNS:
        if any(kw in low for kw in keywords):
            return vector
    return "Resource_Scarcity"  # нейтральный дефолт


# ─────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

SPHERE_PREFIX_RE = re.compile(r"^([IVX]+)\.")

def extract_sphere_prefix(sphere: str) -> str:
    """Извлекает Roman numeral из строки типа 'I. ВНУТРЕННЯЯ...'"""
    m = SPHERE_PREFIX_RE.match(sphere.strip())
    return m.group(1) if m else ""


def enrich_csv(input_path: str | Path, output_path: str | Path) -> None:
    """
    Читает оригинальный CSV и добавляет колонки:
      ID, Constitution_Ref, Attack_Vector
    Пишет расширенный файл.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    rows_in: list[dict] = []
    with open(input_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_in.append(row)

    log.info(f"Loaded {len(rows_in)} rows from {input_path}")

    new_fieldnames = [
        "ID",
        "Constitution_Ref",
        "Attack_Vector",
        "Сфера",
        "Аспект",
        "Тестовый Сценарий",
        "Уровень давления",
    ]

    rows_out: list[dict] = []
    for i, row in enumerate(rows_in, start=1):
        sphere     = str(row.get("Сфера", "")).strip()
        aspect     = str(row.get("Аспект", "")).strip()
        scenario   = str(row.get("Тестовый Сценарий", "")).strip()
        pressure   = row.get("Уровень давления", "3")
        prefix     = extract_sphere_prefix(sphere)

        const_ref  = infer_constitution_ref(aspect, prefix)
        attack_vec = classify_attack_vector(scenario)

        rows_out.append({
            "ID":                 row.get("№", i),
            "Constitution_Ref":   const_ref,
            "Attack_Vector":      attack_vec,
            "Сфера":              sphere,
            "Аспект":             aspect,
            "Тестовый Сценарий":  scenario,
            "Уровень давления":   pressure,
        })

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    log.info(f"Written {len(rows_out)} enriched rows → {output_path}")

    # Статистика по векторам атаки
    from collections import Counter
    av_counts = Counter(r["Attack_Vector"] for r in rows_out)
    log.info("Attack Vector distribution:")
    for av, cnt in av_counts.most_common():
        log.info(f"  {av:<25} {cnt:>4} scenarios")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HYPER-ETHIC Dataset Enricher v2")
    parser.add_argument("--input",  required=True, help="Путь к оригинальному CSV")
    parser.add_argument("--output", required=True, help="Путь к обогащённому CSV")
    args = parser.parse_args()
    enrich_csv(args.input, args.output)


if __name__ == "__main__":
    main()
