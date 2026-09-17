"""Подсчёт токенов и денег: оценка до запроса и факт после.

Токен — единица, которой измеряется и объём контекста, и счёт от провайдера.
Знать их количество нужно в двух разных моментах:

  ДО запроса — чтобы понять, влезет ли разговор в контекст модели, и решить,
  что делать, если не влезает. Точного числа тут не получить: у каждой модели
  свой токенизатор, и запускать его локально — это отдельная зависимость,
  которая тянет за собой словари на десятки мегабайт. Поэтому здесь оценка
  по классам символов, откалиброванная на реальных ответах API.

  ПОСЛЕ запроса — точное число приходит в поле `usage` ответа. Оно и идёт в
  историю, в счётчики и в расчёт стоимости.

Оценка намеренно смещена в сторону завышения: лучше зря обрезать лишнюю реплику,
чем упереться в лимит модели и получить отказ.

Публичное API:
  estimate(text)                  -> int      — оценка числа токенов в тексте
  estimate_messages(messages)     -> int      — оценка для списка сообщений
  cost(prompt, completion, model) -> float    — стоимость в долларах
  Budget                                      — разбор бюджета контекста
  format_tokens(n)                -> str      — «1 234» вместо «1234»
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Сколько токенов приходится на символ каждого класса. Значения получены
# сравнением с полем usage реальных ответов API: кириллица кодируется
# экономнее, чем кажется, а цифры и знаки — заметно дороже букв. Пробелы
# отдельного токена почти никогда не занимают: BPE приклеивает их к
# следующему слову.
TOKENS_PER_CHAR = {
    "кириллица": 0.26,
    "латиница": 0.18,
    "цифры": 0.70,
    "пробелы": 0.02,
    "прочее": 1.05,
}

# Точность оценки, замеренная на реальных ответах API (см. README, раздел
# «Насколько точна оценка»): средняя ошибка +1,8 %, наибольшее отклонение 21 %
# на самом коротком тексте, где эта доля — всего три токена. На текстах длиннее
# сотни токенов ошибка укладывается в ±8 %.
ESTIMATE_MEAN_ERROR = 0.018
ESTIMATE_MAX_ERROR = 0.214

# Обёртка каждого сообщения (роль, разделители) и запроса в целом.
MESSAGE_OVERHEAD = 4
REQUEST_OVERHEAD = 8

_КИРИЛЛИЦА = re.compile(r"[а-яёА-ЯЁ]")
_ЛАТИНИЦА = re.compile(r"[a-zA-Z]")
_ЦИФРЫ = re.compile(r"\d")
_ПРОБЕЛЫ = re.compile(r"\s")


def classify(text: str) -> dict[str, int]:
    """Считает символы текста по классам — на этом строится вся оценка."""
    кириллица = len(_КИРИЛЛИЦА.findall(text))
    латиница = len(_ЛАТИНИЦА.findall(text))
    цифры = len(_ЦИФРЫ.findall(text))
    пробелы = len(_ПРОБЕЛЫ.findall(text))
    return {
        "кириллица": кириллица,
        "латиница": латиница,
        "цифры": цифры,
        "пробелы": пробелы,
        "прочее": len(text) - кириллица - латиница - цифры - пробелы,
    }


def estimate(text: str) -> int:
    """Оценивает число токенов в тексте. Точное значение вернёт только API."""
    if not text:
        return 0
    счёт = classify(text)
    return max(1, round(sum(TOKENS_PER_CHAR[к] * n for к, n in счёт.items())))


def estimate_messages(messages: list[dict[str, str]]) -> int:
    """Оценивает объём списка сообщений вместе с накладными расходами обёртки."""
    if not messages:
        return 0
    return (
        sum(estimate(m.get("content", "")) + MESSAGE_OVERHEAD for m in messages)
        + REQUEST_OVERHEAD
    )


def cost(prompt_tokens: int, completion_tokens: int, price_in: float, price_out: float) -> float:
    """Стоимость запроса в долларах по цене за миллион токенов."""
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def format_tokens(count: int) -> str:
    """Разделяет разряды: 12345 -> «12 345»."""
    return f"{count:,}".replace(",", " ")


def format_cost(amount: float) -> str:
    """Деньги в удобочитаемом виде: очень мелкие суммы не превращаются в 0.00."""
    if amount <= 0:
        return "бесплатно"
    if amount < 0.01:
        return f"{amount:.6f} $"
    return f"{amount:.4f} $"


@dataclass
class Budget:
    """Разбор того, как разговор укладывается в контекст модели.

    Контекст модели делится на две части: то, что мы отправляем (промпт), и то,
    что модель напишет в ответ (max_tokens). Место под ответ нужно резервировать
    заранее, иначе длинный разговор оставит модели ноль места, и провайдер
    ответит ошибкой ещё до того, как начнёт генерировать.
    """

    limit: int              # контекстное окно модели, токенов
    reserved: int           # зарезервировано под ответ (max_tokens)
    system: int             # системный промпт
    history: int            # история диалога
    question: int           # текущий вопрос пользователя
    overhead: int = 0       # служебная обёртка провайдера

    @property
    def prompt(self) -> int:
        """Сколько токенов уйдёт в запросе."""
        return self.system + self.history + self.question + self.overhead

    @property
    def available(self) -> int:
        """Сколько токенов остаётся под промпт после резерва на ответ."""
        return max(0, self.limit - self.reserved)

    @property
    def free(self) -> int:
        """Запас: сколько ещё можно добавить, не выйдя за лимит."""
        return self.available - self.prompt

    @property
    def overflow(self) -> bool:
        """Разговор уже не помещается в контекст модели."""
        return self.free < 0

    @property
    def usage_share(self) -> float:
        """Доля занятого контекста, 0..1 и больше при переполнении."""
        return self.prompt / self.available if self.available else 1.0

    def describe(self) -> str:
        """Строка для интерфейсов: сколько занято и сколько осталось."""
        итог = (
            f"контекст {format_tokens(self.prompt)} из {format_tokens(self.available)} "
            f"токенов ({self.usage_share:.0%})"
        )
        if self.overflow:
            return итог + f", превышение на {format_tokens(-self.free)}"
        return итог + f", запас {format_tokens(self.free)}"

    def to_dict(self) -> dict:
        return {
            "limit": self.limit,
            "reserved": self.reserved,
            "system": self.system,
            "overhead": self.overhead,
            "history": self.history,
            "question": self.question,
            "prompt": self.prompt,
            "available": self.available,
            "free": self.free,
            "overflow": self.overflow,
            "usage_share": round(self.usage_share, 4),
        }
