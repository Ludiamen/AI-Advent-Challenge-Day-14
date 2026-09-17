"""Каталог моделей, между которыми можно переключать агента.

Все провайдеры OpenAI-совместимы, поэтому модель описывается тремя вещами:
адресом API, идентификатором модели и переменной окружения с ключом. Агент
берёт отсюда всё нужное, а интерфейсы показывают пользователю список.

Состав проверен живыми запросами: у Groq моделей Llama больше нет, у z.ai
бесплатно работает только glm-4.5-flash, платные модели без баланса отвечают
ошибкой. Здесь оставлено то, что действительно отвечает.

Размер контекстного окна взят из ответа эндпоинта /models самого провайдера, а
не из статей в интернете. Именно он ограничивает длину разговора: когда история
перестаёт помещаться, провайдер отвечает ошибкой, а не обрезает лишнее сам.
Модель allam-2-7b с окном 4096 токенов держится в каталоге как испытательный
стенд — на ней переполнение наступает за считанные реплики.

История диалога от модели не зависит: она хранится как обычные пары
«роль — текст», поэтому модель можно сменить посреди разговора, и агент
продолжит с тем же контекстом.

Публичное API:
  MODELS              — все модели (ключ -> описание)
  DEFAULT_MODEL       — ключ модели по умолчанию
  get(key)            — описание модели по ключу
  available()         — ключи моделей, для которых задан ключ API
  ROLES               — какая модель на какой точке вызова агента
  for_role(role)      — ключ модели для роли, с чередованием провайдеров
  escalate(key)       — следующая ступень лестницы эскалации
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Model:
    """Одна модель: где живёт, как называется и чем интересна."""

    key: str
    api_id: str
    provider: str
    base_url: str
    env_var: str
    label: str
    free: bool
    context_window: int          # сколько токенов помещается в запрос вместе с ответом
    price_in: float = 0.0        # долларов за миллион входных токенов
    price_out: float = 0.0       # долларов за миллион выходных
    # Токенизатор у каждой модели свой, и один и тот же русский текст обходится
    # им по-разному. Коэффициент — во сколько раз фактический расход отличается
    # от базовой оценки agent/tokens.py; измерен одним текстом на всех моделях.
    token_factor: float = 1.0
    # Провайдер добавляет к запросу собственную обёртку (служебный системный
    # блок, разделители ролей). У gpt-oss на Groq это целых 72 токена.
    request_overhead: int = 8
    # Модели семейства gpt-oss принимают reasoning_effort: на механических
    # задачах вроде сжатия истории «low» убирает почти всё скрытое рассуждение.
    # Без него сжиматель тратит сотни токенов на размышления и порой возвращает
    # пустой ответ, упершись в предел длины.
    supports_effort: bool = False
    note: str = ""

    @property
    def api_key(self) -> str:
        return os.getenv(self.env_var, "").strip()

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    @property
    def price_text(self) -> str:
        if self.free:
            return "бесплатно"
        return f"${self.price_in:g} / ${self.price_out:g} за 1M"


_GROQ = "https://api.groq.com/openai/v1"
_ZAI = "https://api.z.ai/api/paas/v4"
_DEEPSEEK = "https://api.deepseek.com"

MODELS: dict[str, Model] = {
    "groq-120b": Model(
        key="groq-120b", api_id="openai/gpt-oss-120b", provider="Groq",
        base_url=_GROQ, env_var="GROQ_API_KEY", label="GPT-OSS 120B",
        free=True, context_window=131_072,
        token_factor=0.83, request_overhead=72,
        supports_effort=True,
        note="Самая сильная из бесплатных, отвечает за 3-4 секунды. По умолчанию.",
    ),
    "groq-20b": Model(
        key="groq-20b", api_id="openai/gpt-oss-20b", provider="Groq",
        base_url=_GROQ, env_var="GROQ_API_KEY", label="GPT-OSS 20B",
        free=True, context_window=131_072,
        token_factor=0.83, request_overhead=72,
        supports_effort=True,
        note="Вдвое быстрее старшей и тоже бесплатна; на простых вопросах разницы почти нет.",
    ),
    "groq-qwen27b": Model(
        key="groq-qwen27b", api_id="qwen/qwen3.8-27b", provider="Groq",
        base_url=_GROQ, env_var="GROQ_API_KEY", label="Qwen 3.8 27B",
        free=True, context_window=131_042,
        token_factor=0.74, request_overhead=11,
        note="Другое семейство моделей. Провайдер меняет версии молча — см. «--проверить».",
    ),
    "groq-allam7b": Model(
        key="groq-allam7b", api_id="allam-2-7b", provider="Groq",
        base_url=_GROQ, env_var="GROQ_API_KEY", label="Allam 2 7B",
        free=True, context_window=4_096,
        token_factor=2.54, request_overhead=8,
        note=(
            "Контекст всего 4096 токенов — на ней переполнение видно за пару реплик. "
            "Для содержательных ответов слаба, зато незаменима как испытательный стенд."
        ),
    ),
    "glm-flash": Model(
        key="glm-flash", api_id="glm-4.5-flash", provider="z.ai",
        base_url=_ZAI, env_var="ZAI_API_KEY", label="GLM 4.5 Flash",
        free=True, context_window=131_072,
        token_factor=0.89, request_overhead=6,
        note="Бесплатная модель z.ai. Отвечает заметно медленнее остальных.",
    ),
    "ds-flash": Model(
        key="ds-flash", api_id="deepseek-v4-flash", provider="DeepSeek",
        base_url=_DEEPSEEK, env_var="DEEPSEEK_API_KEY", label="DeepSeek V4 Flash",
        free=False, context_window=131_072, price_in=0.44, price_out=1.32,
        token_factor=0.92, request_overhead=5,
        note="Платная. Хороша на длинных документах и юридических формулировках.",
    ),
    "ds-pro": Model(
        key="ds-pro", api_id="deepseek-v4-pro", provider="DeepSeek",
        base_url=_DEEPSEEK, env_var="DEEPSEEK_API_KEY", label="DeepSeek V4 Pro",
        free=False, context_window=131_072, price_in=1.32, price_out=3.96,
        token_factor=0.92, request_overhead=5,
        note="Платная и самая медленная, зато самая внимательная к деталям.",
    ),
}

DEFAULT_MODEL = os.getenv("LLM_MODEL_KEY", "groq-120b")


def get(key: str) -> Model:
    """Описание модели по ключу; понятная ошибка, если ключ неизвестен."""
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(
            f"Неизвестная модель «{key}». Доступны: {', '.join(MODELS)}."
        ) from None


def available() -> list[str]:
    """Модели, для которых в окружении есть ключ соответствующего провайдера."""
    return [key for key, model in MODELS.items() if model.has_key]


def describe() -> list[dict]:
    """Список моделей для интерфейсов: без ключей, только то, что можно показать."""
    return [
        {
            "key": model.key,
            "label": model.label,
            "provider": model.provider,
            "api_id": model.api_id,
            "free": model.free,
            "context_window": model.context_window,
            "price_in": model.price_in,
            "price_out": model.price_out,
            "price_text": model.price_text,
            "token_factor": model.token_factor,
            "note": model.note,
            "available": model.has_key,
        }
        for model in MODELS.values()
    ]


# --- роли: какая модель на какой точке вызова ---------------------------------
# Не каждой точке нужна сильная модель, и не каждой — вообще модель. Здесь
# записано распределение по фактическим точкам вызова агента.
#
# Принцип: бесплатное по умолчанию, платное — по факту провала, а не «на всякий
# случай». Единственная сильная бесплатная модель — groq-120b, поэтому она и
# стоит дефолтом там, где цена ошибки высока (планирование, исполнение).
#
# Для частых мелких вызовов список содержит две модели РАЗНЫХ провайдеров:
# groq-allam7b и glm-flash сидят на независимых лимитах, и агент чередует их
# по кругу. Так десяток классификаций подряд не упирается в минутный лимит Groq.

ROLES: dict[str, dict[str, Any]] = {
    "маршрутизация": {
        # Порядок не алфавитный, а по замеру: на пробе «отнеси реплику к слою»
        # (см. RESULTS.md) allam-2-7b ответил «Профиль» вместо «знания», а
        # glm-4.5-flash ответил верно. Обе модели остаются в списке — вторая
        # нужна ради независимого лимита, — но первой берётся та, что не
        # ошиблась.
        "models": ["glm-flash", "groq-allam7b"],
        "note": "Куда писать реплику — в профиль, в знания или только в короткую память.",
        "why": "Вызов на каждое сообщение: важнее скорость и лимиты, чем глубина.",
    },
    "сжатие": {
        "models": ["groq-20b"],
        "note": "Свернуть рабочую память завершённой задачи в запись журнала решений.",
        "why": "Раз на задачу, но терять детали нельзя — слабая модель не годится.",
    },
    "планирование": {
        "models": ["groq-120b"],
        "note": "Разложить задачу на шаги на стадии planning.",
        "why": "Весь дальнейший ход задачи строится на этом ответе.",
    },
    "исполнение": {
        "models": ["groq-120b"],
        "note": "Основной ответ агента: код Django/GeoDjango, разбор legacy-схемы.",
        "why": "Сильнейшая из бесплатных; платная подключается только по эскалации.",
    },
    "фронтенд": {
        "models": ["groq-qwen27b"],
        "note": "Обвязка карты: OpenLayers, слои, инструменты редактирования.",
        "why": "Код здесь шаблоннее серверного, и другое семейство моделей справляется.",
    },
    "судья-инвариантов": {
        # Не самая дешёвая намеренно. На живом прогоне Дня 14 allam-2-7b
        # возвращал неразбираемый вердикт, а glm-4.5-flash — пустой ответ, и
        # смысловые инварианты оставались без проверки вовсе. Выглядело это как
        # «нарушений нет», что хуже отсутствия проверки: создаёт ложное
        # спокойствие.
        "models": ["groq-20b"],
        "note": "Судит, нарушает ли ответ архитектурное решение или бизнес-правило.",
        "why": "Вердикт должен быть разбираемым; на слабых моделях он им не был.",
    },
    "мягкая-проверка": {
        "models": ["groq-allam7b", "glm-flash"],
        "note": "Стиль и идиомы ответа — то, что нельзя проверить регуляркой.",
        "why": "Узкая задача после каждого ответа, снова вопрос лимитов.",
    },
}

# Жёсткие инварианты (стек, БД, секреты в URL) в этой таблице отсутствуют
# намеренно: их проверяет код в agent/validator.py. Модель, которую просят
# «проверить, нет ли тут Laravel», ошибается и поддаётся уговорам; grep — нет.

# Лестница эскалации: при провале валидации агент поднимается на одну ступень,
# а не прыгает сразу на самую дорогую модель.
ESCALATION = ["groq-allam7b", "groq-20b", "groq-120b", "ds-pro"]

_счётчики: dict[str, int] = {}


def for_role(role: str, offset: int = -1) -> str:
    """Ключ модели для роли; список ролей перебирается по кругу.

    offset=-1 (по умолчанию) — взять следующую по внутреннему счётчику, то есть
    чередовать провайдеров. Явный offset делает выбор воспроизводимым: этим
    пользуются тесты и сравнение, где два прогона должны идти на одной модели.
    """
    описание = ROLES.get(role)
    if описание is None:
        raise KeyError(f"Неизвестная роль «{role}». Доступны: {', '.join(ROLES)}.")
    варианты = [к for к in описание["models"] if MODELS[к].has_key] or описание["models"]
    if offset < 0:
        offset = _счётчики.get(role, 0)
        _счётчики[role] = offset + 1
    return варианты[offset % len(варианты)]


def escalate(model_key: str) -> str:
    """Следующая ступень эскалации; на вершине лестницы возвращает её же."""
    if model_key not in ESCALATION:
        # Модель вне лестницы (например, groq-qwen27b) — поднимаем на сильную
        # бесплатную, оттуда лестница пойдёт обычным порядком.
        return "groq-120b"
    шаг = ESCALATION.index(model_key) + 1
    return ESCALATION[min(шаг, len(ESCALATION) - 1)]


def describe_roles() -> list[dict]:
    """Таблица ролей для интерфейсов и справки CLI."""
    return [
        {
            "role": роль,
            "models": описание["models"],
            "chosen": for_role(роль, offset=0),
            "note": описание["note"],
            "why": описание["why"],
        }
        for роль, описание in ROLES.items()
    ]


def verify(keys: list[str] | None = None) -> list[dict[str, Any]]:
    """Проверяет, отвечает ли сегодня каждая модель каталога.

    Нужно потому, что провайдеры снимают модели и переименовывают версии молча.
    Каталог, перенесённый из прошлого дня, однажды указывает на модель, которой
    больше нет, и узнаётся это посреди сценария, на четвёртом шаге из пяти,
    после того как три первых уже потратили токены.

    Проверка идёт настоящим запросом на один токен, а не сверкой со списком
    /models. Первая версия сверялась со списком и объявила несуществующими
    glm-4.5-flash и deepseek-v4-flash — обе прекрасно отвечают, просто в
    перечне провайдера их нет. Проверка, которая кричит «волки» на рабочей
    модели, хуже, чем её отсутствие.
    """
    import httpx

    итог: list[dict[str, Any]] = []
    for ключ in keys or list(MODELS):
        модель = MODELS[ключ]
        строка: dict[str, Any] = {
            "key": ключ, "api_id": модель.api_id, "provider": модель.provider,
        }
        if not модель.has_key:
            строка["status"] = "нет ключа"
            итог.append(строка)
            continue

        тело = {
            "model": модель.api_id,
            "messages": [{"role": "user", "content": "ок"}],
            "max_tokens": 1,
        }
        try:
            ответ = httpx.post(
                f"{модель.base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {модель.api_key}",
                         "Content-Type": "application/json"},
                json=тело, timeout=30,
            )
        except httpx.HTTPError as exc:
            строка["status"] = "сеть недоступна"
            строка["detail"] = str(exc)[:80]
            итог.append(строка)
            continue

        if ответ.is_success:
            строка["status"] = "отвечает"
        elif ответ.status_code == 404:
            строка["status"] = "МОДЕЛИ НЕТ"
            строка["detail"] = _сообщение(ответ)
        elif ответ.status_code in (401, 403):
            строка["status"] = "ключ отклонён"
            строка["detail"] = _сообщение(ответ)
        elif ответ.status_code in (429, 413):
            # Лимит расхода означает, что модель есть: провайдер её узнал.
            строка["status"] = "отвечает (упёрлись в лимит)"
        else:
            строка["status"] = f"ошибка {ответ.status_code}"
            строка["detail"] = _сообщение(ответ)
        итог.append(строка)
    return итог


def _сообщение(ответ: Any) -> str:
    try:
        return str((ответ.json().get("error") or {}).get("message", ""))[:100]
    except ValueError:
        return ответ.text[:100]
