"""Единственное место, где агент ходит в сеть.

В этот день у агента много точек вызова LLM и они идут к разным моделям:
маршрутизатор реплик — к дешёвой, сжатие завершённой задачи — к средней,
основной ответ — к сильной. Держать HTTP-запрос внутри самого агента стало
неудобно, поэтому он вынесен сюда: любая часть системы просит «позови такую-то
модель с такими-то сообщениями» и получает текст плюс расход.

Все провайдеры OpenAI-совместимы, так что разница между ними сводится к адресу,
идентификатору модели и переменной с ключом — всё это лежит в catalog.py.

Публичное API:
  LLMError                — единственный тип ошибки наружу
  Reply                   — текст ответа вместе с расходом токенов и денег
  call(model_key, msgs)   — синхронный вызов модели
  Client                  — то же самое, но с переиспользуемым HTTP-соединением
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv

from agent import catalog, tokens

load_dotenv()

log = logging.getLogger("agent.llm")

DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
DEFAULT_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))

# Бесплатные тарифы ограничивают не только запросы в минуту, но и токены в
# минуту. Провайдер сам подсказывает, через сколько повторить, поэтому ждать
# тут осмысленно — в отличие от сетевой ошибки, где ждать нечего.
RATE_LIMIT_RETRIES = 4
RATE_LIMIT_PAUSE = 8.0


class LLMError(RuntimeError):
    """Не удалось получить ответ от модели."""


@dataclass
class Reply:
    """Ответ модели вместе с тем, во что он обошёлся."""

    text: str
    model_key: str
    model_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    elapsed: float = 0.0
    finish_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model_id": self.model_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "elapsed": round(self.elapsed, 2),
            "finish_reason": self.finish_reason,
        }


@dataclass
class Client:
    """Клиент к OpenAI-совместимым провайдерам с общим HTTP-соединением."""

    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: float = DEFAULT_TIMEOUT
    # Сводный расход по всем вызовам клиента — по нему интерфейсы показывают,
    # во что обошёлся ответ агента целиком, включая служебные вызовы.
    spent: dict[str, Any] = field(default_factory=lambda: {"calls": 0, "tokens": 0, "cost": 0.0})
    _http: httpx.Client | None = None

    def __post_init__(self) -> None:
        self._http = httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def reset_spent(self) -> None:
        self.spent = {"calls": 0, "tokens": 0, "cost": 0.0}

    def call(
        self,
        model_key: str,
        messages: list[dict[str, str]],
        max_tokens: int = 0,
        temperature: float | None = None,
        low_effort: bool = False,
    ) -> Reply:
        """Зовёт модель и возвращает ответ; при сбое поднимает LLMError.

        low_effort — для механических задач (маршрутизация, сжатие): у моделей
        семейства gpt-oss это убирает скрытое рассуждение, на которое иначе
        уходят сотни выходных токенов.
        """
        model = catalog.get(model_key)
        ключ = model.api_key
        if not ключ:
            raise LLMError(
                f"Не задан ключ API для модели «{model.key}»: нужна переменная "
                f"{model.env_var}. Скопируйте .env.example в .env и впишите ключ."
            )
        # Ключ уходит в HTTP-заголовок, а тот допускает только ASCII. Лишний
        # символ при копировании иначе обернётся невнятной ошибкой кодировки
        # из недр HTTP-клиента.
        if not ключ.isascii():
            raise LLMError(
                f"Ключ {model.env_var} содержит недопустимые символы — вероятно, "
                "при копировании в него попал лишний знак. Проверьте .env."
            )

        payload = self._payload(model, messages, max_tokens or self.max_tokens,
                                self.temperature if temperature is None else temperature,
                                low_effort)
        url = f"{model.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {ключ}", "Content-Type": "application/json"}

        http = self._http or httpx.Client(timeout=self.timeout)
        начало = time.monotonic()
        последняя_ошибка = ""
        с_лимитом = 0
        попытка = 0

        while попытка < 2 + RATE_LIMIT_RETRIES:
            попытка += 1
            log.info("Запрос к %s (попытка %d)", model.api_id, попытка)
            try:
                response = http.post(url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                последняя_ошибка = f"сетевая ошибка — {exc}"
            else:
                if response.status_code in (401, 403):
                    raise LLMError(
                        f"Провайдер {model.provider} отклонил ключ "
                        f"({response.status_code}). Проверьте {model.env_var} в .env."
                    )
                if response.is_success:
                    ответ = self._parse(response, model, time.monotonic() - начало)
                    self.spent["calls"] += 1
                    self.spent["tokens"] += ответ.total_tokens
                    self.spent["cost"] += ответ.cost
                    return ответ
                # 413 у Groq означает не «слишком длинный запрос», а «запрос
                # больше, чем осталось в минутном лимите расхода». Лечится тем
                # же ожиданием, что и 429.
                if response.status_code in (429, 413):
                    подсказка = _подсказка(response)
                    if _суточный_лимит(подсказка):
                        # Минутный лимит проходит сам, суточный — нет. Ждать и
                        # повторять тут значит потратить несколько минут, чтобы
                        # в конце получить ту же ошибку.
                        raise LLMError(
                            f"У {model.provider} закончилась суточная квота: {подсказка} "
                            "Ждать бесполезно — квота обновится только завтра. "
                            "Возьмите модель другого провайдера "
                            "(например, glm-flash или ds-flash)."
                        )
                    с_лимитом += 1
                    if с_лимитом > RATE_LIMIT_RETRIES:
                        raise LLMError(
                            f"Провайдер {model.provider} ограничивает расход: {подсказка}"
                        )
                    пауза = _пауза(response, с_лимитом)
                    log.warning("Лимит расхода, ждём %.0f с: %s", пауза, подсказка)
                    time.sleep(пауза)
                    continue
                последняя_ошибка = (
                    f"провайдер вернул {response.status_code}: {response.text[:300]}"
                )

            if попытка >= 2 + с_лимитом:
                break
            log.warning("Повтор запроса: %s", последняя_ошибка)
            time.sleep(2)

        raise LLMError(f"Не удалось получить ответ от модели {model.key}: {последняя_ошибка}")

    def _payload(
        self,
        model: catalog.Model,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        low_effort: bool,
    ) -> dict[str, Any]:
        """Тело запроса в OpenAI-совместимом формате."""
        # Просить больше, чем помещается в окно модели, бессмысленно: провайдер
        # ответит ошибкой ещё до генерации.
        потолок = max(256, model.context_window // 2)
        payload: dict[str, Any] = {
            "model": model.api_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": min(max_tokens, потолок),
            "stream": False,
        }
        # У DeepSeek V4 режим размышления включён по умолчанию, а в нём
        # temperature и top_p, по документации провайдера, не действуют. Агенту
        # нужна предсказуемость, поэтому режим отключается явно.
        if "api.deepseek.com" in model.base_url:
            payload["thinking"] = {"type": "disabled"}
        if low_effort and model.supports_effort:
            payload["reasoning_effort"] = "low"
        return payload

    @staticmethod
    def _parse(response: httpx.Response, model: catalog.Model, elapsed: float) -> Reply:
        """Достаёт текст и расход из ответа провайдера."""
        try:
            data = response.json()
            выбор = data["choices"][0]
            текст = (выбор["message"].get("content") or "").strip()
            finish = выбор.get("finish_reason", "")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"Неожиданный формат ответа провайдера: {response.text[:300]}"
            ) from exc

        if not текст:
            raise LLMError(f"Модель {model.key} вернула пустой ответ.")

        usage = data.get("usage") or {}
        вход = usage.get("prompt_tokens", 0) or 0
        выход = usage.get("completion_tokens", 0) or 0
        return Reply(
            text=текст,
            model_key=model.key,
            model_id=data.get("model", model.api_id),
            prompt_tokens=вход,
            completion_tokens=выход,
            cost=tokens.cost(вход, выход, model.price_in, model.price_out),
            elapsed=elapsed,
            finish_reason=finish,
        )


def _пауза(response: httpx.Response, попытка: int) -> float:
    """Сколько ждать перед повтором: по заголовку провайдера или с запасом."""
    заголовок = response.headers.get("retry-after", "")
    try:
        return max(1.0, min(60.0, float(заголовок)))
    except ValueError:
        return min(60.0, RATE_LIMIT_PAUSE * попытка)


# Суточная квота отличается от минутной только текстом сообщения: код ответа у
# них один и тот же — 429.
_СУТОЧНЫЕ = ("per day", "tokens per day", "tpd", "requests per day", "rpd",
             "daily", "в сутки")


def _суточный_лимит(сообщение: str) -> bool:
    низ = (сообщение or "").lower()
    return any(признак in низ for признак in _СУТОЧНЫЕ)


def _подсказка(response: httpx.Response) -> str:
    """Человекочитаемое пояснение провайдера про лимит."""
    try:
        сообщение = (response.json().get("error") or {}).get("message", "")
    except ValueError:
        сообщение = ""
    return (сообщение or response.text)[:200]


def call(model_key: str, messages: list[dict[str, str]], **kwargs: Any) -> Reply:
    """Разовый вызов модели без создания клиента — для скриптов и тестов."""
    client = Client()
    try:
        return client.call(model_key, messages, **kwargs)
    finally:
        client.close()
