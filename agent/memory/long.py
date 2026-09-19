"""Долговременная память: профиль, решения и знания — три отдельных хранилища.

Формулировка задания — «долговременная (профиль, решения, знания)» — это не
перечисление синонимов, а подсказка на три разных по природе вещи, которые
нельзя валить в один файл:

  ПРОФИЛЬ (profile.json) — кто перед агентом и в каких рамках он работает:
  стиль ответов, ограничения по стеку, зачем всё это нужно, и отдельным
  списком — инварианты. Перезаписывается: у пользователя один текущий профиль,
  история его правок никого не интересует.

  РЕШЕНИЯ (decisions.jsonl) — журнал того, что уже решено и почему. Дописывается
  и никогда не переписывается: смысл записи «выбрали Django, отклонили Laravel,
  потому что…» именно в том, что через полгода её можно прочитать дословно.
  Формат JSON Lines выбран ради этого — дописать строку в конец файла нельзя
  «наполовину», в отличие от перезаписи целого JSON-массива.

  ЗНАНИЯ (knowledge.json) — факты предметной области: структура legacy-схем,
  имена контроллеров, доменная логика сети. В отличие от решений, знание не
  привязано ко времени и его можно уточнять. В отличие от профиля, знаний
  много, и в промпт они идут не целиком, а отбором по релевантности.

Разделение проверяется глазами: три файла, и видно, что где лежит.

Публичное API:
  ProfileStore(path)     — .load() / .save() / .update(...) / .invariants()
  DecisionsStore(path)   — .add(...) / .all() / .recent(n)
  KnowledgeStore(path)   — .add(...) / .all() / .relevant(query, tags, limit) / .forget(id)
  LongTermMemory(dir)    — все три вместе для одного пользователя
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from agent import preferences
from agent.memory.working import SAFE_NAME
from agent.scenarios import ScenarioStore

# Тип инварианта определяет, как его проверять. Жёсткие проверяются кодом —
# это принципиально: модель, которую попросили «проверить, нет ли тут Laravel»,
# ошибается и поддаётся уговорам, а поиск по тексту — нет.
#
#   запрет-слов       слова не должны появляться ни в коде, ни в тексте (кроме
#                     явного отказа от них: «Laravel здесь не подойдёт»);
#   запрет-в-коде     слово можно упоминать, но нельзя писать на нём код. Этот
#                     тип появился не сразу: первая версия запрещала слово
#                     «CodeIgniter» целиком — и ловила ложные срабатывания на
#                     каждом втором ответе, потому что профиль сам требует
#                     указывать, какому legacy-компоненту отвечает решение;
#   запрет-регулярок  выражение не должно встречаться нигде — этим проверяются
#                     секреты в URL и подобные шаблоны.
HARD_TYPES = ("запрет-слов", "запрет-в-коде", "запрет-регулярок")
SOFT_TYPE = "мягкий"

_СЛОВО = re.compile(r"[a-zA-Zа-яёА-ЯЁ0-9_]{3,}")

# Слова, которые в запросе ничего не говорят о теме и только мешают отбору
# знаний по совпадению слов.
_ШУМ = {
    "как", "что", "для", "или", "это", "нам", "мне", "нужно", "надо", "быть",
    "есть", "под", "при", "без", "над", "про", "все", "всё", "его", "их", "them",
    "the", "and", "for", "with", "you", "can", "мы", "вы", "они", "который",
    "которая", "которые", "чтобы", "если", "тогда", "потом", "также", "ещё",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LongTermError(RuntimeError):
    """Ошибка работы с долговременной памятью."""


def _write_json(path: str, data: Any) -> None:
    """Пишет JSON атомарно: временный файл и замена, без промежуточного состояния."""
    временный = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(временный, "w", encoding="utf-8") as файл:
            json.dump(data, файл, ensure_ascii=False, indent=2)
        os.replace(временный, path)
    except OSError as exc:
        raise LongTermError(f"Не удалось записать «{path}»: {exc}") from exc


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as файл:
            return json.load(файл)
    except FileNotFoundError:
        return default
    except (OSError, json.JSONDecodeError) as exc:
        raise LongTermError(f"Файл «{path}» испорчен: {exc}") from exc


def _words(text: str) -> set[str]:
    return {с.lower() for с in _СЛОВО.findall(text or "")} - _ШУМ


# --- профиль ------------------------------------------------------------------

class ProfileStore:
    """Профиль пользователя: стиль, ограничения, контекст и инварианты."""

    layer = "долговременная"
    sublayer = "профиль"

    def __init__(self, path: str) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        """Текущий профиль, дополненный умолчаниями предпочтений.

        Умолчания подставляются при каждом чтении, а не при создании файла:
        профиль правят руками и присылают формы, а список предпочтений со
        временем растёт. Иначе профиль, записанный вчера, завтра оказался бы
        неполным, и половина кода проверяла бы ключи на существование.
        """
        каркас = {
            "user_id": "",
            "ограничения": {},
            "контекст": "",
            "инварианты": [],
            "обновлён": "",
        }
        данные = _read_json(self.path, каркас)
        каркас.update(данные if isinstance(данные, dict) else {})
        return preferences.normalize(каркас)

    def save(self, profile: dict[str, Any]) -> None:
        profile["обновлён"] = _now()
        _write_json(self.path, profile)

    def update(self, section: str, key: str, value: str) -> dict[str, Any]:
        """Правит одну запись профиля: стиль, ограничения или контекст.

        Раздел и ключ задаются явно — маршрутизатор не может «просто дописать
        что-нибудь в профиль», он обязан сказать, во что именно пишет.
        """
        профиль = self.load()
        if section == "контекст":
            профиль["контекст"] = value
        elif section in preferences.SECTIONS:
            # Предпочтения типизированы: значение вне списка допустимых
            # отклоняется здесь, а не превращается в строку, которую потом никто
            # не сможет ни применить, ни проверить.
            if not key:
                raise LongTermError(f"Для раздела «{section}» нужен ключ записи.")
            try:
                профиль = preferences.set_value(профиль, section, key, value)
            except preferences.PreferenceError as exc:
                raise LongTermError(str(exc)) from exc
        elif section == "ограничения":
            if not key:
                raise LongTermError("Для раздела «ограничения» нужен ключ записи.")
            профиль.setdefault(section, {})[key] = value
        else:
            разделы = ", ".join(preferences.SECTIONS) + ", ограничения, контекст"
            raise LongTermError(
                f"Неизвестный раздел профиля «{section}». Допустимы: {разделы}."
            )
        self.save(профиль)
        return профиль

    def forget(self, section: str, key: str) -> bool:
        профиль = self.load()
        if section in ("ограничения",) and key in профиль.get(section, {}):
            del профиль[section][key]
            self.save(профиль)
            return True
        return False

    # --- инварианты ----------------------------------------------------------

    def invariants(self, hard_only: bool = False) -> list[dict[str, Any]]:
        """Инварианты профиля; hard_only — только те, что проверяет код."""
        все = self.load().get("инварианты", [])
        if hard_only:
            return [и for и in все if и.get("тип") in HARD_TYPES]
        return все

    def add_invariant(self, invariant: dict[str, Any]) -> None:
        """Добавляет инвариант, проверив, что его вообще можно проверить."""
        код = (invariant.get("код") or "").strip()
        тип = invariant.get("тип", SOFT_TYPE)
        if not код:
            raise LongTermError("У инварианта должен быть код.")
        if тип not in HARD_TYPES + (SOFT_TYPE,):
            raise LongTermError(
                f"Неизвестный тип инварианта «{тип}». "
                f"Допустимы: {', '.join(HARD_TYPES + (SOFT_TYPE,))}."
            )
        if тип in HARD_TYPES and not invariant.get("значения"):
            raise LongTermError(
                f"Жёсткий инвариант «{код}» без значений проверить нечем: "
                "нужен список запрещённых слов или регулярных выражений."
            )
        if тип == "запрет-регулярок":
            for выражение in invariant["значения"]:
                try:
                    re.compile(выражение)
                except re.error as exc:
                    raise LongTermError(
                        f"Инвариант «{код}»: выражение «{выражение}» не компилируется: {exc}"
                    ) from exc
        профиль = self.load()
        профиль.setdefault("инварианты", [])
        профиль["инварианты"] = [и for и in профиль["инварианты"] if и.get("код") != код]
        профиль["инварианты"].append(invariant)
        self.save(профиль)


# --- решения ------------------------------------------------------------------

class DecisionsStore:
    """Журнал решений: append-only, одна строка JSON на решение."""

    layer = "долговременная"
    sublayer = "решения"

    def __init__(self, path: str) -> None:
        self.path = path

    def add(
        self,
        title: str,
        decision: str,
        reason: str = "",
        alternatives: list[str] | None = None,
        task_id: str = "",
        source: str = "пользователь",
    ) -> dict[str, Any]:
        """Дописывает решение в конец журнала.

        source показывает, откуда решение взялось: «пользователь» — сказано
        прямо, «свёртка задачи» — получено сжатием рабочей памяти при завершении.
        """
        title = (title or "").strip()
        decision = (decision or "").strip()
        if not title or not decision:
            raise LongTermError("У решения должны быть и заголовок, и сам текст решения.")
        запись = {
            "id": len(self.all()) + 1,
            "заголовок": title,
            "решение": decision,
            "причина": (reason or "").strip(),
            "альтернативы": alternatives or [],
            "task_id": task_id,
            "источник": source,
            "создано": _now(),
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as файл:
                файл.write(json.dumps(запись, ensure_ascii=False) + "\n")
        except OSError as exc:
            raise LongTermError(f"Не удалось записать решение: {exc}") from exc
        return запись

    def all(self) -> list[dict[str, Any]]:
        """Весь журнал в порядке появления; битые строки пропускаются."""
        try:
            with open(self.path, encoding="utf-8") as файл:
                строки = файл.readlines()
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise LongTermError(f"Не удалось прочитать журнал решений: {exc}") from exc

        итог = []
        for строка in строки:
            строка = строка.strip()
            if not строка:
                continue
            try:
                итог.append(json.loads(строка))
            except json.JSONDecodeError:
                continue
        return итог

    def recent(self, limit: int = 5) -> list[dict[str, Any]]:
        """Последние решения — именно они чаще всего и нужны в промпте."""
        return self.all()[-limit:] if limit > 0 else []


# --- знания -------------------------------------------------------------------

class KnowledgeStore:
    """Факты предметной области с тегами; в промпт идут отбором, а не целиком."""

    layer = "долговременная"
    sublayer = "знания"

    def __init__(self, path: str) -> None:
        self.path = path

    def all(self) -> list[dict[str, Any]]:
        данные = _read_json(self.path, [])
        return данные if isinstance(данные, list) else []

    def add(
        self,
        fact_id: str,
        text: str,
        topic: str = "",
        tags: list[str] | None = None,
        source: str = "",
    ) -> dict[str, Any]:
        """Добавляет или уточняет факт. Совпадение по fact_id заменяет прежний."""
        fact_id = (fact_id or "").strip()
        text = (text or "").strip()
        if not fact_id or not text:
            raise LongTermError("У факта должны быть идентификатор и текст.")
        запись = {
            "id": fact_id,
            "тема": topic,
            "текст": text,
            "теги": [т.lower().strip() for т in (tags or []) if т.strip()],
            "источник": source,
            "обновлён": _now(),
        }
        факты = [ф for ф in self.all() if ф.get("id") != fact_id]
        факты.append(запись)
        _write_json(self.path, факты)
        return запись

    def forget(self, fact_id: str) -> bool:
        факты = self.all()
        осталось = [ф for ф in факты if ф.get("id") != fact_id]
        if len(осталось) == len(факты):
            return False
        _write_json(self.path, осталось)
        return True

    def relevant(
        self,
        query: str = "",
        tags: list[str] | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        """Отбирает факты, относящиеся к запросу, и ранжирует их по совпадению.

        Отбор нарочно простой — пересечение слов запроса со словами факта плюс
        совпадение тегов. Векторный поиск точнее, но требует отдельной модели
        эмбеддингов; на десятках фактов проекта разницы не будет, а зависимость
        появится. Порог ставится здесь: если не совпало ничего, факт в промпт
        не идёт — лучше не подсказать, чем набить контекст посторонним.
        """
        if limit <= 0:
            return []
        слова_запроса = _words(query)
        теги = {т.lower() for т in (tags or [])}

        оценённые = []
        for факт in self.all():
            слова_факта = _words(факт.get("текст", "") + " " + факт.get("тема", ""))
            теги_факта = {т.lower() for т in факт.get("теги", [])}
            # Совпадение по тегу весомее случайного совпадения слова: тег
            # ставится руками и означает «это про planning» или «это про 1С».
            вес = 2.0 * len(теги & теги_факта) + len(слова_запроса & слова_факта)
            вес += 1.5 * len(слова_запроса & теги_факта)
            if вес > 0:
                оценённые.append((вес, факт))

        оценённые.sort(key=lambda пара: (-пара[0], пара[1].get("id", "")))
        return [факт for _, факт in оценённые[:limit]]


# --- всё вместе ---------------------------------------------------------------

class LongTermMemory:
    """Три хранилища одного пользователя в одном каталоге."""

    layer = "долговременная"

    def __init__(self, directory: str, user_id: str) -> None:
        # Имя пользователя превращается в путь, поэтому проверяется так же
        # строго, как идентификатор задачи: без этого «--кто ../../чужой»
        # пишет профиль за пределы каталога памяти — проверено, писало.
        if not SAFE_NAME.match(user_id or ""):
            raise LongTermError(
                f"Недопустимое имя пользователя «{user_id}»: нужны буквы, цифры, "
                "дефис и подчёркивание, до 64 символов."
            )
        self.user_id = user_id
        self.directory = os.path.join(directory, user_id)
        os.makedirs(self.directory, exist_ok=True)
        self.profile = ProfileStore(os.path.join(self.directory, "profile.json"))
        self.decisions = DecisionsStore(os.path.join(self.directory, "decisions.jsonl"))
        self.knowledge = KnowledgeStore(os.path.join(self.directory, "knowledge.json"))
        # Сценарии лежат отдельным файлом, а не внутри профиля: профиль целиком
        # уходит в каждый запрос, а сценарий — нет, в промпт идёт только текущий
        # шаг. См. agent/scenarios.py.
        self.scenarios = ScenarioStore(os.path.join(self.directory, "scenarios.json"))

    def stats(self) -> dict[str, Any]:
        профиль = self.profile.load()
        return {
            "user_id": self.user_id,
            "настроен": bool(профиль.get("настроен")),
            "предпочтения": sum(len(профиль.get(р, {})) for р in preferences.SECTIONS),
            "сводка": preferences.summary(профиль),
            "ограничения": len(профиль.get("ограничения", {})),
            "инварианты": len(профиль.get("инварианты", [])),
            "решения": len(self.decisions.all()),
            "знания": len(self.knowledge.all()),
            "сценарии": len(self.scenarios.all()),
        }

    def files(self) -> dict[str, str]:
        """Пути к файлам — интерфейсы показывают их, чтобы слои были видны глазами."""
        return {
            "профиль": self.profile.path,
            "сценарии": self.scenarios.path,
            "решения": self.decisions.path,
            "знания": self.knowledge.path,
        }
