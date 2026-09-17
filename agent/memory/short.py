"""Краткосрочная память: реплики текущего диалога.

Это самый быстротечный слой. Сюда попадает всё, что сказали друг другу
пользователь и агент, — дословно и без разбора. Ценность этих записей падает с
каждой новой репликой: через двадцать сообщений подробности первого вопроса уже
не нужны, а вот местоимения и уточнения двух последних реплик критичны.
Поэтому слой устроен как окно: хранится всё, а в промпт уходит хвост.

Почему SQLite, а не список в оперативной памяти: диалог должен переживать
перезапуск процесса — это уже проверено в предыдущих днях. Запись атомарна, и
оборванный на полуслове процесс не оставляет после себя испорченный файл.

Чего здесь принципиально НЕТ: выводов, договорённостей и фактов. Реплика
«переходим на PostGIS 3.6» лежит тут как реплика, но как договорённость она
живёт в долговременной памяти — туда её кладёт MemoryManager отдельным
решением. Разделение намеренное: короткая память забывчива по своей природе, и
хранить в ней то, что нужно помнить неделями, — ошибка проектирования.

Публичное API:
  ShortTermMemory(path)                — открыть (и при необходимости создать) базу
  .append(session, role, content)      — дописать реплику
  .window(session, messages, chars)    — хвост диалога для промпта
  .all(session)                        — вся история сессии
  .sessions()                          — список сессий со сводкой
  .clear(session)                      — забыть диалог
  .stats(session)                      — счётчики по сессии
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

# Роли в терминах OpenAI-совместимого API: история хранится сразу в том виде,
# в котором её потом отдавать модели.
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLES = (ROLE_USER, ROLE_ASSISTANT)

DEFAULT_SESSION = "основная"

# Два ограничения работают вместе: по числу сообщений — чтобы диалог не
# разрастался, по символам — чтобы одна огромная реплика не съела весь контекст.
DEFAULT_MAX_MESSAGES = 10
DEFAULT_MAX_CHARS = 6_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session    TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    task_id    TEXT NOT NULL DEFAULT '',
    stage      TEXT NOT NULL DEFAULT '',
    tokens     INTEGER NOT NULL DEFAULT 0,
    cost       REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ShortTermError(RuntimeError):
    """Ошибка работы с краткосрочной памятью."""


class ShortTermMemory:
    """Диалог текущей сессии поверх файла SQLite."""

    layer = "краткосрочная"

    def __init__(self, path: str) -> None:
        """Открывает базу, создавая файл и таблицу, если их ещё нет.

        path=":memory:" даёт временное хранилище в оперативной памяти — им
        пользуются тесты, чтобы не трогать файл на диске.
        """
        self.path = path
        try:
            # check_same_thread=False: веб-сервер обслуживает запросы в разных
            # потоках, а запись у нас короткая и защищена самим SQLite.
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.executescript(_SCHEMA)
            self._db.commit()
        except sqlite3.Error as exc:
            raise ShortTermError(f"Не удалось открыть диалог «{path}»: {exc}") from exc

    # --- запись --------------------------------------------------------------

    def append(
        self,
        session: str,
        role: str,
        content: str,
        task_id: str = "",
        stage: str = "",
        tokens: int = 0,
        cost: float = 0.0,
    ) -> int:
        """Дописывает реплику и возвращает её номер.

        task_id и stage записываются рядом не для промпта, а для разбора: по ним
        потом видно, на каком этапе задачи прозвучала та или иная фраза.
        """
        if role not in ROLES:
            raise ShortTermError(f"Неизвестная роль «{role}». Допустимы: {', '.join(ROLES)}.")
        content = (content or "").strip()
        if not content:
            raise ShortTermError("Пустая реплика не сохраняется.")
        try:
            курсор = self._db.execute(
                "INSERT INTO messages (session, role, content, task_id, stage, tokens, cost,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session, role, content, task_id, stage, tokens, cost, _now()),
            )
            self._db.commit()
        except sqlite3.Error as exc:
            raise ShortTermError(f"Не удалось сохранить реплику: {exc}") from exc
        return int(курсор.lastrowid or 0)

    # --- чтение --------------------------------------------------------------

    def all(self, session: str = DEFAULT_SESSION) -> list[dict[str, Any]]:
        """Вся история сессии в порядке появления."""
        строки = self._db.execute(
            "SELECT * FROM messages WHERE session = ? ORDER BY id", (session,)
        ).fetchall()
        return [dict(с) for с in строки]

    def window(
        self,
        session: str = DEFAULT_SESSION,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> list[dict[str, str]]:
        """Хвост диалога для отправки модели: последние реплики в пределах лимитов.

        Отбор идёт с конца: свежие реплики важнее давних. Ограничение по
        символам срабатывает раньше ограничения по числу сообщений, если в
        диалоге была одна очень длинная реплика.
        """
        if max_messages <= 0:
            return []
        строки = self._db.execute(
            "SELECT role, content FROM messages WHERE session = ? ORDER BY id DESC LIMIT ?",
            (session, max_messages),
        ).fetchall()

        окно: list[dict[str, str]] = []
        объём = 0
        for строка in строки:
            длина = len(строка["content"])
            if окно and объём + длина > max_chars:
                break
            окно.append({"role": строка["role"], "content": строка["content"]})
            объём += длина
        окно.reverse()
        return окно

    def sessions(self) -> list[dict[str, Any]]:
        """Все сессии со сводкой: сколько реплик и когда была последняя."""
        строки = self._db.execute(
            "SELECT session, COUNT(*) AS messages, MAX(created_at) AS last"
            " FROM messages GROUP BY session ORDER BY last DESC"
        ).fetchall()
        return [dict(с) for с in строки]

    def stats(self, session: str = DEFAULT_SESSION) -> dict[str, Any]:
        """Счётчики сессии: реплики, символы, токены, деньги."""
        строка = self._db.execute(
            "SELECT COUNT(*) AS messages, COALESCE(SUM(LENGTH(content)), 0) AS chars,"
            " COALESCE(SUM(tokens), 0) AS tokens, COALESCE(SUM(cost), 0) AS cost"
            " FROM messages WHERE session = ?",
            (session,),
        ).fetchone()
        итог = dict(строка)
        итог["session"] = session
        return итог

    # --- очистка -------------------------------------------------------------

    def clear(self, session: str = DEFAULT_SESSION) -> int:
        """Забывает диалог сессии и возвращает число стёртых реплик.

        Долговременная и рабочая память при этом не трогаются: «начать разговор
        заново» и «забыть проект» — разные действия, и путать их нельзя.
        """
        курсор = self._db.execute("DELETE FROM messages WHERE session = ?", (session,))
        self._db.commit()
        return курсор.rowcount

    def close(self) -> None:
        self._db.close()
