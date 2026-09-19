"""MemoryManager — единственная точка записи во все три слоя.

Требование дня звучит так: «вы явно выбираете, что и куда сохраняется». Значит,
записи не должны расползаться по коду — иначе через неделю никто не ответит на
вопрос, почему фраза оказалась в профиле. Поэтому запись возможна только через
этот класс, и у каждой записи есть ПРАВИЛО, по которому она сделана.

Правила, в порядке приоритета:

  1. явное-указание   Пользователь сам сказал, куда писать («запомни в знания:…»,
                      ключ CLI --запомни). Модель не привлекается вовсе.
  2. реплика-диалога  Любая реплика пользователя и любой ответ агента идут в
                      краткосрочную память. Дословно, без разбора.
  3. шаг-задачи       Промежуточный результат текущей задачи (план, найденные
                      данные, вывод проверки) идёт в рабочую память.
  4. завершение-задачи При переходе в done рабочая память сворачивается в одну
                      запись журнала решений, после чего задача из рабочей
                      памяти убирается. Сворачивает отдельная средняя модель.
  5. разбор-реплики   Всё остальное: дешёвая модель предлагает слой, и
                      предложение исполняется только выше порога уверенности.

Каждая запись — и применённая, и отклонённая — попадает в журнал маршрутизации
(routing.jsonl). Это и есть прямой ответ на вопрос задания «какие данные
попадают в каждый слой»: журнал читается глазами и не требует веры на слово.

Публичное API:
  MemoryManager(base_dir, user_id, session, client)
  .remember_message(role, text)        — правило «реплика-диалога»
  .remember_step(key, value)           — правило «шаг-задачи»
  .remember_explicit(target, ...)      — правило «явное-указание»
  .route(text, apply=True)             — правило «разбор-реплики»
  .finish_task(state)                  — правило «завершение-задачи»
  .journal(limit) / .stats() / .files()
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from agent import catalog
from agent.invariants import InvariantStore, merge
from agent.llm import Client, LLMError
from agent.memory.long import LongTermError, LongTermMemory
from agent.memory.router import DEFAULT_THRESHOLD, Router, Routing
from agent.memory.short import ShortTermMemory
from agent.memory.working import DONE, TaskState, WorkingMemory

log = logging.getLogger("agent.memory")

SHORT = "краткосрочная"
WORKING = "рабочая"
LONG = "долговременная"
LAYERS = (SHORT, WORKING, LONG)

# Режимы разбора свободных реплик.
AUTO = "авто"        # писать самому, если уверенность выше порога
ASK = "спросить"     # только предлагать; решение принимает пользователь
OFF = "выкл"         # не звать маршрутизатор вообще
ROUTER_MODES = (AUTO, ASK, OFF)

DEFAULT_DIR = os.getenv("MEMORY_DIR", "memory")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryManager:
    """Три слоя памяти и единственный вход на запись в них."""

    def __init__(
        self,
        base_dir: str = DEFAULT_DIR,
        user_id: str = "инженер",
        session: str = "основная",
        client: Client | None = None,
        router_mode: str = AUTO,
        threshold: float = DEFAULT_THRESHOLD,
        router_model: str = "",
        summarizer_model: str = "",
    ) -> None:
        if router_mode not in ROUTER_MODES:
            raise ValueError(
                f"Неизвестный режим маршрутизатора «{router_mode}». "
                f"Допустимы: {', '.join(ROUTER_MODES)}."
            )
        self.base_dir = base_dir
        self.user_id = user_id
        self.session = session
        self.router_mode = router_mode
        self.client = client

        os.makedirs(base_dir, exist_ok=True)
        # Три слоя — три разных хранилища. Это не украшение: диалог должен
        # быстро дописываться и так же быстро стираться, состояние задачи —
        # переживать очистку диалога, а профиль и знания — читаться глазами и
        # править руками. Один общий файл не даёт ни того, ни другого, ни третьего.
        self.short = ShortTermMemory(os.path.join(base_dir, "dialog.db"))
        self.working = WorkingMemory(os.path.join(base_dir, "working"))
        self.long = LongTermMemory(os.path.join(base_dir, "long"), user_id)

        # Инварианты проекта лежат отдельным файлом на уровне всей памяти, а не
        # в профиле пользователя. Профиль принадлежит человеку, инварианты —
        # проекту: переключение «--кто» не должно менять то, что запрещено.
        self.invariants = InvariantStore(os.path.join(base_dir, "invariants.json"))

        self.journal_path = os.path.join(base_dir, "routing.jsonl")
        self.router = (
            Router(client, model_key=router_model, threshold=threshold)
            if client is not None else None
        )
        self.threshold = threshold
        self.summarizer_model = summarizer_model or catalog.for_role("сжатие", offset=0)

    # --- журнал --------------------------------------------------------------

    def _log(
        self,
        rule: str,
        layer: str,
        sublayer: str,
        text: str,
        applied: bool,
        reason: str = "",
        model_key: str = "",
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Записывает решение о маршрутизации — и применённое, и отклонённое."""
        запись = {
            "когда": _now(),
            "сессия": self.session,
            "пользователь": self.user_id,
            "правило": rule,
            "слой": layer,
            "подслой": sublayer,
            "текст": text[:300],
            "применено": applied,
            "причина": reason,
            "модель": model_key,
            "уверенность": None if confidence is None else round(confidence, 2),
        }
        try:
            with open(self.journal_path, "a", encoding="utf-8") as файл:
                файл.write(json.dumps(запись, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("Не удалось записать журнал маршрутизации: %s", exc)
        return запись

    def journal(self, limit: int = 20) -> list[dict[str, Any]]:
        """Последние решения маршрутизации; битые строки пропускаются."""
        try:
            with open(self.journal_path, encoding="utf-8") as файл:
                строки = файл.readlines()
        except FileNotFoundError:
            return []
        записи = []
        for строка in строки:
            строка = строка.strip()
            if not строка:
                continue
            try:
                записи.append(json.loads(строка))
            except json.JSONDecodeError:
                continue
        return записи[-limit:] if limit > 0 else записи

    # --- правило 2: реплика диалога ------------------------------------------

    def remember_message(
        self,
        role: str,
        text: str,
        task_id: str = "",
        stage: str = "",
        tokens: int = 0,
        cost: float = 0.0,
    ) -> int:
        """Кладёт реплику в краткосрочную память. Без разбора и без модели."""
        номер = self.short.append(
            self.session, role, text, task_id=task_id, stage=stage, tokens=tokens, cost=cost
        )
        self._log("реплика-диалога", SHORT, "", text, applied=True,
                  reason=f"роль {role}, номер {номер}")
        return номер

    # --- правило 3: шаг задачи -----------------------------------------------

    def remember_step(self, state: TaskState, key: str, value: str) -> TaskState:
        """Кладёт промежуточный результат в рабочую память текущей задачи.

        Сюда идёт то, что нужно до конца задачи и не нужно после: список таблиц,
        найденное имя контроллера, вывод makemigrations. Долговременная память
        от таких записей только замусорилась бы.
        """
        state.remember(key, value)
        self.working.save(state)
        self._log("шаг-задачи", WORKING, state.task_id, f"{key}: {value}", applied=True,
                  reason=f"стадия {state.stage}")
        return state

    def remember_plan(self, state: TaskState, steps: list[str]) -> TaskState:
        """План задачи — тоже рабочая память, но отдельным полем."""
        state.set_plan(steps)
        self.working.save(state)
        self._log("шаг-задачи", WORKING, state.task_id,
                  "; ".join(steps)[:300], applied=True, reason="план задачи")
        return state

    # --- правило 1: явное указание -------------------------------------------

    def remember_explicit(
        self,
        target: str,
        value: str,
        key: str = "",
        section: str = "",
        task_id: str = "",
        reason: str = "",
        alternatives: list[str] | None = None,
    ) -> dict[str, Any]:
        """Пишет туда, куда велено. Самый приоритетный путь: модель не участвует.

        target: профиль | знания | решения | рабочая | краткосрочная
        """
        target = (target or "").strip().lower()
        if target == "профиль":
            self.long.profile.update(section or "ограничения", key, value)
            слой, подслой = LONG, "профиль"
        elif target == "знания":
            self.long.knowledge.add(key or value[:40], value, topic=section,
                                    tags=[т for т in (section or "").split() if т],
                                    source="указано пользователем")
            слой, подслой = LONG, "знания"
        elif target == "решения":
            self.long.decisions.add(key or value[:60], value, reason=reason,
                                    alternatives=alternatives, task_id=task_id,
                                    source="указано пользователем")
            слой, подслой = LONG, "решения"
        elif target == "рабочая":
            if not task_id:
                raise LongTermError("Чтобы писать в рабочую память, нужна текущая задача.")
            состояние = self.working.load(task_id)
            состояние.remember(key or value[:40], value)
            self.working.save(состояние)
            слой, подслой = WORKING, task_id
        elif target == "краткосрочная":
            self.short.append(self.session, "user", value)
            слой, подслой = SHORT, ""
        else:
            raise LongTermError(
                f"Неизвестный слой «{target}». Допустимы: профиль, знания, решения, "
                "рабочая, краткосрочная."
            )
        return self._log("явное-указание", слой, подслой, value, applied=True,
                         reason="указано пользователем")

    # --- правило 5: разбор свободной реплики ---------------------------------

    def route(self, text: str, apply: bool = True) -> tuple[Routing, dict[str, Any]]:
        """Спрашивает маршрутизатор, стоит ли что-то сохранить из реплики.

        Возвращает предложение и запись журнала. Предложение исполняется только
        при apply=True, режиме «авто» и уверенности выше порога — остальное
        попадает в журнал как отклонённое, чтобы потом было видно, что именно
        агент решил не запоминать.
        """
        if self.router_mode == OFF or self.router is None:
            запись = self._log("разбор-реплики", SHORT, "", text, applied=False,
                               reason="маршрутизатор выключен")
            return Routing(why="маршрутизатор выключен"), запись

        предложение = self.router.classify(text)

        if предложение.failed:
            запись = self._log("разбор-реплики", SHORT, "", text, applied=False,
                               reason="маршрутизатор не справился с ответом",
                               model_key=предложение.model_key)
            return предложение, запись

        if not предложение.wants_write:
            запись = self._log("разбор-реплики", SHORT, "", text, applied=False,
                               reason=предложение.why or "реплика диалога",
                               model_key=предложение.model_key,
                               confidence=предложение.confidence)
            return предложение, запись

        слой = LONG
        подслой = предложение.target
        ниже_порога = предложение.confidence < self.threshold
        только_предложить = self.router_mode == ASK or not apply

        if ниже_порога or только_предложить:
            причина = (
                f"уверенность {предложение.confidence:.2f} ниже порога {self.threshold:.2f}"
                if ниже_порога else "режим «спросить»: ждём подтверждения"
            )
            запись = self._log("разбор-реплики", слой, подслой, предложение.value,
                               applied=False, reason=причина,
                               model_key=предложение.model_key,
                               confidence=предложение.confidence)
            return предложение, запись

        self._apply(предложение)
        запись = self._log("разбор-реплики", слой, подслой, предложение.value, applied=True,
                           reason=предложение.why, model_key=предложение.model_key,
                           confidence=предложение.confidence)
        return предложение, запись

    def apply_routing(self, routing: Routing) -> dict[str, Any]:
        """Исполняет отложенное предложение — этим пользуется режим «спросить»."""
        self._apply(routing)
        return self._log("разбор-реплики", LONG, routing.target, routing.value, applied=True,
                         reason="подтверждено пользователем",
                         model_key=routing.model_key, confidence=routing.confidence)

    def _apply(self, routing: Routing) -> None:
        if routing.target == "профиль":
            self.long.profile.update(routing.section or "ограничения",
                                     routing.key, routing.value)
        elif routing.target == "знания":
            self.long.knowledge.add(routing.key or routing.value[:40], routing.value,
                                    source="разбор реплики")
        elif routing.target == "решения":
            self.long.decisions.add(routing.key or routing.value[:60], routing.value,
                                    reason=routing.why, source="разбор реплики")

    # --- правило 4: завершение задачи ----------------------------------------

    def finish_task(self, state: TaskState, note: str = "",
                    model_key: str = "") -> dict[str, Any]:
        """Сворачивает завершённую задачу в запись журнала решений.

        Это единственный штатный мост между рабочей и долговременной памятью.
        Рабочая память после свёртки очищается: держать в ней завершённое —
        значит носить в промпте подробности, которые уже никому не нужны.
        """
        if state.stage != DONE:
            state.transition(DONE, note or "задача завершена")
            self.working.save(state)

        сводка = self._summarize(state, model_key)
        запись = self.long.decisions.add(
            title=сводка["заголовок"],
            decision=сводка["решение"],
            reason=сводка["причина"],
            alternatives=сводка.get("альтернативы", []),
            task_id=state.task_id,
            source="свёртка задачи",
        )
        self._log("завершение-задачи", LONG, "решения", сводка["решение"], applied=True,
                  reason=f"задача {state.task_id} свёрнута в решение",
                  model_key=сводка.get("модель", ""))
        self.working.drop(state.task_id)
        self._log("завершение-задачи", WORKING, state.task_id, "", applied=True,
                  reason="рабочая память задачи очищена после свёртки")
        return запись

    def _summarize(self, state: TaskState, model_key: str = "") -> dict[str, Any]:
        """Сжимает состояние задачи в решение. Модель — средняя, не самая дешёвая.

        model_key приходит от агента, когда пользователь выбрал модель явно.
        Без этого выходило неожиданное: человек переключается на DeepSeek, чтобы
        не упираться в суточную квоту Groq, пять шагов сценария честно идут на
        DeepSeek — а свёртка в конце всё равно уходит в Groq по своей роли и
        роняет весь прогон на последнем шаге.

        Сжатие — та самая точка, где экономия выходит боком: слабая модель теряет
        именно причину решения, а ради причины журнал и ведётся. Если модели нет
        под рукой или она недоступна, свёртка делается кодом — без красот, но
        без потерь.
        """
        план = "\n".join(f"- {ш}" for ш in state.plan) or "— плана не было"
        данные = "\n".join(f"- {к}: {з}" for к, з in state.collected.items()) or "— пусто"
        запасной = {
            "заголовок": state.title or state.task_id,
            "решение": f"Задача «{state.task_id}» завершена.\nПлан:\n{план}\nСобрано:\n{данные}",
            "причина": "свёртка без модели: пересказ состояния задачи как есть",
            "альтернативы": [],
            "модель": "",
        }
        if self.client is None:
            return запасной

        сообщения = [
            {
                "role": "system",
                "content": (
                    "Ты сворачиваешь завершённую инженерную задачу в одну запись журнала "
                    "решений. Отвечай ТОЛЬКО JSON без markdown: "
                    '{"заголовок":"до 70 символов","решение":"что сделано и к чему пришли, '
                    '3-5 предложений","причина":"почему именно так","альтернативы":["что '
                    'рассматривали и отклонили"]}. Не выдумывай того, чего нет во входных '
                    "данных: чего не было — оставь пустым."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Задача: {state.task_id} — {state.title}\n"
                    f"Стадия: {state.stage}\n"
                    f"Переходов: {len(state.transitions)}\n\nПлан:\n{план}\n\nСобранные данные:\n{данные}"
                ),
            },
        ]
        try:
            ответ = self.client.call(model_key or self.summarizer_model, сообщения,
                                     max_tokens=700, temperature=0.2, low_effort=True)
        except LLMError as exc:
            log.warning("Свёртка задачи без модели (%s)", exc)
            return запасной

        текст = ответ.text.strip()
        начало, конец = текст.find("{"), текст.rfind("}")
        if начало != -1 and конец > начало:
            try:
                данные_ответа = json.loads(текст[начало:конец + 1])
            except json.JSONDecodeError:
                данные_ответа = {}
        else:
            данные_ответа = {}

        if not данные_ответа.get("решение"):
            return запасной
        return {
            "заголовок": str(данные_ответа.get("заголовок") or state.title or state.task_id)[:70],
            "решение": str(данные_ответа["решение"]),
            "причина": str(данные_ответа.get("причина", "")),
            "альтернативы": [str(а) for а in (данные_ответа.get("альтернативы") or [])],
            "модель": ответ.model_key,
        }

    # --- сводка --------------------------------------------------------------

    def all_invariants(self) -> list:
        """Проектные инварианты плюс личные из профиля.

        Личные могут только добавить запретов: совпадение кодов решается в
        пользу проектного. Ослабить общее ограничение под себя нельзя — это
        ровно тот случай, ради которого инварианты и заводят.
        """
        from agent.invariants import Invariant
        личные = [
            Invariant.from_dict(и)
            for и in (self.long.profile.load().get("инварианты") or [])
        ]
        return merge(self.invariants.all(), личные)

    def stats(self) -> dict[str, Any]:
        """Сколько записей в каждом слое — этим интерфейсы показывают наполнение."""
        коротко = self.short.stats(self.session)
        долго = self.long.stats()
        return {
            SHORT: {
                "реплик": коротко["messages"],
                "символов": коротко["chars"],
                "файл": self.short.path,
            },
            WORKING: {
                "задач": len(self.working.tasks()),
                "каталог": self.working.directory,
            },
            LONG: {
                "инвариантов_проекта": len(self.invariants.all()),
                "профиль": долго["предпочтения"] + долго["ограничения"],
                "настроен": долго["настроен"],
                "сводка": долго["сводка"],
                "инвариантов": долго["инварианты"],
                "решений": долго["решения"],
                "знаний": долго["знания"],
                "сценариев": долго["сценарии"],
                "каталог": self.long.directory,
            },
        }

    def files(self) -> dict[str, str]:
        """Где физически лежит каждый слой — чтобы разделение было видно глазами."""
        пути = {
            "краткосрочная (диалог)": self.short.path,
            "рабочая (задачи)": self.working.directory,
        }
        пути["инварианты проекта"] = self.invariants.path
        пути.update({f"долговременная ({к})": п for к, п in self.long.files().items()})
        пути["журнал маршрутизации"] = self.journal_path
        return пути
