"""PromptBuilder — единственная точка ЧТЕНИЯ памяти.

MemoryManager отвечает за то, что куда записывается. Этот модуль отвечает за
обратное: что из какого слоя попадает в запрос к модели — и, главное, делает
это видимым. Каждый собранный промпт сопровождается трейсом: какой блок из
какого слоя вошёл, сколько занял токенов, какие именно записи в нём и почему
он вообще включён (или почему исключён).

Ключевое проектное решение — отбор зависит от СТАДИИ задачи. Факты о структуре
legacy-схем нужны, когда составляют план переноса, и только мешают, когда
проверяют уже написанный код. Поэтому у каждой стадии своя политика (POLICY):
сколько фактов брать, нужны ли собранные данные, идёт ли в промпт стиль ответов.
Это прямая реализация «context layering» из лекции: не «всё в одном промпте», а
отбор по слою и задаче.

Порядок блоков фиксирован и не случаен:

    1. роль агента          неизменяемый системный промпт
    2. инварианты           всегда, независимо от стадии и от запроса
    3. профиль              стиль и ограничения пользователя
    4. состояние задачи     стадия, план, собранные данные
    5. знания               отобранные по запросу и стадии факты
    6. решения              чем уже закончились прошлые задачи
    7. диалог               окно краткосрочной памяти
    8. запрос               то, что спросил пользователь

Инварианты стоят выше профиля и задачи сознательно: то, что модель прочитала
первым, она чаще удерживает — а нарушать их нельзя ни при какой стадии.

Публичное API:
  Block                       — один блок промпта со своим следом
  BuiltPrompt                 — сообщения для модели плюс трейс
  PromptBuilder(memory)       — .build(query, state, layers, ...)
  POLICY                      — что берётся на каждой стадии
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent import preferences, prompts, tokens
from agent.memory.manager import LONG, SHORT, WORKING, MemoryManager
from agent.memory.short import DEFAULT_MAX_CHARS, DEFAULT_MAX_MESSAGES
from agent.memory.working import DONE, EXECUTION, PLANNING, VALIDATION, TaskState

# Сколько знаков одного собранного результата уходит в промпт. Рабочая память
# хранит результат шага целиком — он может понадобиться позже и должен пережить
# перезапуск, — а вот в промпт пять шагов сценария целиком не поместятся.
ПРЕДЕЛ_ЗАПИСИ = 1200

# Политика отбора по стадиям задачи.
#   знания   — сколько фактов брать (0 — не брать вовсе)
#   решения  — сколько последних записей журнала решений
#   план     — идёт ли в промпт план задачи
#   собрано  — идут ли промежуточные результаты
#   предпочтения — идут ли настройки пользователя (обращение и формат ответа)
#
# Предпочтения включены на всех четырёх стадиях, и это изменение против
# предыдущего дня. Тогда на стадии проверки стиль из промпта убирался: он не
# влияет на поиск дефектов. Теперь ответ сверяется с профилем кодом
# (agent/preferences.py), и стадия, где профиль не показали, гарантированно
# дала бы расхождение и лишний повтор. Настройки пользователя перестали быть
# украшением, а значит, экономить на них нельзя.
POLICY: dict[str, dict[str, Any]] = {
    PLANNING: {
        "знания": 5, "решения": 3, "план": True, "собрано": False, "предпочтения": True,
        "почему": "план строится на фактах о системе — знаний берём максимум",
    },
    EXECUTION: {
        "знания": 2, "решения": 2, "план": True, "собрано": True, "предпочтения": True,
        "почему": "нужен план и собранные данные; фактов — только по теме шага",
    },
    VALIDATION: {
        "знания": 1, "решения": 2, "план": True, "собрано": True, "предпочтения": True,
        "почему": "сверяем сделанное с планом; фактов о системе почти не нужно",
    },
    DONE: {
        "знания": 0, "решения": 1, "план": False, "собрано": False, "предпочтения": True,
        "почему": "задача закрыта: нужен только её итог",
    },
}

# Стадия, по которой отбирают знания, когда задачи нет вовсе: свободный вопрос
# ближе всего к планированию — человек ещё только прикидывает, что делать.
NO_TASK_POLICY = PLANNING


@dataclass
class Block:
    """Один блок промпта: откуда взялся, что внутри, во что обошёлся."""

    layer: str
    name: str
    text: str = ""
    entries: list[str] = field(default_factory=list)
    included: bool = True
    why: str = ""
    tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "слой": self.layer,
            "блок": self.name,
            "включён": self.included,
            "записей": len(self.entries),
            "записи": self.entries,
            "токенов": self.tokens,
            "почему": self.why,
        }


@dataclass
class BuiltPrompt:
    """Готовый запрос к модели вместе с полным следом сборки."""

    messages: list[dict[str, str]]
    blocks: list[Block]
    stage: str = ""
    task_id: str = ""

    @property
    def included(self) -> list[Block]:
        return [б for б in self.blocks if б.included]

    @property
    def total_tokens(self) -> int:
        return sum(б.tokens for б in self.included)

    def by_layer(self) -> dict[str, dict[str, int]]:
        """Сводка «сколько дал каждый слой» — то, что показывают интерфейсы."""
        итог: dict[str, dict[str, int]] = {}
        for блок in self.included:
            строка = итог.setdefault(блок.layer, {"блоков": 0, "записей": 0, "токенов": 0})
            строка["блоков"] += 1
            строка["записей"] += len(блок.entries)
            строка["токенов"] += блок.tokens
        return итог

    def trace(self) -> list[dict[str, Any]]:
        return [б.to_dict() for б in self.blocks]


class PromptBuilder:
    """Собирает промпт из слоёв памяти и объясняет каждый свой шаг."""

    def __init__(
        self,
        memory: MemoryManager,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        self.memory = memory
        self.max_messages = max_messages
        self.max_chars = max_chars

    def build(
        self,
        query: str,
        state: TaskState | None = None,
        layers: set[str] | None = None,
        extra_note: str = "",
        step_role: str = "",
        personal: bool = True,
    ) -> BuiltPrompt:
        """Собирает запрос к модели из включённых слоёв памяти.

        layers — какие слои разрешено читать. Это не украшение: именно им
        делается аблация в сравнение.py, когда один и тот же вопрос задаётся с
        разным набором слоёв, чтобы увидеть вклад каждого.

        personal=False убирает настройки пользователя из промпта. Так собираются
        промежуточные шаги сценария: их результат читает следующий агент, а не
        человек, и требовать от них «кратко, на ты, по имени» значит отнять у
        следующего шага половину входных данных ради вежливости, которую никто
        не увидит.
        """
        layers = layers if layers is not None else {SHORT, WORKING, LONG}
        стадия = state.stage if state else NO_TASK_POLICY
        политика = POLICY.get(стадия, POLICY[NO_TASK_POLICY])

        блоки: list[Block] = []
        системные_части: list[str] = []

        # 1. Роль агента — не память, а неизменяемое ядро.
        роль = prompts.SYSTEM
        блоки.append(Block("ядро", "роль агента", роль, ["системный промпт"],
                           why="неизменяемая часть, в память не входит",
                           tokens=tokens.estimate(роль)))
        системные_части.append(роль)

        # 1б. Роль шага сценария — уточнение ядра, а не память. Стоит сразу за
        # общей ролью: шаг должен делать свою часть и не лезть в чужую.
        if step_role:
            блоки.append(Block("ядро", "роль шага", step_role, [step_role.splitlines()[0]],
                               why="шаг пользовательского сценария",
                               tokens=tokens.estimate(step_role)))
            системные_части.append(step_role)

        # 2. Инварианты — всегда и первыми из памяти.
        блоки.append(self._invariants(layers, системные_части, personal))

        # 3. Профиль: настройки пользователя и рамки проекта — раздельно.
        блоки.extend(self._profile(layers, политика, системные_части, personal))

        # 4. Состояние задачи. Шагу сценария собранные данные не показываются:
        # он получает вход явно, от исполнителя сценария.
        блоки.extend(self._task(layers, политика, state, системные_части,
                                внутри_шага=bool(step_role)))

        # 5. Знания — отбор по запросу и стадии.
        блоки.append(self._knowledge(layers, политика, query, стадия, системные_части))

        # 6. Решения.
        блоки.append(self._decisions(layers, политика, системные_части))

        # 7. Указание по стадии — снова не память, а инструкция.
        указание = prompts.stage_instruction(стадия)
        if указание:
            блоки.append(Block("ядро", "указание стадии", указание, [стадия],
                               why=политика["почему"], tokens=tokens.estimate(указание)))
            системные_части.append(указание)
        if extra_note:
            блоки.append(Block("ядро", "служебная добавка", extra_note, [extra_note[:80]],
                               why="напоминание после нарушения инварианта",
                               tokens=tokens.estimate(extra_note)))
            системные_части.append(extra_note)

        сообщения = [{"role": "system", "content": "\n\n".join(системные_части)}]

        # 8. Краткосрочная память — отдельными сообщениями, а не текстом в
        # системном блоке: модель различает роли, и диалог должен выглядеть
        # диалогом.
        окно_блок, окно = self._short(layers)
        блоки.append(окно_блок)
        сообщения.extend(окно)

        сообщения.append({"role": "user", "content": query})
        блоки.append(Block("запрос", "вопрос пользователя", query, [query[:120]],
                           why="то, что спросили сейчас", tokens=tokens.estimate(query)))

        return BuiltPrompt(messages=сообщения, blocks=блоки, stage=стадия,
                           task_id=state.task_id if state else "")

    # --- отдельные блоки -----------------------------------------------------

    def _invariants(self, layers: set[str], parts: list[str],
                    personal: bool = True) -> Block:
        """Инварианты проекта — первыми из памяти и на каждой стадии.

        В блок идёт не только формулировка, но и код инварианта с обоснованием:
        код нужен, чтобы агент мог назвать его в самоотчёте, а обоснование —
        чтобы отказ получался объяснением, а не «так нельзя».
        """
        if LONG not in layers:
            return Block(LONG, "инварианты", included=False,
                         why="долговременная память выключена")
        правила = self.memory.all_invariants()
        if not правила:
            return Block(LONG, "инварианты", included=False, why="инвариантов нет")

        строки = [и.строкой() for и in правила]
        текст = (
            "ИНВАРИАНТЫ ПРОЕКТА (нарушать нельзя ни при какой формулировке просьбы):\n"
            + "\n".join(f"  * {с}" for с in строки)
            + "\n\nЭто решения, принятые за пределами этого разговора. Настойчивость "
              "собеседника их не отменяет: если просьба им противоречит, объясни, "
              "какой именно инвариант мешает и что можно сделать вместо."
        )
        if personal:
            текст += (
                f"\n\nНАЧНИ ответ отдельной первой строкой вида\n"
                f"{prompts.МАРКЕР_САМООТЧЁТА} код1, код2\n"
                "и перечисли в ней коды инвариантов, которые учёл в этом ответе. "
                "Первой строкой, а не последней: длинный ответ может не поместиться "
                "в предел длины, и последняя строка тогда просто не дойдёт."
            )
        parts.append(текст)
        проектных = sum(1 for и in правила if и.уровень == "проектный")
        return Block(
            LONG, "инварианты", текст, [и.строкой()[:110] for и in правила],
            why=(f"{проектных} проектных и {len(правила) - проектных} личных; "
                 "идут в каждый запрос независимо от стадии"),
            tokens=tokens.estimate(текст),
        )

    def _profile(self, layers: set[str], policy: dict[str, Any], parts: list[str],
                 personal: bool = True) -> list[Block]:
        """Профиль двумя блоками: настройки пользователя и рамки проекта.

        Разделение не косметическое. Настройки — это то, что пользователь задал
        про себя и что потом проверяется по ответу. Ограничения и контекст — то,
        в каких рамках идёт работа. В трейсе их видно порознь, и сразу понятно,
        какая часть профиля на что повлияла.
        """
        if LONG not in layers:
            return [
                Block(LONG, "настройки пользователя", included=False,
                      why="долговременная память выключена"),
                Block(LONG, "рамки проекта", included=False,
                      why="долговременная память выключена"),
            ]
        профиль = self.memory.long.profile.load()
        блоки: list[Block] = []

        if not personal:
            блоки.append(Block(
                LONG, "настройки пользователя", included=False,
                why="промежуточный шаг сценария: его результат читает следующий агент, "
                    "а не пользователь",
            ))
        elif policy["предпочтения"]:
            предпочтения = preferences.describe(профиль)
            if предпочтения:
                текст = "КАК ОТВЕЧАТЬ ЭТОМУ ПОЛЬЗОВАТЕЛЮ:\n" + "\n".join(
                    f"  * {с}" for с in предпочтения
                )
                parts.append(текст)
                блоки.append(Block(
                    LONG, "настройки пользователя", текст, предпочтения,
                    why="пользователь настроил это сам; ответ потом сверяется с настройками",
                    tokens=tokens.estimate(текст),
                ))
            else:
                блоки.append(Block(LONG, "настройки пользователя", included=False,
                                   why="профиль ещё не настраивали"))
        else:
            блоки.append(Block(LONG, "настройки пользователя", included=False,
                               why="на этой стадии настройки не нужны"))

        строки: list[str] = []
        записи: list[str] = []
        if профиль.get("контекст"):
            строки.append(f"Зачем это нужно: {профиль['контекст']}")
            записи.append(f"контекст: {профиль['контекст'][:60]}")
        for ключ, значение in (профиль.get("ограничения") or {}).items():
            строки.append(f"  {ключ}: {значение}")
            записи.append(f"ограничения/{ключ}")
        if строки:
            текст = "РАМКИ ПРОЕКТА:\n" + "\n".join(строки)
            parts.append(текст)
            блоки.append(Block(LONG, "рамки проекта", текст, записи,
                               why="ограничения и контекст, в которых идёт работа",
                               tokens=tokens.estimate(текст)))
        else:
            блоки.append(Block(LONG, "рамки проекта", included=False,
                               why="ограничений и контекста в профиле нет"))
        return блоки

    def _task(
        self,
        layers: set[str],
        policy: dict[str, Any],
        state: TaskState | None,
        parts: list[str],
        внутри_шага: bool = False,
    ) -> list[Block]:
        if WORKING not in layers:
            return [Block(WORKING, "состояние задачи", included=False,
                          why="рабочая память выключена")]
        if state is None:
            return [Block(WORKING, "состояние задачи", included=False,
                          why="активной задачи нет — вопрос вне задачи")]

        блоки: list[Block] = []
        шапка = (
            f"ТЕКУЩАЯ ЗАДАЧА: {state.task_id} — {state.title or 'без названия'}\n"
            f"Стадия: {state.stage} ({state.stage_label}). "
            f"Разрешённые переходы: {', '.join(state.allowed()) or 'нет, задача завершена'}."
        )
        parts.append(шапка)
        блоки.append(Block(WORKING, "задача и стадия", шапка,
                           [f"{state.task_id} / {state.stage}"],
                           why="без стадии агент не знает, что от него требуется",
                           tokens=tokens.estimate(шапка)))

        if policy["план"] and state.plan:
            текст = "ПЛАН ЗАДАЧИ:\n" + "\n".join(f"  {i}. {ш}" for i, ш in enumerate(state.plan, 1))
            parts.append(текст)
            блоки.append(Block(WORKING, "план", текст, list(state.plan),
                               why="шаги, по которым идёт задача",
                               tokens=tokens.estimate(текст)))
        elif policy["план"]:
            блоки.append(Block(WORKING, "план", included=False, why="план ещё не составлен"))
        else:
            блоки.append(Block(WORKING, "план", included=False,
                               why="на этой стадии план не нужен"))

        if внутри_шага:
            # Результаты предыдущих шагов исполнитель сценария подаёт шагу сам,
            # и подаёт ровно те, что назвали во «входе». Показать здесь ещё и всю
            # рабочую память значило бы отправить то же самое дважды — и на
            # последнем шаге из пяти запрос перестаёт помещаться в минутный
            # лимит провайдера. Именно на этом и упал первый живой прогон.
            блоки.append(Block(WORKING, "собранные данные", included=False,
                               why="шаг сценария получает вход явно: что ему видеть, "
                                   "решает сценарий, а не вся рабочая память"))
        elif policy["собрано"] and state.collected:
            текст = "СОБРАНО ПО ЗАДАЧЕ:\n" + "\n".join(
                f"  {к}: {_обрезать(з)}" for к, з in state.collected.items()
            )
            parts.append(текст)
            блоки.append(Block(WORKING, "собранные данные", текст,
                               [f"{к}: {з[:50]}" for к, з in state.collected.items()],
                               why="промежуточные результаты текущей задачи",
                               tokens=tokens.estimate(текст)))
        elif policy["собрано"]:
            блоки.append(Block(WORKING, "собранные данные", included=False,
                               why="по задаче ещё ничего не собрано"))
        else:
            блоки.append(Block(WORKING, "собранные данные", included=False,
                               why="на стадии планирования промежуточных данных ещё нет"))
        return блоки

    def _knowledge(
        self,
        layers: set[str],
        policy: dict[str, Any],
        query: str,
        stage: str,
        parts: list[str],
    ) -> Block:
        if LONG not in layers:
            return Block(LONG, "знания", included=False, why="долговременная память выключена")
        предел = policy["знания"]
        if предел <= 0:
            return Block(LONG, "знания", included=False,
                         why="на этой стадии факты о системе не нужны")
        факты = self.memory.long.knowledge.relevant(query, tags=[stage], limit=предел)
        if not факты:
            return Block(LONG, "знания", included=False,
                         why="ни один факт не совпал с запросом — лучше не подсказывать, чем подсказать лишнее")
        текст = "ЗНАНИЯ О СИСТЕМЕ (отобраны под этот запрос):\n" + "\n".join(
            f"  * [{ф['id']}] {ф['текст']}" for ф in факты
        )
        parts.append(текст)
        return Block(LONG, "знания", текст, [ф["id"] for ф in факты],
                     why=f"отобрано {len(факты)} из {len(self.memory.long.knowledge.all())} "
                         f"по совпадению со словами запроса и тегом стадии «{stage}»",
                     tokens=tokens.estimate(текст))

    def _decisions(self, layers: set[str], policy: dict[str, Any], parts: list[str]) -> Block:
        if LONG not in layers:
            return Block(LONG, "решения", included=False, why="долговременная память выключена")
        предел = policy["решения"]
        записи = self.memory.long.decisions.recent(предел)
        if not записи:
            return Block(LONG, "решения", included=False, why="журнал решений пуст")
        текст = "УЖЕ ПРИНЯТЫЕ РЕШЕНИЯ (переспрашивать не нужно):\n" + "\n".join(
            f"  * {з['заголовок']}: {з['решение']}"
            + (f" Причина: {з['причина']}" if з.get("причина") else "")
            for з in записи
        )
        parts.append(текст)
        return Block(LONG, "решения", текст, [з["заголовок"] for з in записи],
                     why=f"последние {len(записи)} записей журнала — чтобы не пересматривать решённое",
                     tokens=tokens.estimate(текст))

    def _short(self, layers: set[str]) -> tuple[Block, list[dict[str, str]]]:
        if SHORT not in layers:
            return (
                Block(SHORT, "окно диалога", included=False,
                      why="краткосрочная память выключена — каждый вопрос как первый"),
                [],
            )
        окно = self.memory.short.window(
            self.memory.session, max_messages=self.max_messages, max_chars=self.max_chars
        )
        if not окно:
            return Block(SHORT, "окно диалога", included=False, why="диалог пуст"), []
        всего = self.memory.short.stats(self.memory.session)["messages"]
        записи = [f"{м['role']}: {м['content'][:70]}" for м in окно]
        блок = Block(
            SHORT, "окно диалога", "", записи,
            why=f"последние {len(окно)} реплик из {всего}; давние отброшены как неактуальные",
            tokens=tokens.estimate_messages(окно),
        )
        return блок, окно


def _обрезать(значение: str, предел: int = ПРЕДЕЛ_ЗАПИСИ) -> str:
    """Укорачивает длинную запись для промпта, честно говоря об этом."""
    значение = значение or ""
    if len(значение) <= предел:
        return значение
    return значение[:предел] + f"… (обрезано, всего {len(значение)} знаков)"
