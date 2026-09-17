"""Три слоя памяти агента — три отдельных модуля и три отдельных хранилища.

    short.py    краткосрочная   реплики текущего диалога        SQLite
    working.py  рабочая         состояние текущей задачи        JSON на задачу
    long.py     долговременная  профиль, решения, знания        три файла

Плюс два модуля, которые сами ничего не хранят, но решают, что куда положить:

    router.py   дешёвая модель предлагает слой для свободной реплики
    manager.py  единственная точка записи; правила маршрутизации и их журнал

Разделение проведено не по удобству кода, а по сроку жизни данных: реплика
живёт до конца разговора, состояние задачи — до конца задачи, профиль и знания —
пока проект не закончится. Смешивать их в одном хранилище значит либо стирать
нужное вместе с ненужным, либо носить в каждом промпте то, что давно неактуально.
"""

from agent.memory.long import (
    DecisionsStore, KnowledgeStore, LongTermError, LongTermMemory, ProfileStore,
)
from agent.memory.manager import ASK, AUTO, LAYERS, LONG, OFF, SHORT, WORKING, MemoryManager
from agent.memory.router import Router, Routing
from agent.memory.short import ShortTermMemory, ShortTermError
from agent.memory.working import (
    DONE, EXECUTION, PLANNING, STAGES, STAGE_LABELS, TRANSITIONS, VALIDATION,
    ЖДЁТ, ГОТОВ, ИДЁТ, ИЗ_ПЛАНА, ИЗ_СЦЕНАРИЯ, НАРУШЕН_ИНВАРИАНТ, НА_ПЕРЕХОДЕ,
    НЕТ_СВЕДЕНИЙ, НИЧЕГО, ОЖИДАНИЯ, ОЖИДАНИЯ_СЛОВАМИ, ОТВЕТ, ПОДТВЕРДИТЬ,
    ПО_КОМАНДЕ, ПРИЧИНЫ_ПАУЗЫ, ПРОДОЛЖИТЬ, РЕШЕНИЕ, ЗАПУСТИТЬ,
    TaskState, TaskStep, TransitionError, WorkingMemory, WorkingMemoryError,
)

__all__ = [
    "ShortTermMemory", "ShortTermError",
    "WorkingMemory", "WorkingMemoryError", "TaskState", "TransitionError",
    "STAGES", "STAGE_LABELS", "TRANSITIONS", "PLANNING", "EXECUTION", "VALIDATION", "DONE",
    "TaskStep", "ОЖИДАНИЯ", "ОЖИДАНИЯ_СЛОВАМИ", "ПРИЧИНЫ_ПАУЗЫ",
    "ЗАПУСТИТЬ", "ПРОДОЛЖИТЬ", "ПОДТВЕРДИТЬ", "ОТВЕТ", "РЕШЕНИЕ", "НИЧЕГО",
    "ПО_КОМАНДЕ", "НА_ПЕРЕХОДЕ", "НЕТ_СВЕДЕНИЙ", "НАРУШЕН_ИНВАРИАНТ",
    "ИЗ_СЦЕНАРИЯ", "ИЗ_ПЛАНА", "ЖДЁТ", "ИДЁТ", "ГОТОВ",
    "LongTermMemory", "LongTermError", "ProfileStore", "DecisionsStore", "KnowledgeStore",
    "MemoryManager", "Router", "Routing",
    "LAYERS", "SHORT", "WORKING", "LONG", "AUTO", "ASK", "OFF",
]
