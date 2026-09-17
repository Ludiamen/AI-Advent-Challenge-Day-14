"""Сценарии: пользователь сам задаёт, кто и в каком порядке делает его работу.

Это вторая половина персонализации. Профиль отвечает на вопрос «как со мной
разговаривать», сценарий — на вопрос «что делать, когда я прошу вот это».
Пользователь описывает: «на запрос "напиши фичу" сначала пусть аналитик соберёт
требования, потом архитектор спроектирует, потом backend напишет код, потом
ревьюер проверит» — и агент делает именно это, а не отвечает одним махом.

Каждый шаг — отдельный вызов модели со своей ролью, своей моделью из каталога и
своей стадией задачи. Результат шага ложится в рабочую память и становится
входом следующего, поэтому сценарий не «промпт с перечислением ролей», а
настоящая последовательность: архитектор видит, что написал аналитик, и ничего
больше.

Сценарии лежат отдельным файлом рядом с профилем, а не внутри него. Причина
практическая: профиль целиком уходит в каждый запрос, а сценарий — нет. В промпт
идёт только описание текущего шага, всё остальное — исполняемая конфигурация, и
раздувать ею блок «профиль» значит платить токенами за то, что модели не нужно.

Публичное API:
  Step, Scenario               — шаг и сценарий
  ScenarioStore(path)          — хранилище сценариев пользователя
  .all() / .get(name) / .match(query) / .add(scenario) / .remove(name)
  ScenarioRunner(agent)        — исполнение сценария шаг за шагом
  StepResult, RunResult        — что получилось
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from agent import catalog
from agent.memory.working import (
    DONE, ИЗ_СЦЕНАРИЯ, НАРУШЕН_ИНВАРИАНТ, НА_ПЕРЕХОДЕ, НЕТ_СВЕДЕНИЙ, НИЧЕГО,
    ОТВЕТ, ПОДТВЕРДИТЬ, ПРОДОЛЖИТЬ, РЕШЕНИЕ, STAGES, TaskStep,
)

if TYPE_CHECKING:  # только для подсказок типов: импорт агента сюда внёс бы цикл
    from agent.agent import Answer, MemoryAgent

# Сколько знаков результата предыдущего шага уходит на вход следующему. Предел
# нужен: пять шагов по три тысячи знаков не оставят места под сам вопрос.
ВХОД_ШАГА = 2500

# Как шаг сообщает, что дальше идти нечем. Маркер разбирается кодом, а решение
# принимает модель: спросить её «хватает ли сведений?» отдельным вызовом дороже
# и менее надёжно, чем попросить закончить ответ условленной строкой. Требование
# ставится только тем шагам, которым это разрешено (может_спросить).
МАРКЕР_ВОПРОСА = "НУЖНЫ СВЕДЕНИЯ:"

УКАЗАНИЕ_ПРО_ВОПРОС = (
    "Если для честного выполнения шага не хватает сведений, которых нет ни в "
    "запросе, ни в результатах предыдущих шагов, ни в знаниях о системе, — "
    "не додумывай. Закончи ответ отдельной последней строкой вида\n"
    f"{МАРКЕР_ВОПРОСА} <перечисли одним предложением, что именно нужно узнать>\n"
    "Если сведений хватает, этой строки быть не должно."
)

_ПРОБЕЛЫ = re.compile(r"\s+")
_НЕ_СЛОВО = re.compile(r"[^a-zA-Zа-яёА-ЯЁ0-9]+")


class ScenarioError(RuntimeError):
    """Ошибка описания или исполнения сценария."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slug(text: str) -> str:
    """Идентификатор задачи из имени сценария: «напиши фичу» -> «напиши-фичу»."""
    очищено = _НЕ_СЛОВО.sub("-", (text or "").strip().lower()).strip("-")
    return (очищено or "сценарий")[:40]


@dataclass
class Step:
    """Один шаг сценария: кто, на какой модели, на какой стадии и зачем."""

    агент: str                          # как этот шаг зовут: аналитик, backend, ревьюер
    задача: str                         # что он должен сделать
    роль: str = "исполнение"            # роль модели из catalog.ROLES
    стадия: str = "execution"           # стадия задачи, на которой идёт шаг
    вход: list[str] = field(default_factory=lambda: ["запрос"])
    описание: str = ""                  # кто он такой — идёт в системный промпт шага
    # Разрешено ли шагу остановить сценарий и спросить у человека. Имеет смысл
    # для шагов, которые собирают требования: аналитику без ответа дальше идти
    # нечем, а backend-у додумать недостающее проще, чем ждать.
    может_спросить: bool = False

    def validate(self) -> None:
        if not self.агент.strip():
            raise ScenarioError("У шага должно быть имя агента.")
        if not self.задача.strip():
            raise ScenarioError(f"У шага «{self.агент}» не сказано, что делать.")
        if self.роль not in catalog.ROLES:
            raise ScenarioError(
                f"Шаг «{self.агент}»: нет роли модели «{self.роль}». "
                f"Есть: {', '.join(catalog.ROLES)}."
            )
        if self.стадия not in STAGES:
            raise ScenarioError(
                f"Шаг «{self.агент}»: нет стадии «{self.стадия}». "
                f"Есть: {', '.join(STAGES)}."
            )

    @property
    def model_key(self) -> str:
        return catalog.for_role(self.роль, offset=0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Step":
        известные = {поле for поле in cls.__dataclass_fields__}
        return cls(**{к: з for к, з in data.items() if к in известные})


@dataclass
class Scenario:
    """Именованная последовательность шагов с условиями запуска."""

    имя: str
    шаги: list[Step] = field(default_factory=list)
    триггеры: list[str] = field(default_factory=list)
    описание: str = ""
    создан: str = field(default_factory=_now)

    def validate(self) -> None:
        if not self.имя.strip():
            raise ScenarioError("У сценария должно быть имя.")
        if not self.шаги:
            raise ScenarioError(f"В сценарии «{self.имя}» нет ни одного шага.")
        имена = [ш.агент for ш in self.шаги]
        if len(set(имена)) != len(имена):
            raise ScenarioError(
                f"В сценарии «{self.имя}» повторяются имена шагов: результаты "
                "перезаписали бы друг друга."
            )
        for шаг in self.шаги:
            шаг.validate()
        # Стадии должны идти по разрешённым переходам конечного автомата задачи,
        # иначе сценарий сломается на середине — уже потратив токены.
        _проверить_маршрут(self)

    def matches(self, query: str) -> bool:
        текст = _ПРОБЕЛЫ.sub(" ", (query or "").lower())
        for триггер in self.триггеры:
            если = _ПРОБЕЛЫ.sub(" ", триггер.lower().strip())
            if если and если in текст:
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        данные = asdict(self)
        данные["шаги"] = [ш.to_dict() for ш in self.шаги]
        return данные

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        шаги = [Step.from_dict(ш) for ш in data.get("шаги", [])]
        return cls(
            имя=data.get("имя", ""),
            шаги=шаги,
            триггеры=list(data.get("триггеры", [])),
            описание=data.get("описание", ""),
            создан=data.get("создан", _now()),
        )

    def digest(self) -> str:
        цепочка = " → ".join(f"{ш.агент} ({ш.стадия})" for ш in self.шаги)
        return f"{self.имя}: {цепочка}"


def _проверить_маршрут(scenario: Scenario) -> None:
    """Сверяет стадии шагов с разрешёнными переходами задачи."""
    from agent.memory.working import PLANNING, TRANSITIONS

    текущая = PLANNING
    for шаг in scenario.шаги:
        if шаг.стадия == текущая:
            continue
        if шаг.стадия not in TRANSITIONS.get(текущая, ()):
            разрешено = ", ".join(TRANSITIONS.get(текущая, ())) or "никуда"
            raise ScenarioError(
                f"Сценарий «{scenario.имя}»: шаг «{шаг.агент}» просит стадию "
                f"«{шаг.стадия}», а из «{текущая}» можно только в {разрешено}."
            )
        текущая = шаг.стадия


# --- хранилище ----------------------------------------------------------------

class ScenarioStore:
    """Сценарии одного пользователя: отдельный файл рядом с профилем."""

    layer = "долговременная"
    sublayer = "сценарии"

    def __init__(self, path: str) -> None:
        self.path = path

    def all(self) -> list[Scenario]:
        try:
            with open(self.path, encoding="utf-8") as файл:
                данные = json.load(файл)
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError) as exc:
            raise ScenarioError(f"Файл сценариев «{self.path}» испорчен: {exc}") from exc
        if not isinstance(данные, list):
            return []
        return [Scenario.from_dict(с) for с in данные]

    def save_all(self, scenarios: list[Scenario]) -> None:
        временный = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(временный, "w", encoding="utf-8") as файл:
                json.dump([с.to_dict() for с in scenarios], файл,
                          ensure_ascii=False, indent=2)
            os.replace(временный, self.path)
        except OSError as exc:
            raise ScenarioError(f"Не удалось записать сценарии: {exc}") from exc

    def get(self, name: str) -> Scenario | None:
        имя = (name or "").strip().lower()
        for сценарий in self.all():
            if сценарий.имя.lower() == имя:
                return сценарий
        return None

    def match(self, query: str) -> Scenario | None:
        """Первый сценарий, чей триггер нашёлся в запросе.

        Порядок важен: сценарии проверяются в том порядке, в котором лежат в
        файле, и пользователь может переставить их сам. Совпадения не
        взвешиваются — «самый подходящий» здесь было бы гаданием.
        """
        for сценарий in self.all():
            if сценарий.matches(query):
                return сценарий
        return None

    def add(self, scenario: Scenario, overwrite: bool = True) -> Scenario:
        scenario.validate()
        сценарии = [с for с in self.all() if с.имя.lower() != scenario.имя.lower()]
        if len(сценарии) != len(self.all()) and not overwrite:
            raise ScenarioError(f"Сценарий «{scenario.имя}» уже есть.")
        сценарии.append(scenario)
        self.save_all(сценарии)
        return scenario

    def remove(self, name: str) -> bool:
        сценарии = self.all()
        осталось = [с for с in сценарии if с.имя.lower() != (name or "").strip().lower()]
        if len(осталось) == len(сценарии):
            return False
        self.save_all(осталось)
        return True


# --- исполнение ---------------------------------------------------------------

@dataclass
class StepResult:
    """Чем закончился один шаг сценария."""

    агент: str
    стадия: str
    модель: str
    текст: str
    answer: "Answer | None" = None
    токенов: int = 0
    стоимость: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "агент": self.агент,
            "стадия": self.стадия,
            "модель": self.модель,
            "текст": self.текст,
            "токенов": self.токенов,
            "стоимость": self.стоимость,
            "расхождения": [о.to_dict() for о in (self.answer.deviations if self.answer else [])],
            "нарушения": [н.to_dict() for н in (self.answer.violations if self.answer else [])],
        }


@dataclass
class RunResult:
    """Чем закончился сценарий: дошёл до конца или встал на паузу."""

    сценарий: str
    task_id: str
    шаги: list[StepResult] = field(default_factory=list)
    решение: dict[str, Any] | None = None
    на_паузе: bool = False
    причина_паузы: str = ""
    ожидание: str = ""
    ожидание_текст: str = ""
    шаг: int = 0            # на каком шаге стоим
    всего_шагов: int = 0

    @property
    def итог(self) -> str:
        """Ответ последнего шага — его и показывают пользователю."""
        return self.шаги[-1].текст if self.шаги else ""

    @property
    def токенов(self) -> int:
        return sum(ш.токенов for ш in self.шаги)

    @property
    def стоимость(self) -> float:
        return sum(ш.стоимость for ш in self.шаги)

    def to_dict(self) -> dict[str, Any]:
        return {
            "сценарий": self.сценарий,
            "task_id": self.task_id,
            "шаги": [ш.to_dict() for ш in self.шаги],
            "решение": self.решение,
            "токенов": self.токенов,
            "стоимость": self.стоимость,
            "на_паузе": self.на_паузе,
            "причина_паузы": self.причина_паузы,
            "ожидание": self.ожидание,
            "ожидание_текст": self.ожидание_текст,
            "шаг": self.шаг,
            "всего_шагов": self.всего_шагов,
        }


class ScenarioRunner:
    """Прогоняет сценарий по шагам, двигая задачу по стадиям."""

    def __init__(self, agent: "MemoryAgent") -> None:
        self.agent = agent

    def model_for(self, step: Step) -> str:
        """Какая модель на самом деле выполнит шаг.

        Если пользователь выбрал модель явно, она сильнее роли шага. Правило
        живёт здесь одно на всех: интерфейсы показывают ход прогона и обязаны
        называть ту модель, которая работает, а не ту, что записана в сценарии.
        """
        return self.agent.model_key or step.model_key

    def run(
        self,
        scenario: Scenario,
        query: str,
        task_id: str = "",
        on_step: Callable[[Step, int, int], None] | None = None,
        finish: bool = True,
        on_result: Callable[["StepResult", int, int], None] | None = None,
        по_шагам: bool = False,
    ) -> RunResult:
        """Заводит задачу по сценарию и исполняет её, пока не кончится или не встанет.

        on_step вызывается перед каждым шагом, on_result — после. Оба нужны
        интерфейсам: пять шагов идут минуту-две, и отдавать всё одним куском в
        конце значит заставить человека смотреть в неподвижный экран.

        по_шагам=True — режим с подтверждением: перед каждой сменой стадии
        сценарий останавливается и ждёт согласия человека.
        """
        scenario.validate()
        task_id = task_id or f"{slug(scenario.имя)}-{datetime.now().strftime('%m%d-%H%M')}"
        состояние = self.agent.start_task(task_id, scenario.имя, overwrite=True)

        # Шаги переезжают из описания сценария в состояние задачи. С этого
        # момента «где мы стоим» живёт на диске, а не в переменной исполнителя, —
        # иначе продолжить отложенную задачу в другом процессе нечем.
        состояние.set_steps(
            [
                TaskStep(номер=н, имя=ш.агент, источник=ИЗ_СЦЕНАРИЯ,
                         стадия=ш.стадия, может_спросить=ш.может_спросить)
                for н, ш in enumerate(scenario.шаги, 1)
            ],
            сценарий=scenario.имя,
        )
        состояние.запрос = query
        состояние.ждать(ПРОДОЛЖИТЬ)
        self.agent.save_task()

        # В диалог сценарий попадает как один обмен: вопрос человека и тот
        # ответ, который человек прочитает. Промежуточные шаги туда не идут —
        # это переписка агентов между собой.
        self.agent.remember_reply("user", query)
        return self._прогнать(scenario, состояние, on_step, on_result, finish, по_шагам)

    def resume(
        self,
        task_id: str,
        ответ: str = "",
        on_step: Callable[[Step, int, int], None] | None = None,
        on_result: Callable[["StepResult", int, int], None] | None = None,
        finish: bool = True,
        по_шагам: bool = False,
    ) -> RunResult:
        """Продолжает отложенную задачу с того шага, на котором она стоит.

        Ничего не переспрашивает и не пересчитывает: исходный запрос, план,
        результаты пройденных шагов и ответы человека уже лежат в состоянии
        задачи и в рабочей памяти. Именно ради этого состояние и формализовано.
        """
        состояние = self.agent.use_task(task_id)
        if not состояние.сценарий:
            raise ScenarioError(
                f"Задача «{task_id}» велась не по сценарию — продолжать нечего."
            )
        сценарий = self.agent.memory.long.scenarios.get(состояние.сценарий)
        if сценарий is None:
            raise ScenarioError(
                f"Сценарий «{состояние.сценарий}», по которому шла задача, больше не заведён."
            )
        # Что именно подтверждают, надо запомнить ДО снятия паузы: «продолжить»
        # стирает причину, и цикл, увидев несменённую стадию, снова попросит
        # подтвердить тот же переход — и так до бесконечности. На этом и
        # поймали: подтверждение возвращало ровно ту же просьбу подтвердить.
        подтверждён_переход = состояние.причина_паузы == НА_ПЕРЕХОДЕ
        if ответ.strip():
            состояние.ответить(ответ)
        else:
            состояние.продолжить()
        self.agent.save_task()
        return self._прогнать(сценарий, состояние, on_step, on_result, finish,
                              по_шагам, подтверждён_переход)

    # --- общий цикл ----------------------------------------------------------

    def _прогнать(
        self,
        scenario: Scenario,
        состояние,
        on_step,
        on_result,
        finish: bool,
        по_шагам: bool,
        подтверждён_переход: bool = False,
    ) -> RunResult:
        """Один цикл исполнения: от текущего шага до конца или до остановки.

        подтверждён_переход — человек уже согласился на ближайшую смену стадии,
        и переспрашивать о ней нельзя.
        """
        итог = RunResult(сценарий=scenario.имя, task_id=состояние.task_id,
                         всего_шагов=len(scenario.шаги))
        результаты = self._прошлые_результаты(состояние)

        while состояние.текущий_шаг <= len(scenario.шаги):
            номер = состояние.текущий_шаг
            шаг = scenario.шаги[номер - 1]

            # Точка останова 1: пауза по команде. Проверяется перед шагом, а не
            # посреди него: ответ модели, за который уже заплачено, выбрасывать
            # глупо, поэтому текущий шаг всегда доводится до конца.
            if состояние.пауза:
                return self._встали(итог, состояние)

            # Точка останова 2: смена стадии в режиме «по шагам».
            if состояние.stage != шаг.стадия:
                if по_шагам and not подтверждён_переход:
                    состояние.остановить(
                        НА_ПЕРЕХОДЕ, ПОДТВЕРДИТЬ,
                        f"перейти со стадии «{состояние.stage}» на «{шаг.стадия}» "
                        f"и запустить шаг «{шаг.агент}»?",
                    )
                    self.agent.save_task()
                    return self._встали(итог, состояние)
                self.agent.transition(шаг.стадия, f"шаг «{шаг.агент}» сценария")
                # Согласие действует ровно на один переход: следующую смену
                # стадии в режиме «по шагам» снова нужно подтвердить.
                подтверждён_переход = False

            if on_step is not None:
                on_step(шаг, номер, len(scenario.шаги))
            состояние.начать_шаг()
            self.agent.save_task()

            последний = номер == len(scenario.шаги)
            ответ = self.agent.ask(
                self._вход(шаг, состояние.запрос, результаты, состояние.ответы),
                model_key=self.model_for(шаг),
                step_role=self._роль(шаг),
                personal=последний,
                internal=True,
            )
            результаты[шаг.агент] = ответ.text
            self.agent.remember_step(f"шаг «{шаг.агент}»", ответ.text)
            состояние.закончить_шаг(ответ.text)

            результат = StepResult(
                агент=шаг.агент, стадия=шаг.стадия,
                модель=ответ.usage.get("model_key") or шаг.model_key,
                текст=ответ.text, answer=ответ,
                токенов=ответ.usage.get("total_tokens", 0),
                стоимость=ответ.usage.get("cost", 0.0),
            )
            итог.шаги.append(результат)
            if on_result is not None:
                on_result(результат, номер, len(scenario.шаги))

            # Точка останова 3: ответ не уложился в инварианты за все попытки.
            if ответ.blocked:
                состояние.остановить(
                    НАРУШЕН_ИНВАРИАНТ, РЕШЕНИЕ,
                    "шаг «%s» так и не уложился в ограничения проекта: %s"
                    % (шаг.агент, "; ".join(str(н) for н in ответ.violations)),
                )
                self.agent.save_task()
                return self._встали(итог, состояние)

            # Точка останова 4: шагу не хватило сведений и он спросил сам.
            вопрос = вопрос_шага(ответ.text) if шаг.может_спросить else ""
            if вопрос:
                # Указатель к этому моменту уже стоит на следующем шаге, поэтому
                # в пояснении называется тот, кто спросил: иначе выходит
                # «стоим на backend, ждём ответ про SRID» — и непонятно, кому
                # этот SRID понадобился.
                состояние.остановить(
                    НЕТ_СВЕДЕНИЙ, ОТВЕТ, f"шаг «{шаг.агент}» спрашивает: {вопрос}"
                )
                self.agent.save_task()
                return self._встали(итог, состояние)

            self.agent.save_task()

        if finish and self.agent.task is not None:
            if on_step is not None:
                # Свёртка задачи в журнал решений — отдельный вызов модели, и он
                # заметен: на длинном сценарии это десятки секунд после
                # последнего шага.
                on_step(Step(агент="свёртка итога в журнал решений",
                             задача="сжать задачу в решение",
                             роль="сжатие", стадия="done"),
                        len(scenario.шаги) + 1, len(scenario.шаги) + 1)
            if состояние.stage != DONE and not состояние.can_go(DONE):
                self.agent.transition("validation", "сценарий завершён")
            итог.решение = self.agent.finish_task(f"сценарий «{scenario.имя}» отработал")

        if итог.шаги:
            self.agent.remember_reply("assistant", итог.шаги[-1].текст)
        итог.ожидание = НИЧЕГО
        итог.шаг = len(scenario.шаги)
        return итог

    @staticmethod
    def _встали(итог: RunResult, состояние) -> RunResult:
        """Заполняет результат данными остановки — их и показывают интерфейсы."""
        итог.на_паузе = True
        итог.причина_паузы = состояние.причина_паузы
        итог.ожидание = состояние.ожидание
        итог.ожидание_текст = состояние.ожидание_текст
        итог.шаг = состояние.текущий_шаг
        return итог

    @staticmethod
    def _прошлые_результаты(состояние) -> dict[str, str]:
        """Достаёт результаты уже пройденных шагов из рабочей памяти задачи.

        Это и есть «продолжение без повторных объяснений»: заново считать ничего
        не нужно, всё написанное предыдущими шагами лежит в собранных данных.
        """
        итог: dict[str, str] = {}
        for ключ, значение in состояние.collected.items():
            найдено = re.match(r'^шаг «(.+)»$', ключ)
            if найдено:
                итог[найдено.group(1)] = значение
        return итог

    @staticmethod
    def _роль(step: Step) -> str:
        """Системная приписка шага: кто сейчас отвечает и чем он занят."""
        кто = step.описание or f"Ты — {step.агент} в команде миграции."
        текст = (
            f"РОЛЬ НА ЭТОМ ШАГЕ: {step.агент}.\n{кто}\n"
            f"Твоя задача сейчас: {step.задача}\n"
            "Делай только её. Работу следующих шагов не делай — её сделают другие."
        )
        if step.может_спросить:
            текст += "\n\n" + УКАЗАНИЕ_ПРО_ВОПРОС
        return текст

    @staticmethod
    def _вход(step: Step, query: str, results: dict[str, str],
              ответы: list[dict[str, str]] | None = None) -> str:
        """Собирает вход шага: исходный запрос, результаты названных шагов и ответы.

        Ответы человека на остановках идут всем шагам без разбора: если человек
        что-то уточнил, это уточнение действует до конца задачи, а не только на
        том шаге, который спросил.
        """
        части = []
        for источник in step.вход:
            if источник == "запрос":
                части.append(f"ИСХОДНЫЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n{query}")
            elif источник in results:
                кусок = results[источник][:ВХОД_ШАГА]
                хвост = "\n…(дальше обрезано)" if len(results[источник]) > ВХОД_ШАГА else ""
                части.append(f"РЕЗУЛЬТАТ ШАГА «{источник}»:\n{кусок}{хвост}")
            # Шага с таким именем ещё не было — молча пропускаем: это не ошибка
            # конфигурации, а нормальная ситуация для необязательного входа.
        if not части:
            части.append(f"ИСХОДНЫЙ ЗАПРОС ПОЛЬЗОВАТЕЛЯ:\n{query}")
        for запись in (ответы or []):
            части.append(
                f"ОТВЕТ ПОЛЬЗОВАТЕЛЯ на вопрос шага «{запись.get('шаг', '')}»:\n"
                f"{запись.get('ответ', '')}"
            )
        части.append(f"ЧТО НУЖНО СДЕЛАТЬ СЕЙЧАС: {step.задача}")
        return "\n\n".join(части)


def вопрос_шага(текст: str) -> str:
    """Достаёт вопрос шага, если тот сообщил, что сведений не хватает.

    Маркер ищется в последних строках: модель ставит его в конце, а найденный в
    середине ответа он означал бы пересказ инструкции, а не настоящий вопрос.
    """
    строки = [с.strip() for с in (текст or "").strip().split("\n") if с.strip()]
    for строка in reversed(строки[-4:]):
        without = строка.lstrip("*_# ").strip()
        if without.upper().startswith(МАРКЕР_ВОПРОСА):
            вопрос = without[len(МАРКЕР_ВОПРОСА):].strip(" *_")
            return вопрос or "шаг сообщил, что сведений не хватает, но не уточнил каких"
    return ""
