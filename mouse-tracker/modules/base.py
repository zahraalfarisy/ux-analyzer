import time
import uuid
from dataclasses import dataclass, asdict, field


def new_session_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Event:
    type:       str
    ts:         float
    session_id: str
    x:          float = 0.0
    y:          float = 0.0
    meta:       dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def make(cls, type_: str, session_id: str, x=0.0, y=0.0, **meta) -> "Event":
        return cls(type=type_, ts=time.time(), session_id=session_id,
                   x=x, y=y, meta=meta)