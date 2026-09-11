"""
Mouse Tracker Module
Merekam mouse events dan mendeteksi behavior anomali secara real-time.
Output: list of Event dicts yang siap di-fuse dengan sinyal lain.
"""
import math
import time
from collections import deque
from threading import Thread
from typing import Callable, Optional

from pynput import mouse

from modules.base import Event
from config import (RAGE_CLICK_RADIUS, RAGE_CLICK_WINDOW, RAGE_CLICK_MIN,
                    HESITATION_PAUSE, ERRATIC_SPEED_PX_S)


class MouseTracker:
    def __init__(self, session_id: str, on_event: Optional[Callable] = None):
        """
        session_id  : ID sesi rekaman (dari new_session_id())
        on_event    : callback opsional, dipanggil setiap event baru masuk
                      berguna untuk live fusion tanpa harus tunggu rekaman selesai
        """
        self.session_id = session_id
        self.on_event   = on_event or (lambda e: None)
        self.events: list[dict] = []

        self._click_buf: deque = deque(maxlen=20)
        self._last_move: Optional[dict] = None
        self._last_pos:  Optional[tuple] = None
        self._last_move_ts: float = 0.0
        self._listener: Optional[mouse.Listener] = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self):
        """Mulai merekam di background thread."""
        self._listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        self._listener.start()

    def stop(self):
        if self._listener:
            self._listener.stop()

    # ------------------------------------------------------------------ #
    #  Internal handlers                                                   #
    # ------------------------------------------------------------------ #

    def _emit(self, event: Event):
        d = event.to_dict()
        self.events.append(d)
        self.on_event(d)

    def _on_move(self, x, y):
        now  = time.time()
        speed = 0.0

        if self._last_pos and self._last_move_ts:
            dx   = x - self._last_pos[0]
            dy   = y - self._last_pos[1]
            dt   = now - self._last_move_ts
            dist = math.hypot(dx, dy)
            speed = dist / dt if dt > 0 else 0.0

        erratic = speed > ERRATIC_SPEED_PX_S

        ev = Event.make("mouse", self.session_id, x=x, y=y,
                        subtype="move", speed_px_s=round(speed, 1),
                        erratic=erratic)
        self._last_pos     = (x, y)
        self._last_move_ts = now
        self._emit(ev)

    def _on_click(self, x, y, button, pressed):
        if not pressed:
            return
        now = time.time()

        # Rage click: ≥ N klik dalam radius R dan window W detik
        self._click_buf.append({"x": x, "y": y, "t": now})
        nearby = [c for c in self._click_buf
                  if now - c["t"] <= RAGE_CLICK_WINDOW
                  and math.hypot(c["x"] - x, c["y"] - y) < RAGE_CLICK_RADIUS]
        rage = len(nearby) >= RAGE_CLICK_MIN

        # Hesitation: mouse diam > threshold sebelum klik ini
        hesitation = 0.0
        if self._last_move_ts:
            hesitation = now - self._last_move_ts
        hesitated = hesitation >= HESITATION_PAUSE

        ev = Event.make("mouse", self.session_id, x=x, y=y,
                        subtype="click", button=str(button),
                        rage_click=rage,
                        hesitated=hesitated,
                        hesitation_s=round(hesitation, 2))
        self._emit(ev)

    def _on_scroll(self, x, y, dx, dy):
        ev = Event.make("mouse", self.session_id, x=x, y=y,
                        subtype="scroll", dx=dx, dy=dy)
        self._emit(ev)
