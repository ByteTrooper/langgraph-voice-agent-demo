"""In-memory appointment calendar standing in for a real booking backend."""
from __future__ import annotations

SLOTS = ["09:00", "10:00", "11:00", "14:00", "15:00", "16:00"]


class Calendar:
    def __init__(self) -> None:
        self._booked: dict[str, dict] = {}
        self._next = 1000

    def available(self, date: str) -> list[str]:
        taken = {b["time"] for b in self._booked.values() if b["date"] == date}
        return [s for s in SLOTS if s not in taken]

    def book(self, date: str, time: str, name: str) -> dict:
        if time not in self.available(date):
            return {"ok": False, "error": "slot_unavailable"}
        self._next += 1
        booking_id = f"B{self._next}"
        self._booked[booking_id] = {"date": date, "time": time, "name": name}
        return {"ok": True, "booking_id": booking_id}

    def cancel(self, booking_id: str) -> dict:
        if self._booked.pop(booking_id.upper(), None) is None:
            return {"ok": False, "error": "not_found"}
        return {"ok": True}
