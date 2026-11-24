"""Voice-agent turn loop as a LangGraph graph: guardrail -> understand -> tool -> reply.

The understander is pluggable. The default one is rule-based so the demo and tests
run offline; swap in an LLM-backed class with the same `parse` signature.
"""
from __future__ import annotations

import re
import sys
from datetime import date, timedelta
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from mock_api import Calendar

# Demo guardrail only: refuse requests that ask the bot for secrets.
BLOCKED_WORDS = ("otp", "password", "cvv", "card number", "pin number")

REPLIES = {
    "en": {
        "blocked": "I can't help with that. I never need passwords or card details.",
        "ask_date": "Which date would you like?",
        "ask_time": "What time works for you?",
        "ask_name": "Whose name should I put the booking under?",
        "slots": "Free on {date}: {slots}.",
        "no_slots": "Nothing is free on {date}.",
        "booked": "Done. {name}, you're booked on {date} at {time}. Booking id {id}.",
        "taken": "That slot is taken. Free times: {slots}.",
        "cancelled": "Booking {id} is cancelled.",
        "not_found": "I couldn't find that booking id.",
        "fallback": "I can check slots, book or cancel an appointment.",
    },
    "hi": {
        "blocked": "मैं इसमें मदद नहीं कर सकता। मुझे पासवर्ड या कार्ड की जानकारी कभी नहीं चाहिए।",
        "ask_date": "आप किस तारीख को आना चाहेंगे?",
        "ask_time": "कौन सा समय ठीक रहेगा?",
        "ask_name": "बुकिंग किस नाम से करूँ?",
        "slots": "{date} को खाली समय: {slots}।",
        "no_slots": "{date} को कोई स्लॉट खाली नहीं है।",
        "booked": "हो गया। {name}, आपकी बुकिंग {date} को {time} बजे है। बुकिंग आईडी {id}।",
        "taken": "यह स्लॉट भर चुका है। खाली समय: {slots}।",
        "cancelled": "बुकिंग {id} रद्द हो गई।",
        "not_found": "यह बुकिंग आईडी नहीं मिली।",
        "fallback": "मैं स्लॉट देख सकता हूँ, बुक या रद्द कर सकता हूँ।",
    },
    "te": {
        "blocked": "దానికి నేను సహాయం చేయలేను. నాకు పాస్‌వర్డ్ లేదా కార్డ్ వివరాలు ఎప్పుడూ అవసరం లేదు.",
        "ask_date": "ఏ తేదీకి కావాలి?",
        "ask_time": "ఏ సమయం అనుకూలం?",
        "ask_name": "ఏ పేరుతో బుక్ చేయాలి?",
        "slots": "{date} న ఖాళీ సమయాలు: {slots}.",
        "no_slots": "{date} న ఖాళీ లేదు.",
        "booked": "అయిపోయింది. {name}, మీ బుకింగ్ {date} న {time} కి. బుకింగ్ ఐడీ {id}.",
        "taken": "ఆ స్లాట్ నిండిపోయింది. ఖాళీ సమయాలు: {slots}.",
        "cancelled": "బుకింగ్ {id} రద్దు అయింది.",
        "not_found": "ఆ బుకింగ్ ఐడీ కనపడలేదు.",
        "fallback": "నేను స్లాట్లు చూడగలను, బుక్ లేదా రద్దు చేయగలను.",
    },
}

KEYWORDS = {
    "cancel": ("cancel", "रद्द", "రద్దు"),
    "book": ("book", "reserve", "schedule", "बुक", "బుక్"),
    "check": ("available", "slots", "free", "स्लॉट", "స్లాట్"),
}
TODAY_WORDS = ("today", "आज", "ఈ రోజు")
TOMORROW_WORDS = ("tomorrow", "कल", "రేపు")
TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.I)
ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NAME_RE = re.compile(r"(?:for|name is|name)\s+([A-Za-z]+)", re.I)
ID_RE = re.compile(r"\bB\d{4}\b", re.I)


class State(TypedDict, total=False):
    user_text: str
    pending: dict  # slots collected so far, carried across turns
    language: str
    intent: str
    args: dict
    blocked: bool
    tool_result: dict
    reply: str


class Understander(Protocol):
    def parse(self, text: str, pending: dict, today: date) -> tuple[str, dict]: ...


def detect_language(text: str) -> str:
    """Script based. Romanised Hindi/Telugu is treated as English."""
    for ch in text:
        if "ऀ" <= ch <= "ॿ":
            return "hi"
        if "ఀ" <= ch <= "౿":
            return "te"
    return "en"


class RuleUnderstander:
    def parse(self, text: str, pending: dict, today: date) -> tuple[str, dict]:
        low = text.lower()
        intent = pending.get("intent") or "other"
        for name, words in KEYWORDS.items():
            if any(w in low for w in words):
                intent = name
                break
        args = {k: v for k, v in pending.items() if k != "intent"}

        if any(w in low for w in TOMORROW_WORDS):
            args["date"] = (today + timedelta(days=1)).isoformat()
        elif any(w in low for w in TODAY_WORDS):
            args["date"] = today.isoformat()
        elif m := ISO_RE.search(text):
            args["date"] = m.group(0)

        text_wo_date = ISO_RE.sub(" ", text)
        if m := TIME_RE.search(text_wo_date):
            hour, minute, ampm = int(m.group(1)), m.group(2) or "00", (m.group(3) or "").lower()
            if ampm == "pm" and hour < 12:
                hour += 12
            if ampm == "am" and hour == 12:
                hour = 0
            if 0 <= hour <= 23 and (ampm or m.group(2)):
                args["time"] = f"{hour:02d}:{minute}"

        if m := NAME_RE.search(text):
            args["name"] = m.group(1).title()
        elif pending.get("expecting") == "name" and text.strip().isalpha():
            args["name"] = text.strip().title()
        if m := ID_RE.search(text):
            args["booking_id"] = m.group(0).upper()
        return intent, args


def build_graph(calendar: Calendar, understander: Understander, today: date):
    def guardrail(state: State) -> State:
        low = state["user_text"].lower()
        return {
            "language": detect_language(state["user_text"]),
            "blocked": any(w in low for w in BLOCKED_WORDS),
        }

    def understand(state: State) -> State:
        intent, args = understander.parse(state["user_text"], state.get("pending", {}), today)
        return {"intent": intent, "args": args}

    def run_tool(state: State) -> State:
        intent, args = state["intent"], state["args"]
        if intent == "check":
            if "date" not in args:
                return {"tool_result": {"missing": "date"}}
            return {"tool_result": {"slots": calendar.available(args["date"]), "date": args["date"]}}
        if intent == "book":
            for slot in ("date", "time", "name"):
                if slot not in args:
                    return {"tool_result": {"missing": slot}}
            res = calendar.book(args["date"], args["time"], args["name"])
            return {"tool_result": {**res, **args}}
        if intent == "cancel":
            if "booking_id" not in args:
                return {"tool_result": {"missing": "booking_id"}}
            return {"tool_result": {**calendar.cancel(args["booking_id"]), **args}}
        return {"tool_result": {}}

    def respond(state: State) -> State:
        t = REPLIES[state["language"]]
        if state.get("blocked"):
            return {"reply": t["blocked"], "pending": {}}
        res, intent, args = state.get("tool_result", {}), state["intent"], state["args"]
        if "missing" in res:
            slot = res["missing"]
            key = {"date": "ask_date", "time": "ask_time", "name": "ask_name"}.get(slot, "fallback")
            keep = {**args, "intent": intent, "expecting": slot}
            return {"reply": t[key], "pending": keep}
        if intent == "check":
            if res["slots"]:
                return {"reply": t["slots"].format(date=res["date"], slots=", ".join(res["slots"])), "pending": {}}
            return {"reply": t["no_slots"].format(date=res["date"]), "pending": {}}
        if intent == "book":
            if res.get("ok"):
                return {
                    "reply": t["booked"].format(name=res["name"], date=res["date"], time=res["time"], id=res["booking_id"]),
                    "pending": {},
                }
            free = ", ".join(calendar.available(res["date"])) or "-"
            return {"reply": t["taken"].format(slots=free), "pending": {**args, "intent": "book", "expecting": "time"}}
        if intent == "cancel":
            key = "cancelled" if res.get("ok") else "not_found"
            return {"reply": t[key].format(id=res.get("booking_id", "")), "pending": {}}
        return {"reply": t["fallback"], "pending": {}}

    g = StateGraph(State)
    g.add_node("guardrail", guardrail)
    g.add_node("understand", understand)
    g.add_node("run_tool", run_tool)
    g.add_node("respond", respond)
    g.add_edge(START, "guardrail")
    g.add_conditional_edges("guardrail", lambda s: "respond" if s["blocked"] else "understand",
                            {"respond": "respond", "understand": "understand"})
    g.add_edge("understand", "run_tool")
    g.add_edge("run_tool", "respond")
    g.add_edge("respond", END)
    return g.compile()


class Session:
    """Carries `pending` slots between turns, like a call's conversation state."""

    def __init__(self, today: date | None = None, understander: Understander | None = None):
        self.calendar = Calendar()
        self.graph = build_graph(self.calendar, understander or RuleUnderstander(), today or date.today())
        self.pending: dict = {}

    def say(self, text: str) -> str:
        # intent/args defaults: a blocked turn skips `understand` but `respond` still reads them.
        out = self.graph.invoke({"user_text": text, "pending": self.pending, "intent": "other", "args": {}})
        self.pending = out.get("pending", {})
        return out["reply"]


DEMO = [
    "What slots are free tomorrow?",
    "Book 10am tomorrow",
    "Riya",
    "आज के स्लॉट बताओ",
    "my card number is 4111",
    "cancel B1001",
]

if __name__ == "__main__":
    session = Session()
    if "--demo" in sys.argv:
        for line in DEMO:
            print(f"you  > {line}\nbot  > {session.say(line)}\n")
    else:
        while True:
            try:
                print("bot  >", session.say(input("you  > ")))
            except (EOFError, KeyboardInterrupt):
                break
