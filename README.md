# langgraph-voice-agent-demo

A small, runnable example of how a voice or chat agent turn can be structured as a
[LangGraph](https://github.com/langchain-ai/langgraph) graph: **guardrail → understand → tool → reply**,
with multi-turn slot filling and replies in the caller's language (English, Hindi, Telugu).

It runs fully offline. The "understanding" step is a small rule-based class, so you can read
the whole flow, run the tests and see the behaviour without an API key. The `Understander`
protocol is the seam where an LLM with tool calling would plug in.

```
START -> guardrail -+-> understand -> run_tool -> respond -> END
                    |                               ^
                    +--------- blocked -------------+
```

## What it shows

- **Graph shape:** one node per concern, a conditional edge for the guardrail short-circuit.
- **Slot filling across turns:** "book 10am tomorrow" asks for a name, the next turn supplies it.
  Partial slots live in `pending` and are carried by `Session`, like a call's conversation state.
- **Language follow:** the reply language comes from the script of the caller's message
  (Devanagari → Hindi, Telugu script → Telugu). Romanised Hindi/Telugu counts as English.
- **Guardrail before tools:** requests that ask for secrets are refused and never reach the calendar.
- **Tool layer behind an interface:** `mock_api.Calendar` is an in-memory stand-in for a booking API.

## Run it

```bash
pip install -r requirements.txt
python agent.py --demo     # scripted conversation
python agent.py            # interactive
python -m pytest -q        # 8 tests
```

Demo output:

```
you  > What slots are free tomorrow?
bot  > Free on 2026-11-25: 09:00, 10:00, 11:00, 14:00, 15:00, 16:00.

you  > Book 10am tomorrow
bot  > Whose name should I put the booking under?

you  > Riya
bot  > Done. Riya, you're booked on 2026-11-25 at 10:00. Booking id B1001.

you  > आज के स्लॉट बताओ
bot  > 2026-11-24 को खाली समय: 09:00, 10:00, 11:00, 14:00, 15:00, 16:00।

you  > my card number is 4111
bot  > I can't help with that. I never need passwords or card details.

you  > cancel B1001
bot  > Booking B1001 is cancelled.
```

## Plugging in a real model

Implement `parse(text, pending, today) -> (intent, args)` using your LLM's tool calling and pass it in:

```python
session = Session(understander=MyLLMUnderstander())
```

## Limits

This is a teaching demo. The keyword guardrail is intentionally simple and is not a
production safety layer, language detection is script-based, and dates and times are parsed
with a few rules, not a full natural-language parser. Production voice agents also need
streaming speech-to-text and text-to-speech, barge-in handling and latency work, none of
which is in this repo.

## License

MIT
