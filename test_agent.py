from datetime import date

import pytest

from agent import Session, detect_language

TODAY = date(2026, 11, 24)


@pytest.fixture
def s():
    return Session(today=TODAY)


def test_language_by_script():
    assert detect_language("hello") == "en"
    assert detect_language("आज के स्लॉट") == "hi"
    assert detect_language("రేపు స్లాట్") == "te"


def test_check_slots(s):
    reply = s.say("what slots are free tomorrow")
    assert "2026-11-25" in reply and "09:00" in reply


def test_book_asks_for_missing_slots_then_books(s):
    assert "name" in s.say("book 10am tomorrow").lower()
    reply = s.say("Riya")
    assert "Riya" in reply and "10:00" in reply and "B1001" in reply
    assert "10:00" not in s.say("slots tomorrow")


def test_double_booking_offers_alternatives(s):
    s.say("book 10am tomorrow for Riya")
    reply = s.say("book 10am tomorrow for Asha")
    assert "taken" in reply and "09:00" in reply


def test_cancel_and_unknown_id(s):
    s.say("book 3pm tomorrow for Riya")
    assert "cancelled" in s.say("cancel B1001")
    assert "couldn't find" in s.say("cancel B1001")


def test_reply_follows_user_language(s):
    assert "खाली" in s.say("आज के स्लॉट बताओ")
    assert "ఖాళీ" in s.say("రేపు స్లాట్లు చూపించు")


def test_guardrail_blocks_secrets_before_any_tool(s):
    reply = s.say("book 10am tomorrow and my card number is 4111")
    assert "can't help" in reply
    assert s.calendar.available("2026-11-25") == ["09:00", "10:00", "11:00", "14:00", "15:00", "16:00"]


def test_unknown_intent_falls_back(s):
    assert "check slots" in s.say("tell me a joke")
