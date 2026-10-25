"""Chat routing: commands about known items start the arm, everything else is conversation."""
from swarmmind.cell.labmate import parse_command

KNOWN = {"screwdriver", "spring", "tape_measure", "file", "superglue", "stepper_motor"}


def test_fetch_phrasings():
    for text, item in [("bring me the screwdriver", "screwdriver"), ("Can you hand me a spring please?", "spring"),
                       ("fetch tape measure", "tape_measure"), ("pick up the file", "file"),
                       ("grab the stepper motors", "stepper_motor"), ("Could you please get me my screwdriver.", "screwdriver")]:
        assert parse_command(text, KNOWN) == {"intent": "fetch", "item": item, "to": "handover"}, text


def test_put_in_tray():
    for text in ["put the spring in the tray", "move the spring onto the safe tray", "Place a spring into the tray please"]:
        assert parse_command(text, KNOWN) == {"intent": "fetch", "item": "spring", "to": "safe_tray"}, text


def test_conversation_is_not_a_command():
    for text in ["give me a summary of the bench", "what is on the table?", "get well soon", "bring me the banana",
                 "describe the bench", "is the superglue open?", ""]:
        assert parse_command(text, KNOWN) is None, text
