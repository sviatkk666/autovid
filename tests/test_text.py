"""Deterministic text utilities: slugify, humanizer cleanup, scene splitting."""

from autovid.modules.humanizer import deterministic_cleanup
from autovid.modules.parser import _parse_deterministic
from autovid.util import slugify


def test_slugify():
    assert slugify("Why Discipline Beats Motivation!") == "why-discipline-beats-motivation"
    assert slugify("  ") == "project"                      # never empty
    assert len(slugify("x" * 200)) <= 50


def test_cleanup_strips_ai_tells():
    out = deterministic_cleanup(
        "It's important to note that discipline delves into habit — every day. "
        "Furthermore, it utilizes a plethora of tricks."
    )
    lower = out.lower()
    for tell in ("important to note", "delves", "furthermore", "utilizes", "plethora", "—"):
        assert tell not in lower
    assert "digs into" in lower and "plenty of" in lower


def test_cleanup_recapitalizes_after_opener():
    out = deterministic_cleanup("In today's world, discipline wins.")
    assert out == "Discipline wins."


def test_deterministic_parse_keeps_narration_verbatim():
    text = ("One two three four five. Six seven eight nine ten.\n\n"
            "A second paragraph starts a new scene.")
    cfg = {"parser": {"words_per_second": 2.5, "target_scene_seconds": 2}}
    scenes = _parse_deterministic(text, cfg)

    assert len(scenes) >= 2
    assert [s.id for s in scenes] == list(range(1, len(scenes) + 1))
    # Concatenating scene text reproduces the script (whitespace aside).
    assert " ".join(s.text for s in scenes).split() == text.split()
    assert all(s.est_duration_sec > 0 for s in scenes)
