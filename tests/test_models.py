"""Project / Channel documents through the storage layer (filesystem backend)."""

from autovid.channel import Channel
from autovid.project import Project, Scene


def test_project_save_load_list_delete():
    p = Project(slug="t-video", title="Test Video", channel="t-chan",
                scenes=[Scene(id=1, text="hello"), Scene(id=2, text="world")])
    p.save()
    assert Project.exists("t-video")

    loaded = Project.load("t-video")
    assert loaded.title == "Test Video"
    assert [s.text for s in loaded.scenes] == ["hello", "world"]
    assert "t-video" in [x.slug for x in Project.list()]

    Project.delete("t-video")
    assert not Project.exists("t-video")
    assert not p.dir.exists()  # asset dir removed with the document


def test_project_tolerates_unknown_fields():
    """Docs written by newer/older versions must still load (extra keys dropped)."""
    p = Project(slug="t-compat", scenes=[Scene(id=1, text="x")])
    p.save()
    from autovid.storage import store
    doc = store().get("project", "t-compat")
    doc["added_in_v99"] = True
    doc["scenes"][0]["scene_field_v99"] = "y"
    store().put("project", "t-compat", doc)

    loaded = Project.load("t-compat")
    assert loaded.scenes[0].text == "x"
    Project.delete("t-compat")


def test_scene_ids_are_stable_keys():
    p = Project(slug="t-ids", scenes=[Scene(id=1, text="a"), Scene(id=3, text="b")])
    assert p.next_scene_id() == 4          # max+1, never reuses a deleted id
    assert p.scene_by_id(3).text == "b"
    assert p.scene_by_id(99) is None


def test_channel_roundtrip_and_missing():
    ch = Channel(slug="t-chan", name="Test Channel", niche="educational")
    ch.save()
    assert Channel.load("t-chan").name == "Test Channel"
    assert "t-chan" in [c.slug for c in Channel.list()]

    Channel.delete("t-chan")
    assert not Channel.exists("t-chan")
    try:
        Channel.load("t-chan")
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass
