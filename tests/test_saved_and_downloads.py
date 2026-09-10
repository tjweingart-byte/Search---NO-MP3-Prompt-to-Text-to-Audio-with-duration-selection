"""Save for later and download - which are deliberately not the same thing."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import entitlements
import saved as saved_mod


@pytest.fixture
def store(tmp_path):
    return saved_mod.SavedStore(str(tmp_path / "saved.db"))


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setenv("FREE_MAX_DOWNLOADS", "2")
    entitlements.reload_tiers()
    yield
    for name in entitlements.LIMIT_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    entitlements.reload_tiers()


# --- saving ---------------------------------------------------------------

def test_saving_the_same_episode_twice_is_one_item(store):
    """Keyed on (question, length), which is also the script cache's key: two
    saves differing only in something the cache ignores are one episode."""
    first = store.save("u", "why bonds move", 3, title="Bonds")
    second = store.save("u", "why bonds move", 3, title="Bonds")
    assert first.id == second.id
    assert len(store.items("u")) == 1


def test_saving_again_into_a_folder_files_it_rather_than_failing(store):
    """From the listener's side they pressed save and it is saved, which is
    true either way - so the second press should do the useful thing."""
    folder = store.create_folder("u", "Commute")
    store.save("u", "why bonds move", 3)
    again = store.save("u", "why bonds move", 3, folder_id=folder["id"])
    assert again.folder_id == folder["id"]


def test_an_episode_with_no_question_cannot_be_saved(store):
    with pytest.raises(saved_mod.SavedError):
        store.save("u", "   ", 3)


def test_saving_into_somebody_elses_folder_is_refused(store):
    theirs = store.create_folder("them", "Private")
    with pytest.raises(saved_mod.SavedError):
        store.save("u", "why bonds move", 3, folder_id=theirs["id"])


def test_one_listeners_shelf_is_not_anothers(store):
    store.save("u", "why bonds move", 3)
    assert store.items("them") == []


# --- folders --------------------------------------------------------------

def test_folders_count_what_is_in_them(store):
    folder = store.create_folder("u", "Commute")
    store.save("u", "a question", 3, folder_id=folder["id"])
    assert store.folders("u")[0]["items"] == 1


def test_two_folders_cannot_share_a_name(store):
    store.create_folder("u", "Commute")
    with pytest.raises(saved_mod.SavedError):
        store.create_folder("u", "commute")


def test_deleting_a_folder_unfiles_its_episodes_rather_than_deleting_them(store):
    """Deleting somebody's saved episodes because they tidied their folders is
    the kind of surprise that stops people using a feature - and a download
    inside it would be bytes on their phone held against their limit with
    nothing pointing at them."""
    folder = store.create_folder("u", "Commute")
    store.save("u", "a question", 3, folder_id=folder["id"])
    assert store.delete_folder("u", folder["id"]) == 1
    remaining = store.items("u")
    assert len(remaining) == 1
    assert remaining[0].folder_id == ""


def test_a_folder_needs_a_name(store):
    with pytest.raises(saved_mod.SavedError):
        store.create_folder("u", "   ")


# --- downloads ------------------------------------------------------------

def test_a_saved_episode_is_not_downloaded_until_it_is_asked_for(store):
    """The whole distinction. Save for later is a pointer and needs the
    internet; download is the audio on the device."""
    item = store.save("u", "why bonds move", 3)
    assert item.downloaded is False
    assert store.download_status("u", "free")["used"] == 0


def test_downloading_takes_a_slot_and_releasing_gives_it_back(store, small):
    item = store.save("u", "why bonds move", 3)
    store.reserve_download("u", item.id, "free")
    assert store.download_status("u", "free")["remaining"] == 1
    assert store.release_download("u", item.id) is True
    assert store.download_status("u", "free")["remaining"] == 2


def test_releasing_a_download_keeps_the_episode_saved(store, small):
    """"I need the space" and "I am not interested any more" are different
    requests, and merging them loses somebody's list while they tidy their
    phone."""
    item = store.save("u", "why bonds move", 3)
    store.reserve_download("u", item.id, "free")
    store.release_download("u", item.id)
    assert len(store.items("u")) == 1
    assert store.item("u", item.id).downloaded is False


def test_a_full_shelf_says_what_to_clear(store, small):
    """A limit without a remedy is a dead end, and on a phone the listener
    cannot go and look somewhere else."""
    for i in range(2):
        item = store.save("u", f"question {i}", 3)
        store.reserve_download("u", item.id, "free")
    third = store.save("u", "question 3", 3)
    with pytest.raises(saved_mod.DownloadLimit) as exc:
        store.reserve_download("u", third.id, "free")
    assert exc.value.candidates
    assert "Remove one" in str(exc.value)


def test_what_to_clear_offers_the_least_recently_played_first(store, small):
    """Not the oldest: the one saved first is often the one kept on purpose."""
    keep = store.save("u", "keep this", 3)
    stale = store.save("u", "never played", 3)
    store.reserve_download("u", keep.id, "free", at=100)
    store.reserve_download("u", stale.id, "free", at=200)
    store.played("u", keep.id, at=300)

    third = store.save("u", "another", 3)
    with pytest.raises(saved_mod.DownloadLimit) as exc:
        store.reserve_download("u", third.id, "free")
    assert exc.value.candidates[0]["id"] == stale.id


def test_a_better_tier_holds_more(store, small):
    for i in range(3):
        item = store.save("u", f"question {i}", 3)
        store.reserve_download("u", item.id, "plus")
    assert store.download_status("u", "plus")["used"] == 3


def test_the_unlimited_tier_has_no_server_side_cap(store):
    for i in range(12):
        item = store.save("u", f"question {i}", 3)
        store.reserve_download("u", item.id, "unlimited")
    status = store.download_status("u", "unlimited")
    assert status["unlimited"] is True
    assert status["remaining"] == entitlements.UNLIMITED


def test_downloading_twice_is_not_two_slots(store, small):
    item = store.save("u", "why bonds move", 3)
    store.reserve_download("u", item.id, "free")
    store.reserve_download("u", item.id, "free")
    assert store.download_status("u", "free")["used"] == 1


def test_removing_a_downloaded_item_frees_its_slot(store, small):
    item = store.save("u", "why bonds move", 3)
    store.reserve_download("u", item.id, "free")
    store.remove("u", item.id)
    assert store.download_status("u", "free")["used"] == 0


def test_the_size_is_estimated_before_the_listener_agrees_to_it(store):
    """So the popup can say "about 8 MB" before they say yes rather than
    after."""
    assert saved_mod.estimated_bytes(3) == 3 * 60 * saved_mod.BYTES_PER_SECOND
    assert saved_mod.estimated_bytes(0) > 0


def test_the_device_can_correct_the_estimate(store, small):
    """An episode ends when it runs out of substance, so the real size is
    usually smaller than the ceiling the estimate assumed."""
    item = store.save("u", "why bonds move", 3)
    store.reserve_download("u", item.id, "free")
    store.confirm_download("u", item.id, 4_000_000)
    assert store.item("u", item.id).bytes == 4_000_000


def test_confirming_a_download_nobody_reserved_does_not_create_one(store, small):
    item = store.save("u", "why bonds move", 3)
    store.confirm_download("u", item.id, 4_000_000)
    assert store.item("u", item.id).downloaded is False


# --- deletion -------------------------------------------------------------

def test_forget_erases_the_whole_shelf(store):
    folder = store.create_folder("u", "Commute")
    store.save("u", "a question", 3, folder_id=folder["id"])
    store.save("them", "their question", 3)
    store.forget("u")
    assert store.items("u") == [] and store.folders("u") == []
    assert len(store.items("them")) == 1
