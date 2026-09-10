"""Saving, downloading and sharing, through the API."""
from __future__ import annotations

import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as appmod
import entitlements


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(appmod, "_rate_limit", lambda request: None)
    with TestClient(appmod.app) as c:
        yield c


@pytest.fixture
def account(client):
    client.post("/api/auth/signup", json={"email": "a@b.com", "password": "password12"})
    return client.get("/api/auth/me").json()["user_id"]


@pytest.fixture
def one_slot(monkeypatch):
    monkeypatch.setenv("FREE_MAX_DOWNLOADS", "1")
    entitlements.reload_tiers()
    yield
    for name in entitlements.LIMIT_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    entitlements.reload_tiers()


# --- saving ---------------------------------------------------------------

def test_saving_returns_the_download_status_with_it(client, account):
    """The interface asks about downloading the moment something is saved, and
    a popup whose answer is "you have no room" would be a worse question than
    not asking."""
    body = client.post("/api/saved", json={
        "query": "why bonds move", "minutes": 3, "title": "Bonds"}).json()
    assert body["item"]["downloaded"] is False
    assert body["downloads"]["limit"] > 0
    assert body["item"]["estimated_bytes"] > 0


def test_the_shelf_needs_an_account(client):
    """The settled boundary: an account gates what is *kept*. A saved episode
    is the definition of kept."""
    assert client.get("/api/saved").status_code == 401
    assert client.post("/api/saved", json={"query": "q", "minutes": 3}).status_code == 401


def test_folders_can_be_made_and_filtered_by(client, account):
    folder = client.post("/api/saved/folders", json={"name": "Commute"}).json()["folder"]
    client.post("/api/saved", json={"query": "in the folder", "minutes": 3,
                                    "folder_id": folder["id"]})
    client.post("/api/saved", json={"query": "unfiled", "minutes": 3})
    filtered = client.get(f"/api/saved?folder_id={folder['id']}").json()
    assert [i["query"] for i in filtered["items"]] == ["in the folder"]


def test_deleting_a_folder_keeps_its_episodes(client, account):
    folder = client.post("/api/saved/folders", json={"name": "Commute"}).json()["folder"]
    client.post("/api/saved", json={"query": "in the folder", "minutes": 3,
                                    "folder_id": folder["id"]})
    client.request("DELETE", f"/api/saved/folders/{folder['id']}")
    assert len(client.get("/api/saved").json()["items"]) == 1


# --- downloading ----------------------------------------------------------

def test_a_download_hands_back_the_stream_to_fill_it_with(client, account):
    """There is no file to serve - nothing writes one. The client downloads by
    streaming the same endpoint it would play from, and keeps what arrives."""
    item = client.post("/api/saved", json={
        "query": "why bonds move", "minutes": 3}).json()["item"]
    got = client.post(f"/api/saved/{item['id']}/download").json()
    assert got["item"]["downloaded"] is True
    assert got["stream"].startswith("/api/audio?q=")
    assert "fmt=pcm" in got["stream"]


def test_a_full_shelf_is_a_409_naming_what_to_clear(client, account, one_slot):
    """A capacity, not a rate - so not a 429. And a limit without a remedy is
    a dead end on a phone, which is why the candidates ride along."""
    first = client.post("/api/saved", json={"query": "first", "minutes": 3}).json()["item"]
    client.post(f"/api/saved/{first['id']}/download")
    second = client.post("/api/saved", json={"query": "second", "minutes": 3}).json()["item"]

    refused = client.post(f"/api/saved/{second['id']}/download")
    assert refused.status_code == 409
    extra = json.loads(refused.headers["X-FAM-Downloads"])
    assert [c["query"] for c in extra["candidates"]] == ["first"]
    assert extra["status"]["remaining"] == 0


def test_freeing_a_slot_makes_room_without_losing_the_episode(client, account, one_slot):
    first = client.post("/api/saved", json={"query": "first", "minutes": 3}).json()["item"]
    client.post(f"/api/saved/{first['id']}/download")
    released = client.request("DELETE", f"/api/saved/{first['id']}/download").json()
    assert released["downloads"]["remaining"] == 1
    # Still saved. "I need the space" and "I am not interested" are different
    # requests.
    assert len(client.get("/api/saved").json()["items"]) == 1


def test_the_device_can_correct_the_estimate(client, account):
    item = client.post("/api/saved", json={"query": "q", "minutes": 3}).json()["item"]
    client.post(f"/api/saved/{item['id']}/download")
    fixed = client.post(f"/api/saved/{item['id']}/download/confirm",
                        json={"bytes": 4_000_000}).json()
    assert fixed["item"]["bytes"] == 4_000_000


def test_a_better_tier_holds_more(client, account, one_slot):
    appmod.ACCOUNTS.set_plan(account, "plus")
    for i in range(4):
        item = client.post("/api/saved", json={"query": f"q{i}", "minutes": 3}
                           ).json()["item"]
        assert client.post(f"/api/saved/{item['id']}/download").status_code == 200


# --- sharing --------------------------------------------------------------

def test_sharing_works_without_an_account(client):
    """A share link is the cheapest route FAM has to a listener who does not
    have it yet. Putting a sign-up in front of the act of recommending it
    would be a strange way to grow, and nothing durable is being kept."""
    made = client.post("/api/share", json={
        "query": "why bonds move", "minutes": 3, "title": "Bonds"})
    assert made.status_code == 200, made.text
    assert made.json()["url"]


def test_a_share_carries_wording_for_every_destination(client):
    body = client.post("/api/share", json={
        "query": "why bonds move", "minutes": 3, "title": "Bonds"}).json()
    for key in ("sms", "email", "x", "facebook", "linkedin",
                "instagram_story", "snapchat_story"):
        assert body["targets"][key]["text"], key


def test_a_share_says_when_its_link_is_not_public_yet(client):
    """Without PUBLIC_BASE_URL the link works inside the app and nowhere else.
    A share posted to LinkedIn that resolves to localhost is the quiet failure
    this project keeps a rule about."""
    body = client.post("/api/share", json={"query": "q", "minutes": 3}).json()
    assert body["public"] is False
    assert body["url"].startswith("/s/")


def test_a_public_base_url_makes_the_link_absolute(client, monkeypatch):
    import dataclasses

    import config

    monkeypatch.setattr(appmod, "settings", dataclasses.replace(
        config.settings, public_base_url="https://fam.audio"))
    body = client.post("/api/share", json={"query": "q", "minutes": 3}).json()
    assert body["url"].startswith("https://fam.audio/s/")
    assert body["public"] is True


def test_the_story_card_is_an_image(client):
    """Instagram and Snapchat stories cannot carry a link as text. Without a
    card the listener shares a screenshot of a player UI."""
    body = client.post("/api/share", json={
        "query": "q", "minutes": 3, "title": "Bonds"}).json()
    card = client.get(body["card"])
    assert card.status_code == 200
    assert card.headers["content-type"].startswith("image/svg+xml")


def test_following_a_share_link_lands_on_the_episode_and_is_counted(client):
    body = client.post("/api/share", json={
        "query": "why bonds move", "minutes": 3}).json()
    hop = client.get(body["url"], follow_redirects=False)
    assert hop.status_code == 302
    assert "q=why" in hop.headers["location"]
    assert appmod.SHARES.get(body["share"]["id"])["opens"] == 1


def test_an_unknown_share_link_lands_in_the_app_rather_than_on_an_error(client):
    """Somebody followed a link a friend sent them. A 404 is a worse first
    impression of FAM than the home screen."""
    assert client.get("/s/nope", follow_redirects=False).status_code == 302


def test_the_targets_catalogue_says_which_need_a_picture(client):
    targets = {t["key"]: t for t in client.get("/api/share/targets").json()["targets"]}
    assert targets["instagram_story"]["needs_image"] is True
    assert targets["linkedin"]["needs_image"] is False


# --- deletion -------------------------------------------------------------

def test_deleting_an_account_clears_the_shelf_and_the_shares(client, account):
    client.post("/api/saved", json={"query": "q", "minutes": 3})
    client.post("/api/share", json={"query": "q", "minutes": 3})
    removed = client.request("DELETE", "/api/account").json()["removed"]
    assert removed["saved"] >= 1
    assert removed["shares"] >= 1
