# Friends, sharing, and keeping an episode

Four capabilities that look like one feature and are not: **following**
somebody, **sending** them an episode, **posting** one outside FAM, and
**keeping** one — either as a pointer or as audio on the device.
`social.py`, `messages.py`, `sharing.py` and `saved.py` are the code; this is
what was decided and why.

The property underneath all of it: **none of them generate anything.** A share,
an echo, a saved item and a message are all rows pointing at a question whose
script is already in the shared cache. Sending an episode to ten people costs
ten rows, not ten episodes — their taps are what synthesise audio, against
their own allowances, from one script. That is the same design the browse
surfaces use, and it is why a social layer does not change the cost model.

## Following

Asymmetric, because the app's own copy already said so — "What your followers
are listening to" was ranking co-listener overlap and promising a social
network that did not exist (CLAUDE.md open problem #6). Now there is a graph.

A **friend** is the mutual case and is *derived, never stored*. No request, no
accept, no pending state: two rows that happen to point at each other. That
removes the state machine where this shape usually goes wrong, and it makes it
impossible for the two directions to disagree.

Sharing needs neither. You can send an episode to somebody who does not follow
you back, exactly as you can text them.

Only people who have chosen a handle are findable, and a search needs two
characters — one letter returns most of the listener table, which is a
directory dump rather than a search. Somebody who never set a handle is not
hidden by a privacy setting; they are simply not in a directory.

## Sending an episode inside FAM

An **echo** is a broadcast — you push an episode at everyone who follows you.
A **share** is directed: one person, one episode, deliberately. Both are one
row.

Thread ids are **derived** from the two listener ids, sorted and joined. So
opening a conversation needs no write, and two people opening the same one
simultaneously cannot create two threads — the classic bug in this shape, which
produces a split history nobody can merge afterwards.

A message carries the **question and the length** and never a script or audio.
Those two fields are exactly the script cache's key, so the recipient's play is
a cache hit.

What is deliberately not here: **group threads** (a share to a group is closer
to an echo, and that is a decision to make when somebody wants one), **delivery
receipts** and **typing indicators** (both are promises about somebody else's
attention, and this app has a rule against inventing state it does not have).

## Sharing outside FAM

**FAM posts nothing and holds no token for any platform.** Not a gap — the
correct shape. Every destination is reached from the phone: iOS hands the app a
share sheet, and the story formats have an SDK hand-off where the app passes an
image and a link and the *platform's* app posts it, with the person watching.
So the server's job is to produce the payload and a share is something the
listener completes. No OAuth to maintain, no tokens to leak, no scope reviews
with four companies, and nothing that can post while somebody is asleep.

Nine destinations, and `kind` is what actually differs:

| kind | destinations | what it takes |
|---|---|---|
| `copy` | Copy link | the URL |
| `message` | SMS, email, WhatsApp | text, and a subject for email |
| `link` | X, Facebook, LinkedIn | text with the URL in it |
| `story` | Instagram, Snapchat | **an image**, with a link sticker |

**Stories cannot take a link.** They are pictures. Without a card the listener
shares a screenshot of a player UI, which is not an invitation to anything — so
`sharing.story_card` renders one as SVG at 1080×1920, in FAM's own colours,
with no dependency and nothing stored. The question in it is text a listener
typed and the card is markup, so everything is escaped; the test for that is
the one that would otherwise fail in public.

The templates are **defaults the listener edits**, not copy that gets posted
unseen — which is why they are written to be finished by somebody. Each says
what it is about, that it is short, and where to hear it. None claims the
episode is good: FAM did not write that opinion and the person sharing has not
typed one yet. A test enforces it.

X's caption is trimmed to fit, on a word boundary, and **the link is never the
part that gets cut** — a share with a truncated URL is worse than one with a
shorter sentence.

Without `PUBLIC_BASE_URL` the link comes back relative, `public` is `false`,
and the share sheet says so. Nothing invents a host: a link resolving to
`localhost` posted to LinkedIn is exactly the quiet failure this project keeps
a rule about.

## Save for later, and download

The distinction is the whole design, and both look like a bookmark from
outside:

|  | Save for later | Download |
|---|---|---|
| what it is | a pointer: question, length, title, folder | the audio, on the device |
| costs | one row | a slot, and the phone's storage |
| plays offline | **no** | **yes** |
| limit | none worth having | per tier, and it bites |

A download is an **upgrade to** a saved item rather than a separate list. That
is why the interface asks the question the moment something is saved, and why
one row carries both states — two lists would put the same episode in two
places and make removing it from one of them ambiguous.

### Where the audio lives, and why "no MP3, no audio files" survives

The settled constraint is that raw PCM streams from the engine and is played as
it arrives; writing a *file* is not compatible with that.

A download does not break it, because **the server still writes nothing.** The
episode streams exactly as it always does and the client keeps the bytes it was
already sent — IndexedDB in the browser, the app's own container on iOS. No
file is created server-side, no audio is cached, and no URL serves a stored
episode. What changes is only that the listener's device stops throwing the
samples away.

Two consequences, and both matter:

1. **`saved.py` holds a registry, not audio.** It records that a listener
   claims to hold an episode, so the limit can be enforced and the list shown.
2. **The registry can drift.** A wiped phone or an evicted browser store still
   has rows. So `release` exists, a client re-syncs by releasing what it no
   longer holds, and the count is *what the listener claimed* rather than
   ground truth. Drift costs a slot, which is why the fix is one tap.

Uncompressed PCM is 2.65 MB/minute, so three minutes is ~8 MB and ten of them
is 80 MB — fine on a phone, heavy in a browser. One more argument for Opus over
the stream, which `IOS_APP.md` already has as a prerequisite of the app rather
than a scale question.

### The limit

Per tier, from `entitlements.max_downloads`: **3 / 25 / no server-side cap**.
A *standing capacity*, not a rate, which is why it is not in `quotas.py` — a
windowed counter would hand out a fresh download allowance every morning and
never require anybody to delete anything, which is the opposite of what a shelf
limit is for.

A full shelf is a **409** (a capacity, not a rate) and the refusal **names what
to clear**, offering the least recently played first — not the oldest, because
the episode somebody saved first is often the one they are keeping on purpose.
A limit without a remedy is a dead end, especially on a phone where the
listener cannot go and look somewhere else.

The size is estimated *before* the listener agrees to it, generously: an
episode ends when it runs out of substance, so the real size is usually
smaller, and being told 8 MB and charged 6 is the right direction to be wrong
in. The client confirms the real figure afterwards.

Deleting a folder **unfiles its episodes rather than deleting them**. Losing
somebody's saved episodes because they tidied up is the kind of surprise that
stops people using a feature — and a download inside it would become bytes on
their phone held against their limit with nothing pointing at them.

## In the interface

* **Messages**: the Explore New tile became **Save for Later**, as asked.
  Explore New moved to **myFAM**, where it is now a rail of its own between
  "Made for you" and the crowd — found rather than remembered. That is a better
  home than the tile it lost: it is the only surface offering anything outside
  an established taste, and the interim row under the tiles that kept it
  reachable is gone now that it has a real place.
* **Every player** — search, play-all and Explore — has **share** and **save**.
  Explore included, so the surface where people find things is not the one
  where they cannot keep them.
* **Saving opens the download question**, which says the size, what downloading
  buys ("plays with no signal") and how much room is left. Declining is one tap
  and the episode stays saved.
* **Offline playback goes through the same player** — the same progress bar,
  transport and speed control, driven by `FamAudio.playStored`. A second player
  for offline episodes would be a second player to keep in step, and the two
  would drift.

## What is not built

* **Group threads**, and the decision about what a share to a group means.
* **A share opening the app on a phone.** `/s/<id>` redirects into the web app;
  universal links and an App Store fallback are app-side work.
* **Notifications.** A share arrives silently until somebody opens Messages.
  Push is an App Store capability and a permission prompt, and it belongs with
  the app rather than ahead of it.
* **Service-worker offline for the browser.** Downloads are real in the web
  build — the bytes are in IndexedDB and play from there — but the *page*
  itself still needs the network to load. On iOS that problem does not exist.
