# Phase 4.5 — Obsidian second brain

## 1. Goal

Everything Jarvis knows becomes a folder of Markdown pages you can open in Obsidian:
read it, search it, link it, edit it, and your edits flow back into Jarvis. Jarvis
also searches the notes **you** write there, so "what did I write about the Exinent
valuation?" just works.

Vault: `C:\Users\ashvi\OneDrive\Documents\obsidian_notes` (`[vault] path =
"Documents/obsidian_notes"`, OneDrive-aware). It syncs every 3 s while
`jarvis chat`, `jarvis voice` or `jarvis serve` runs.

## 2. Layout

```
obsidian_notes/
  Home.md                ← start page: today, open reminders, people, projects, recent chats
  About me.md            ← who you are + preferences (sections: Who I am, Preferences, …)
  People/Priya.md        ← one page per person
  Projects/Exinent sale.md
  Topics/Dividend investing.md
  Notes/Plumber is Ravi.md   ← Jarvis's notes (secrets shown as 🔒) + your own
  Journal/2026/2026-10-03.md ← per day: conversations, what Jarvis learned, reminders
  Conversations/2026-10/2026-10-03 14.32 Planning Diwali.md
  Reminders.md           ← checklist
  Inbox/                 ← drop files here; Jarvis learns from them, files them in Notes/
```

Folders tell Jarvis what a page is about (People → person, Projects → project,
Topics → other); links tie it together (a person's page lists the conversations
that mention them; daily pages link to everything from that day).

## 3. A page

```markdown
---
type: person
jarvis: "page:People/Priya.md"     ← which Jarvis record this file shows (don't edit)
aliases: [my sister]               ← yours — kept
---
# Priya
%% jarvis:start — Jarvis keeps this part in sync. Edit, add or delete lines freely. %%
- User's sister is named Priya. ^m12
- Priya is a doctor in Pune. ^m15

### Mentioned in
- [[Conversations/2026-10/…|03 Oct 2026 — Planning Diwali]]
%% jarvis:end %%

## My notes
(anything you write here — Jarvis reads it for search, never rewrites it)
```

`%% … %%` is an Obsidian comment (hidden in reading view). `^m12` ties the line to
memory #12.

## 4. What your edits do

| You… | Jarvis… |
|---|---|
| change a fact line | updates that memory |
| add a line in Jarvis's part of a person/project/topic page or About me | learns it as a new fact about that page (About me: the section decides the kind) |
| delete a fact line | forgets it (after 10 s, in case you're mid-edit) |
| move a line to another page | moves the fact (new subject) |
| rename / move a page (e.g. into `People/Family/`) | follows it; renaming changes the subject name |
| delete a whole person/project page | **keeps the facts** and writes the page again a minute later — delete the *lines* to make Jarvis forget |
| tick `- [x]` in Reminders | marks it done · delete the line → cancelled · add `- [ ] Pay rent 📅 2026-11-01 9am` → new reminder (no date → tomorrow 9:00) |
| edit a note's text | updates the note · delete the note's file → note deleted (after a minute) |
| write anywhere outside the markers | nothing — it's yours; Jarvis keeps it and searches it |
| drop a file in Inbox/ | it's moved to Notes/ (untouched) and Jarvis learns the facts in it |

Conversations, Journal and Home are written by Jarvis only (edits inside their
markers are replaced; write below them instead). Deleting a conversation or day page
removes it from the vault for good (the conversation stays in Jarvis).

## 5. Safety rules

- Your edits are read **before** anything is written; a file you edited in the last
  4 s (or that can't be read — OneDrive, a lock) is left completely alone.
- A page with an edit Jarvis hasn't read yet is never overwritten.
- Facts are only forgotten when their lines are deleted from a page that still
  exists. A missing page, an unreadable file or a vault that looks unavailable
  (no Home / About me) never makes Jarvis forget anything.
- A deleted note/page is only treated as deleted after a minute.
- Jarvis never writes over a file it didn't create (a name clash gets "(2)"), merges
  into a page you created yourself (e.g. your own `People/Mom.md`), and never deletes
  a page with anything of yours on it (text, your properties).
- OneDrive conflict copies (`Priya-DESKTOP-1.md`) are ignored.
- **Secrets** (notes that look like passwords, PINs, codes) are never written to the
  vault — shown as "🔒 … ask Jarvis".
- Text Jarvis can't understand inside its part (tables, your lines under
  "Mentioned in") is moved just below the markers, never dropped.

## 6. Components

```
src/jarvis/vault/
  markdown.py   frontmatter (keeps your keys), page = before / Jarvis part / after, fact lines
  pages.py      what each page should contain (pure, from the database)
  sync.py       VaultSync: read edits → inbox → write pages → index; VaultWorker thread
  index.py      VaultIndex: chunk + embed your notes; search by meaning (+ keywords)
src/jarvis/tools/vault.py   vault_search tool
```

- Memories now have a **subject** (migration 4): the person/project/topic they're
  about (null = you). The learner picks it, reusing known names; `memory_save` takes
  it. Older memories are sorted into pages once, by the chat model.
- Duplicate-checking never merges facts about different subjects.
- `vault_files` records what Jarvis last wrote to each file (to spot your edits);
  `vault_chunks` holds the search index.

## 7. Commands & config

`jarvis vault` (status) · `jarvis vault sync` · `jarvis vault search TEXT` ·
`jarvis vault rebuild` (re-sort memories into pages) · `jarvis vault open` (Obsidian).

`[vault]`: `enabled`, `path`, `poll_s`, `quiet_s`, `conversations`, `journal`,
`inbox`, `search`, `search_conversations`, `user_name` (e.g. `"Adi"`).

## 8. Done when

Automated (39 vault tests): layout, idempotent sync, your notes/properties kept, edit/add/
delete/move facts, About me sections, page rename & move, vault disappearing,
grace period, unreadable files, busy files per page, markers removed, notes edit/
delete/secret, secret detection, reminders tick/delete/add/9am, failed writes don't
duplicate, deleted pages stay deleted, inbox (learn + move, locked, non-UTF-8), search,
worker thread, subjects (learner + one-time sort), conflict copies, name clashes,
lowercase renames, same fact on two pages.

On your PC:
- [ ] `uv run jarvis vault sync` → pages appear in Obsidian; Home links work
- [ ] Your existing memories are sorted into People / Projects / Topics
- [ ] Edit a fact in a person's page → `jarvis memory` shows the change
- [ ] Add a line to About me → Preferences → Jarvis knows it in a new chat
- [ ] Delete a fact line → gone from `jarvis memory` after ~10 s
- [ ] Tick a reminder → `jarvis reminders` no longer lists it
- [ ] Write a note of your own, then ask Jarvis about it → it uses vault_search
- [ ] Drop a .md in Inbox/ while chat runs → "📓 Filed from Inbox", facts learned
- [ ] A garage-code style note shows 🔒 in the vault
