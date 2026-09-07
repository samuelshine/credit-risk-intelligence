# Frontend Design

## Direction: the underwriting desk

The subject is a credit file — an application form, a bureau report, a
repayment ledger, a policy memo. The UI is built to feel like that physical
object made digital: a document you work through, not a SaaS dashboard
bolted onto a model. Deliberately not the generic AI-tool look (warm
cream/terracotta, or near-black with one neon accent), and not a card grid —
this is one continuous working surface with five sections, matching the
assignment's own five-part structure (EDA, prediction, explainability, rules,
chatbot) rather than hiding it behind abstraction.

## Colour

| Token | Hex | Role |
|---|---|---|
| `--paper` | `#F6F8F9` | Page background — cool blue-grey, not cream |
| `--paper-raised` | `#FFFFFF` | Panels, the data strips |
| `--ink` | `#14232E` | Primary text, headings |
| `--slate` | `#5A6E7A` | Secondary text, captions, axis labels |
| `--hairline` | `#C9D4DB` | Rules, borders, chart gridlines |
| `--accent` | `#2F5D8A` | Links, active nav, primary actions |
| `--risk-low` | `#2F6F6A` | Teal — low risk |
| `--risk-medium` | `#B8863B` | Ochre — medium risk |
| `--risk-high` | `#9E2B3E` | Oxblood — high risk |

A three-stop risk scale rather than traffic-light red/yellow/green — desaturated,
dyed tones that read as a print document's ink, not a warning system. Dark
mode inverts the paper/ink relationship (`--paper: #12191F`, `--ink: #E8EDF0`)
and keeps the risk hues, slightly brightened for contrast.

## Type

- **Newsreader** (serif) for section openers and the risk-ruler's numerals —
  a document typeface, not a display face doing tricks.
- **IBM Plex Sans** for UI chrome, body copy, form labels, and tabular
  figures (`font-variant-numeric: tabular-nums` on every number that appears
  in a column).

Both loaded from Google Fonts with a full system fallback stack
(`Newsreader, Georgia, 'Times New Roman', serif` /
`'IBM Plex Sans', -apple-system, 'Segoe UI', sans-serif`), so the page
renders correctly even with no network.

## Layout

```
┌──────────┬──────────────────────────────────────────┐
│          │  The Portfolio                            │
│  ● Port. │  ─────────────────────────                │
│    Score │  Prose column, ~68ch, left-aligned.        │
│    Why   │                                             │
│    Rules │  ┌─────────────────────────────────────┐  │
│    Ask   │  │  full-bleed data strip: chart, table │  │
│          │  └─────────────────────────────────────┘  │
│          │  More prose, back to 68ch.                 │
└──────────┴──────────────────────────────────────────┘
```

Fixed left rail (five sections, always visible — this is a one-page working
tool, not a site with a hamburger menu). Content is a single reading column
capped at ~68ch for prose; charts, tables and the risk ruler break out to the
full content width, so the page alternates between "reading" and "looking,"
which is how an actual credit memo is laid out. Left-aligned throughout — no
centred hero, no marketing-page symmetry.

## The one memorable element: the risk ruler

A horizontal scale showing the real portfolio's score distribution (from
`/api/eda`, live data) with the current applicant's score marked on it. Reused
across three sections — the score result, the risk-band explanation, and
context for a rule's population share — so a viewer learns to read it once.
The only animation in the app: the mark travels into place when a new score
returns, eased over ~400ms. Nothing else moves.

## What the sections say, in plain language

- **The portfolio** — dataset summary, data quality, the 8 business insights, each framed as a finding with a number and a "so what."
- **Score an applicant** — pick a real applicant or enter details by hand; get a probability, a band, and the risk ruler.
- **Why this score** — the same applicant's SHAP factors as plain sentences and a push/pull bar, not a bare technical chart.
- **Policy rules** — the surrogate tree's rules as IF/THEN sentences a credit officer could put in a memo, each with its measured population share and lift.
- **Ask the data** — the chatbot: a question box, the SQL it ran, and the grounded answer, always shown together so the answer is auditable.

## Anti-patterns deliberately avoided

No tracked-out ALL-CAPS eyebrows, no middot-joined meta strings, no
monospace data labels, no arrow-suffixed buttons, no identical rounded cards
with the same drop shadow, no gradient decoration. Structural devices (rules,
numbering, the risk ruler) are used because the content is structured that
way, not decoratively.
