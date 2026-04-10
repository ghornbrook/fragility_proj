# Financial Fragility Project

## What this is

A research project to see whether alternative data sources — GitHub activity, job postings, Glassdoor ratings, and similar — can predict when a public company is about to become financially fragile, before traditional financial metrics catch it.

Think of it as augmenting the Altman Z-score with signals that didn't exist when Altman wrote his paper in 1968.

The final deliverable is a working real-time alert system — a public dashboard with a fragility leaderboard for every S&P 500 company, plus an email alert system that fires when signals trip. The research phases (1-6) build the components; Phase 7 wires them together into a tool. That framing matters: if you keep the end state in mind, every decision in the earlier phases gets easier. ("Will I need this in the live system?" is a great filter for cutting scope.)

## Research question

> Traditional bankruptcy and distress models (Altman Z-score, Beneish M-Score) rely on quarterly financial statements that arrive 30-135 days after the period they describe. Can alternative data signals — SEC filings, disclosure patterns, and behavioral indicators — provide *earlier warning* of fragility events than these quarterly baselines, and can they close the time gap between filings by updating in near-real-time?

> The goal is not to replace Z-score and M-Score. Those models work — they're just slow. The goal is to build a real-time overlay that catches fragility events earlier, fills the gap between quarterly filings, and flags behavioral signals that accounting numbers only reflect with a lag.

### Why this framing matters

It would be tempting to frame this project as "beat the old models with alt-data." Don't. That framing is wrong for three reasons:

1. **Z-score isn't wrong, it's slow.** Altman's model has been refined over 60 years of academic research. The accounting math is good. The problem isn't accuracy; it's latency. Trying to beat Z-score on its own terms is fighting the wrong battle.
2. **Professional quant funds don't replace fundamental models — they overlay on them.** Two Sigma, Citadel, Renaissance, and Point72 all use alt-data as overlays on top of fundamental analysis, not as replacements. This project follows the professional playbook, not the naive one.
3. **Lead time is more valuable than marginal accuracy.** A signal that fires 60 days earlier than Z-score with slightly worse precision is *better* than a signal that matches Z-score with slightly better precision. For a fragility warning system, "early and roughly right" beats "late and precisely right." Price this into every decision you make about what to build.

The right mental model: the quarterly baselines are the slow, high-precision foundation. Alt-data signals are fast, noisier overlays that fill the time gap and catch behavioral leading indicators. The real deliverable is the *composite*.

### Who this tool is for

Every research project has an implicit customer — someone who would actually use the output. Naming that customer sharpens every design decision downstream, because it tells you what success looks like from the user's perspective, not just the researcher's.

**This is a short-activist tool.** Not in the literal sense of "helps you short stocks" — in the sense that the framing, success metrics, and design choices all reflect how short-sellers think about risk. The obsession with lead time over marginal precision (see above) is pure short-seller discipline: shorts are paid for being *right and on time*, not just right. A long investor can be patient. A short investor cannot. Your design choices inherit that mentality whether they're made consciously or not, so they should be made consciously.

**The downstream users are functionally short-side decision-makers**, even if they never enter a short position:

- **Lenders** deciding whether to roll a credit facility, extend new debt, or tighten covenants. Early warning of borrower distress is worth enormous money to them.
- **Suppliers** deciding whether to extend trade credit, demand prepayment, or require letters of credit. A fragile customer is an accounts-receivable write-off waiting to happen.
- **Board members** — especially audit committee chairs and lead directors — deciding whether to push harder on management, replace a CFO, or commission an independent investigation. The earlier they see signals, the more options they have.
- **Auditors** deciding whether to issue a going-concern opinion, expand audit scope, or resign. Their professional liability depends on catching problems early.
- **Institutional investors** on the long side who want to *avoid* fragile names, not short them. The same signal that tells a short-seller when to enter tells a long-only PM when to exit.
- **Credit rating analysts** doing surveillance on outstanding ratings.
- **Short-sellers themselves**, both activist (Hindenburg, Muddy Waters) and quantitative (funds that systematically short companies with deteriorating fundamentals).

All of these users share the same underlying question: *is this company deteriorating in ways the quarterly financials haven't yet revealed, and how much warning can I get before the next adverse event?* That's the question this tool answers.

**What this means concretely for the project:**

1. **Earlier is better than more accurate.** A signal that fires 60 days earlier with 80% precision beats a signal that fires on time with 90% precision. This is the opposite of how academic ML papers are usually framed, which is why it's worth naming explicitly.
2. **False positives are tolerable; false negatives are catastrophic.** A lender who gets a warning on a healthy company can investigate and ignore the alert. A lender who gets no warning on a sick company loses their money. The asymmetry is severe, and it should shape the alert thresholds in Phase 7.
3. **The audience for the final write-up is not an academic peer reviewer.** It's someone who might actually use this to make a decision. Write accordingly — plain language, clear caveats, honest limitations, and a frank discussion of what the tool is and isn't good at. Claims should be defensible in front of someone whose money is on the line, not just in front of a professor.

**What this tool is NOT for:**

- **Long-side stock picking for value plays.** Value investors look for *underpriced* companies, which requires a totally different signal set (capital allocation quality, moat durability, management skill). Fragility prediction doesn't help them find winners.
- **Activist long campaigns.** Activist longs look for underperforming companies where management change can unlock value. Those companies are often *not* fragile — they're inefficient, which is a completely different condition.
- **Sentiment-driven trading.** This is a slow-moving risk tool, not a high-frequency momentum signal.

Being clear about what the tool is not for is as important as being clear about what it is for. Scope discipline on the *research question itself* is what keeps the project from sprawling.

## Working definition of fragility

**Fragility event = 40%+ drop in market cap over any 180-day window, not recovered within 90 days.**

**Why this definition, and why now:** Machine learning needs a target variable — something concrete for the model to learn to predict. "Financial distress" is too vague to code against. A specific number is codeable. The 40% / 180 / 90 choice is a guess; the point is to have *any* working definition so you can start. Once you have price data for the full universe, you'll look at the actual distribution of drawdowns and revise. The definition gets sharper as the data tells you more.

**The habit this builds:** always have a working answer, even if you know it's wrong. A wrong answer you can improve beats a perfect answer you never start on. Real research progresses by having something running end-to-end and then making it better, not by planning until everything is perfect.

## Universe

S&P 500, point-in-time, from January 2017 to today.

**Why point-in-time, and why this matters more than anything else in the project:** If you use today's S&P 500 and backtest to 2017, you'll only be studying companies that *survived*. The bankruptcies, forced removals, and acquisitions — the most interesting data points for a fragility study — would be missing. This is called **survivorship bias**, and it would quietly invalidate the entire project. You wouldn't see anything wrong in your results; they'd just be wrong. This is the single biggest methodological risk in the project, which is why it's the first thing you build.

You have two files to build the universe:
1. Current S&P 500 membership (today's snapshot)
2. Historical changes (adds and removes, by date)

**First task:** open both files, verify the changes file goes back to at least January 2017, and note any gaps. If there's a gap, flag it to Mom before proceeding. Don't start coding until you've confirmed the raw data covers the window you need — there's no point building on a foundation that's missing pieces.

## Tech stack

- **Python** via `uv` (modern package manager — faster and simpler than pip/venv/poetry)
- **Jupyter notebooks** for exploration
- **pandas** for data manipulation
- **yfinance** for historical price data
- **matplotlib** or **plotly** for charts

**Why Jupyter if you've never used it:** a notebook is a file where you run Python one chunk at a time and see the output immediately — a lab notebook for code. You try something, see what happens, try the next thing. It's designed for exploration, where you don't know what you're looking for yet. Production code eventually moves into `.py` files in `src/`, because notebooks are bad for reusable logic. Rule of thumb: notebooks are for figuring things out; modules are for code you'll call again.

## Project structure

```
financial-fragility/
├── README.md              # this file — update it as decisions get made
├── pyproject.toml         # uv manages this — lists your dependencies
├── data/
│   ├── raw/               # source files, never edited, gitignored
│   └── processed/         # cleaned outputs from your code
├── notebooks/
│   ├── 01_build_universe.ipynb
│   ├── 02_price_data.ipynb
│   ├── 03_define_fragility.ipynb
│   ├── 04_baseline_zscore.ipynb
│   └── 05_altdata_github.ipynb
├── src/
│   └── fragility/
│       ├── universe.py    # S&P 500 historical membership
│       ├── prices.py      # yfinance wrappers
│       └── labels.py      # fragility event detection
└── .gitignore             # tells git what NOT to commit (raw data, secrets, etc.)
```

**Why this structure:** notebooks are numbered in the order you work through them, so anyone (including future you) can figure out how to reproduce the project. Each notebook's output feeds the next. The split between `notebooks/` and `src/` matters because the moment you find yourself copy-pasting a function between notebooks, it belongs in `src/` so there's only one copy to fix when it's wrong.

## Phases

### Phase 1 — Environment setup

**What:** Set up the project with `uv`, get Jupyter running, make a hello-world notebook that imports pandas and yfinance and plots something trivial.

**Why:** Before you do anything real, prove the tools work. The most frustrating bug in data work is when you can't tell whether your code is wrong or your environment is broken. Eliminate the environment as a suspect on day one. Every minute spent here saves an hour later.

**Done when:** a notebook runs, imports work, a chart renders, and the whole thing is committed to GitHub.

### Phase 2 — Build the universe

**What:** Upload the two S&P files to Claude. Build a function that takes a date and returns the list of tickers in the S&P 500 on that date.

**Why:** Everything else in the project depends on knowing *who was in the index when*. Get this wrong and every downstream result is wrong. This is also the phase where survivorship bias gets handled — once this function works correctly, you can stop worrying about it.

**Validation tests to write alongside the code** (ask Claude to include these):
- How many companies were in the S&P 500 on 2020-01-01? Should be ~505 — the index sometimes has slightly more than 500 because of dual-class shares like GOOGL/GOOG.
- Was Tesla in the index on 2020-06-01? No — Tesla was added in December 2020. If your function says yes, something's wrong.
- Which companies were removed in March 2020? COVID crash era — easy to cross-check against news articles.

**Done when:** the function works, the validation tests pass, and you've committed both the code and a short note in the README explaining how the function works.

### Phase 3 — Pull price data

**What:** For every company that was ever in the universe during 2017-2026, pull daily price and market cap data via `yfinance`.

**Why:** You can't compute drawdowns without prices, and you can't compute market-cap-based fragility events without market caps. This is the foundational dataset everything else is built on.

**What to expect:** this will be annoying. Delisted tickers (companies that went bankrupt or got acquired) are often missing or broken in `yfinance`. Ticker changes — Facebook became META, Square became Block, Weight Watchers became WW — will trip you up. Some tickers get reused for different companies over time. Expect to spend real time cleaning this, and expect to discover new problems after you think you're done. This is 80% of real data science work and it's never taught in class. Embrace the mess and keep good notes on what you fix.

**Done when:** you have a clean dataframe of daily prices and market caps for every historical S&P 500 member, with gaps documented, and it's committed to GitHub.

### Phase 4 — Label fragility events

**What:** Apply the working fragility definition (40% / 180 days / 90 days non-recovery) to the price data. Produce a list of every historical fragility event in the universe.

**Why:** Without labeled events, there's nothing to predict and nothing to measure against. This is the dataset that turns the project from "playing with data" into "testing a hypothesis."

**The important moment in this phase:** once you have the events labeled, look at the distribution. How many per year? Which sectors show up most? What does a typical drawdown shape look like? **This is where you revisit the 40% / 180 / 90 definition** based on what the data shows. If there are 500 events, the threshold is too loose. If there are 3, it's too tight. Update the definition, re-run, and log the decision in the Decisions section of this README.

**Done when:** events are labeled, the distribution has been examined, the definition has been revised (or explicitly confirmed) based on what you saw, and the decision is logged.

### Phase 5 — Baselines with Z-score and M-score

**What:** Compute two academic baseline models for every company in the universe, quarterly. Backtest both against the fragility events you labeled in Phase 4. Measure each properly with precision, recall, and lead time.

1. **Altman Z-score (1968).** The standard bankruptcy prediction model. Uses five financial ratios covering working capital, retained earnings, operating profitability, market value, and sales-to-assets.
2. **Beneish M-Score (1999).** An earnings manipulation detection model. Uses eight ratios covering receivables growth, gross margin deterioration, leverage changes, and accruals quality. Historical footnote worth knowing: the Beneish M-Score famously flagged Enron as a probable earnings manipulator in 1998, three years before the collapse. The model wasn't designed to predict bankruptcy — only to detect earnings manipulation — but the manipulation *was* the early warning. That's a useful intuition for this project: the behaviors that precede fragility often show up years before the financial collapse, if you know what to look for.

**Why two baselines instead of one:** Z-score and M-Score answer related but distinct research questions. Z-score predicts bankruptcy — the end state of fragility. M-Score detects earnings manipulation — the management behavior that often precedes distress because stressed companies start bending accounting rules before they break operationally. A signal can beat one baseline and not the other, and that difference is itself informative. Having two baselines also makes you a better researcher: it forces you to ask *which* baseline a new signal is beating and *why*, rather than treating "baseline" as a single fuzzy concept.

**Why this matters for Phase 6:** When you test an alt-data signal, "beats the baseline" is a vague claim unless you specify which one. A signal that improves on Z-score but not M-Score is catching something about the path to distress but not the manipulation that precedes it. A signal that improves on M-Score but not Z-score is catching early warning but not the severity of the end state. A signal that beats both is rare and worth writing up carefully.

**A note on effort:** M-Score is trivially easy to compute — it's eight financial ratios and a linear formula, published in the original 1999 paper. Adding it alongside Z-score is maybe a day of work once the financial data is loaded. The return on that day is doubling the analytical power of everything in Phase 6.

**Done when:** both scores are computed for every company-quarter, backtest reports exist for each (precision, recall, lead time), and you have two clear numbers to beat in Phase 6.

### Phase 6 — Alt-data experiments

> **Before you start coding this phase, read 3-4 short-seller research reports end to end.** Short-sellers have been running exactly your research question — "which public companies are about to become fragile" — with real money on the line for decades. Their public reports are some of the best available literature on fragility prediction, and most are free. Start with: Hindenburg Research on Nikola (2020), Hindenburg on Lordstown Motors (2021), Muddy Waters on Luckin Coffee (2020), and Kerrisdale Capital on any recent target. The goal isn't to replicate their work — it's to build intuition for what fragility actually looks like in real cases before you try to detect it algorithmically. Pattern recognition from real examples is what separates a good model from a good-looking model. Budget an afternoon for this; it will pay for itself many times over in Phase 6.

**What:** Test alt-data signals for their ability to predict the fragility events labeled in Phase 4, and measure whether they add anything *beyond* the two baselines from Phase 5. Start with SEC-derived signals (structured, free, legal), then move to behavioral and external data sources.

**Five SEC signals to test, in order:**

1. **Auditor changes (8-K Item 4.01).** When a company fires its auditor or an auditor resigns, they must disclose it in an 8-K filing within four business days. Auditor resignations in the 12-24 months before a fragility event are a well-documented predictor. The data source is EDGAR (SEC's free filing database), which has a structured API. This is the first signal to build because it teaches you the full EDGAR pipeline — once you can pull and parse 8-K filings, everything else in this list is incremental.

2. **10-K risk factor section growth.** The SEC requires companies to list material risks in their annual 10-K filing. Companies facing deteriorating conditions often quietly add risk factors, expand existing ones, or introduce new categories of language in the year or two before a fragility event. A sudden increase in the word count or new-topic count of the risk factors section is a documented signal. This runs on the same EDGAR pipeline as the auditor-change signal — just 10-K filings instead of 8-Ks, and text parsing instead of item-code filtering.

3. **Loughran-McDonald sentiment on MD&A.** The Management Discussion & Analysis section of the 10-K is where management explains the business in their own words. The Loughran-McDonald financial sentiment dictionary (published 2011, freely available) is a word list specifically designed for financial text — words like "deteriorated," "uncertainty," "adverse," "litigation" carry different weight in financial filings than in general text. A rising negative sentiment score in the MD&A in the two years before a fragility event is a signal. Same pipeline, different section and parser.

4. **Executive departures, especially CFO, Controller, and General Counsel.** Unplanned CFO departures in the 12 months before a fragility event are one of the strongest signals in the academic literature — finance chiefs tend to leave when they don't want their name on the next 10-K. Companies must disclose executive departures in an 8-K filing (Item 5.02), so this runs on the same EDGAR pipeline you built for the auditor-change signal. Implementation is a filter on 8-K filings for Item 5.02 plus some text parsing to extract which executive left and whether the departure was described as voluntary, retirement, or "to pursue other opportunities" (a well-known euphemism worth flagging). Pair this signal with auditor changes — together they form a "bad news forced disclosure" composite that has been shown in the literature to outperform either signal alone.

5. **Restatements (8-K Item 4.02 — "Non-reliance on previously issued financial statements").** When a company files a 4.02, they're telling the SEC and investors that their previously reported financials can't be trusted and should not be relied on. This is the single strongest forced-disclosure red flag in accounting — short-sellers treat it as a near-automatic signal. Restatements in the 24 months before a fragility event are a well-documented predictor in the academic literature. It runs on the exact same EDGAR pipeline as the auditor-change and executive-departure signals, just with a different 8-K item code (4.02 instead of 4.01 or 5.02). Once the 8-K infrastructure exists, adding this is near-zero marginal effort and high marginal signal. The three 8-K-based signals together — auditor changes, executive departures, restatements — form a "forced bad-news disclosure" composite that should be much stronger than any one alone.

**After SEC filings, in rough order of priority:**

- GitHub commit activity for public-company repos (free API, clean data, good for learning the experiment loop before tackling harder sources).
- Job postings volume and category shifts (messier — scraping or paid APIs, but a strong leading indicator of hiring freezes and cost-cutting).
- Glassdoor ratings trajectory (hardest — anti-scraping, limited data, but management deterioration often shows up in employee reviews before it shows up in financials).

**Why this order:** SEC signals come first because they're free, structured, legally unambiguous, and already documented in the academic literature. They're also the closest thing to a "ground truth" behavioral signal — companies are *required* to disclose these events, so there's no scraping, no API limits, and no question about data quality. Once the EDGAR pipeline exists, each new SEC signal is incremental. The alt-data sources (GitHub, job postings, Glassdoor) are tested after because they're messier and the methodology should be proven on clean data first.

**The discipline in this phase:** for each signal, the test isn't just "does it correlate with fragility." The three real tests are:

1. **Lead time** — does this signal fire earlier than Z-score and M-Score on the same events? Measure in days. This is the primary success metric for this project, because the whole point of alt-data is closing the time gap.
2. **Update frequency** — how often does this signal refresh? Daily, weekly, monthly? A daily signal with modest predictive power is often more useful than a quarterly signal with higher predictive power, because the daily signal gives you something new to look at 90 times as often.
3. **Marginal information** — does this signal catch events the baselines miss entirely, or does it just re-predict the same events the baselines already caught? A signal that catches 3 events nobody else caught is more valuable than a signal that catches 30 events the baselines also caught.

A signal that beats the baselines on any of these three — earlier, more frequent, or catching missed events — is a success. A signal that matches baselines on all three is not adding anything. Log each result in the Decisions section with numbers for all three dimensions.

**Done when:** you have at least one alt-data signal tested end-to-end with a clear verdict (beats baseline / doesn't beat baseline / inconclusive and here's why), with numbers for lead time, update frequency, and marginal information logged in the Decisions section.

### Phase 7 — Real-time alert system (the capstone)

**What:** Wrap the whole pipeline in a scheduled system that runs automatically, computes all signals on current data, combines them into a composite fragility score per company, and fires alerts when conditions warrant. Expose it through a simple dashboard anyone can view.

**Why:** Everything up to this point has been backward-looking — running historical data through models to test whether signals work. The project framing in Phase 6 (lead time, real-time overlays, closing the gap between quarterly filings) only makes sense if there's actually a real-time system watching for the signals as they happen. Phase 7 is what turns the research into a working tool. It's also what turns the project into something you can show someone — a working dashboard with a public URL is 10x more impressive than a notebook.

**Architecture — start simple, keep it free:**

- **Scheduler:** GitHub Actions, running on a daily cron. Free, no server to maintain, runs in the same repo as the code. When the job runs, it pulls new EDGAR filings, recomputes signals, updates the composite score, and writes alerts to the database. (GitHub Actions gives you 2,000 free minutes per month — more than enough for a daily job.)
- **Storage:** SQLite database. No database server to run, no cloud setup, just a file. Three tables to start: `signals` (every signal computation per company per date), `alerts` (every alert fired with timestamp and severity), `companies` (universe metadata).
- **Dashboard:** Streamlit. It's a Python-first web framework designed for data scientists who don't want to learn HTML/CSS/JavaScript. You write Python, it becomes a web app. Streamlit Community Cloud hosts it for free with a public URL.
- **Notification:** email via Gmail SMTP for critical alerts (free, Python-native, 10 lines of code). Dashboard for browsing everything else.

Total cost: zero. Total infrastructure to maintain: zero. This matters — a system that's expensive or annoying to run will quietly get turned off.

**Alert design — the conceptual stuff that matters most:**

The code for an alert system is easy. The hard part is tuning it so it's useful instead of annoying. Four things to get right:

1. **Severity tiers.** Not all signals are equal. Use at least three levels:
   - **Critical** — composite score crosses a high threshold, *or* a single high-signal event fires (restatement, going-concern language, unexpected CFO departure, auditor resignation). Triggers an email. Fires rarely.
   - **Warning** — composite score is elevated, or 2+ moderate signals fired within 30 days. Dashboard only, no email.
   - **Info** — any individual signal fired. Dashboard only, easy to filter out.

2. **Alert fatigue is the #1 failure mode of monitoring systems.** An alert system that fires too often becomes noise and gets ignored. This is how every production monitoring system in industry eventually dies. Rule of thumb: if you're getting more than 5 critical alerts per week from an S&P 500 fragility monitor, your thresholds are too loose. Tune tightly. It's better to miss a few events than to fire on everything and train yourself to ignore the alerts.

3. **Dead man's switch.** The system should alert you when *it* stops working, not just when it detects fragility. "I haven't received an alert in two weeks" could mean "nothing is fragile" or "the pipeline has been broken for two weeks." You need to know which. Build a weekly "system status" email that says something like *"Pipeline healthy. 500 companies scored. 3 alerts fired this week. Last EDGAR pull: [timestamp]."* Silence from a broken system should never look like silence from a calm market.

4. **De-duplication.** If the same company trips the same alert every day for a week, that's one alert, not seven. Track which alerts have already fired and suppress repeats within a rolling window (7-14 days is reasonable). Without this, one bad company will drown the whole inbox.

**Dashboard design — what to show:**

Think of this as a *fragility leaderboard*. The landing page should be a single sortable table:

| Company | Composite Score | Change (7d) | Z-Score | M-Score | Active Signals | Last Alert |
|---------|-----------------|-------------|---------|---------|----------------|------------|

Color-code the composite score (green/yellow/red). Click any row to see a detail page with that company's signal history over time — a multi-line chart showing Z-score, M-Score, and the composite on the same axes, with vertical markers for each alert that fired. The key visual is the **lead time comparison**: show where alt-data signals fired *before* the Z-score would have moved, if they did. That's the whole point of the project, made visible.

Keep the design simple. No login, no accounts. A shareable URL anyone can view.

**Validation — how you know it's working:**

Before you trust the alert system on live data, run it in **backtest mode** on historical data. Pretend it's running on January 1, 2018, feed it historical data one day at a time, and see what alerts it would have fired. It should catch the major fragility events you labeled in Phase 4 — and it should catch them earlier than the Z-score and M-Score baselines did. If it doesn't, the alert logic is broken, and deploying it live won't fix that.

Once live, spot-check it periodically against companies you know are currently in the news for distress. If the system isn't flagging them, either something is wrong or the signals aren't as good as you thought — both worth knowing.

### Scope discipline for Phase 7

Two rules that matter more than anything else in this phase:

**1. Phases 1-6 is a complete project. Phase 7 is the stretch goal.**

If you only get through Phases 1-6 in the time you have, that's already a successful research project — you've built a universe, labeled fragility events, computed baselines, tested alt-data signals, and produced real findings. That's enough to write up and defend. Phase 7 is what transforms a research project into a portfolio piece, but the research has to actually work first. A beautiful dashboard built on broken signals is worse than ugly notebooks with real insights — it looks impressive until someone asks a hard question, and then it falls apart.

Rule: **don't start Phase 7 until Phases 1-6 are solid.** "Solid" means the signals have been tested, the baselines work, the results are documented in the Decisions log, and you've spot-checked the outputs against cases you know. If any of that is still wobbly, go fix it before wiring anything into a dashboard. The dashboard will amplify whatever is underneath it — good or bad.

**2. One killer view done well beats ten half-built views.**

Once you're in Phase 7, you'll be tempted to add features. Sector breakdowns. User accounts. Historical charts with ten filters. Fancy animations. Email digests with custom preferences. A mobile version. Resist all of it. The scope for Phase 7 is:

- A fragility leaderboard (sortable table of all companies, color-coded composite score)
- A per-company detail page (signal history chart with lead-time markers)
- Email alerts for critical events
- A dead man's switch

That's it. Nothing else until those four things work end to end, are backtested, and have been used for at least a week on live data. Once that's done and stable, *then* you can add one more thing. Then use it for a week. Then maybe add another.

The failure mode to avoid: building five things to 60% and zero things to 100%. A dashboard with one view that works perfectly is far more impressive — and far more useful — than a dashboard with five views that each have a bug. This is the single most common mistake in personal projects, and it's the reason most of them never get shown to anyone.

Rule: **ship a minimum version. Use it. Then decide what to add based on what you actually missed, not what sounded cool at the start.**

**Done when:**
- The system runs automatically on a daily schedule (GitHub Actions)
- It computes all signals on current data
- It fires tiered alerts (critical / warning / info) with de-duplication
- It has a public Streamlit dashboard with a fragility leaderboard and per-company detail views
- It has a working dead man's switch (weekly status email)
- It has been backtested against historical fragility events with documented lead times per event
- A non-technical person can open the dashboard URL and understand what they're looking at

## Research hygiene

These are the habits that separate good research from research you can't trust. Build them from day one.

**1. Commit to GitHub early and often.**
Every time something works, commit it. End of each session, commit. Good rule: if you'd be upset to lose the last hour of work, commit. GitHub is your backup, your history, and your safety net. Write commit messages that explain *why* you changed something, not just *what* — "fix universe function" is useless; "fix off-by-one in universe function: was including companies on their removal date" is useful. Six months from now you'll thank yourself.

**2. Validate everything against something you already know.**
Never trust code that processes data without sanity-checking the output against facts you already know. When you build the universe function, ask it questions whose answers you know. When you pull prices, spot-check a few famous companies on famous dates (Apple on the day of a stock split, Tesla on its S&P 500 inclusion day). When you label fragility events, check whether SVB, Bed Bath & Beyond, and Peloton show up — if they don't, something's wrong. This is the single most important habit in data work.

**3. When something surprises you, stop and investigate.**
If a number looks weird, a chart looks off, or code does something you didn't expect — do not "fix" it by working around it. Figure out *why* it happened. Surprises are almost always bugs, and bugs you ignore early become disasters later. Write down what you found in the Decisions log. Some of the best insights in research come from running down a surprise.

**4. Raw data is sacred.**
Anything in `data/raw/` is never edited, ever. Not even to fix a typo. If you need to clean or transform it, the output goes in `data/processed/` and the transformation lives in a notebook or script. That way if something goes wrong three weeks from now, you can always go back to the source of truth and rerun. Also: don't commit raw data to git. Add `data/raw/` to `.gitignore`. Large files bloat the repo, and some data has licensing restrictions.

**5. Restart your notebook and run top-to-bottom regularly.**
Jupyter notebooks have a sneaky failure mode: you can run cells out of order, and the notebook appears to work even though the code is broken. Every so often — definitely before committing anything important — click "Restart & Run All" and make sure the whole notebook runs cleanly from top to bottom. If it doesn't, the notebook is broken even if it looked fine a minute ago.

**6. Keep a decision log.**
Every real decision — changing the fragility threshold, dropping a data source, picking a model, discovering something unexpected — goes in the Decisions section below, with the date and the reasoning. This is your lab notebook. It's also the easiest way to write the project up later: a good decisions log is basically a rough draft of the final report.

**7. Double-check before you believe a result.**
When you get a result that seems great, be suspicious. Good results are usually bugs. Re-run with different parameters, spot-check the inputs, ask "what would make this result fake?" and then check those things. The researchers who get into trouble are the ones who see a good number and run straight to publish. The ones who do good work are the ones who spend an hour trying to break their own result before they believe it.

**8. Don't delete old work, rename it.**
If you're rewriting a notebook, don't overwrite it — rename the old one to `01_build_universe_v1.ipynb` and make a new `01_build_universe.ipynb`. Commit both. Old approaches are valuable: they show what you tried, and sometimes you need to go back. Git keeps history too, but visible filenames are faster to scan.

## Decisions

*(Log decisions here as you make them. Format: date, decision, reasoning. When you revise something, add a new entry — don't edit the old one.)*

- **2026-04-10:** Working fragility definition set at 40% drawdown / 180 days / 90-day non-recovery. Will revise after seeing the drawdown distribution in Phase 4.
- **2026-04-10:** Universe is S&P 500, point-in-time, 2017-present. Built from two local files (current membership + historical changes) rather than scraping Wikipedia.
- **2026-04-10:** Stack is Python + uv + Jupyter + pandas + yfinance. Notebooks for exploration, `src/fragility/` for reusable code.
- **2026-04-10:** Executive departure tracking promoted from "after SEC filings" list to a primary SEC signal (now signal #4 in Phase 6). Reason: it runs on the same 8-K pipeline as the auditor-change signal, it's one of the strongest signals in the academic literature, and pairing it with auditor changes creates a "forced bad-news disclosure" composite. Decided against using LinkedIn directly as a leading indicator — scraping has legal and technical problems, and the underlying signal shows up in 8-K filings anyway, just with a 1-4 week delay. The delay is an acceptable trade-off for a free, legal, structured data source.
- **2026-04-10:** Phase 5 expanded from one baseline (Altman Z-score) to two (Z-score + Beneish M-Score). Reason: Z-score predicts bankruptcy, M-Score detects earnings manipulation that often precedes distress. Two baselines give sharper tests for alt-data signals in Phase 6 — a new signal can beat one but not the other, and that difference is diagnostic. M-Score is a one-day implementation so the marginal cost is small.
- **2026-04-10:** Added restatements (8-K Item 4.02) as the fifth SEC signal in Phase 6. Same EDGAR pipeline as auditor changes and executive departures — different 8-K item code. Strongest forced-disclosure red flag in accounting per the short-seller and academic literature. The three 8-K signals together form a "forced bad-news disclosure" composite.
- **2026-04-10:** Added short-seller research report reading as required prep for Phase 6. Reason: short-sellers run the same research question with real money on the line and publish their reasoning publicly. Reading 3-4 reports builds qualitative intuition for what fragility looks like in real cases before trying to detect it algorithmically.
- **2026-04-10:** Reframed the project goal from "replace traditional fragility models with alt-data" to "augment traditional models with faster, earlier, and behavioral signals." Reason: Z-score and M-Score aren't wrong, they're slow. They update quarterly with 30-135 day lags. The real opportunity for alt-data isn't better accounting math — it's closing the time gap and catching behavioral leading indicators that accounting numbers only reflect with a lag. Phase 6 success metric changed from "beats baseline precision" to "lead time, update frequency, and marginal information over baselines." The real deliverable is a composite that combines baselines with alt-data overlays, not a standalone alt-data model.
- **2026-04-10:** Added Phase 7 — real-time alert system as the capstone deliverable. Reason: the project framing (lead time, real-time overlays over slow quarterly baselines) only matters if there's actually a real-time system watching for signals. A working dashboard with public URL is also a much better deliverable than a notebook — shareable, demonstrable, and useful. Stack decided: GitHub Actions (scheduler, free) + SQLite (storage, no server) + Streamlit (dashboard, Python-first) + Gmail SMTP (email alerts, free). Total cost: zero. Backtest validation before live deployment is required.
- **2026-04-10:** Added scope discipline rules to Phase 7. Phases 1-6 is a complete project; Phase 7 is explicit stretch. Phase 7 scope hard-capped at four things: leaderboard, detail page, email alerts, dead man's switch. No feature additions until the minimum version is shipped, backtested, and used on live data for at least a week. Reason: scope creep is the most common failure mode in personal projects, and a broken Phase 7 is worse than no Phase 7.
- **2026-04-10:** Added explicit "Who this tool is for" framing. The tool is a short-activist tool in spirit, even though its direct users are mostly not short-sellers — lenders, suppliers, audit committees, credit analysts, long-only PMs looking to avoid fragile names. All share the same underlying question: is this company deteriorating in ways the quarterly financials haven't yet revealed? Reason for naming this explicitly: it sharpens every downstream design decision, especially the asymmetric cost of false negatives vs. false positives, and it tells the author what audience to write the final output for (a decision-maker with money on the line, not an academic reviewer).

## Getting started

Once the repo is created and you're in the project directory, open Claude Code and paste this prompt:

> "I'm on macOS and new to Python development. Help me set up a new project called `financial-fragility` using `uv` for package management. I want Jupyter notebooks for exploration and a `src/fragility/` folder for reusable Python modules. Walk me through every command, explain what each one does, and help me verify it's working by running a hello-world notebook that imports pandas and yfinance and plots Apple's stock price for the last year."

Once that works, commit everything to GitHub with a commit message like "initial project setup: uv, jupyter, hello world notebook working." Then upload the two S&P files into `data/raw/`, start on `01_build_universe.ipynb`, and go.

Good luck. Take your time, double-check your work, and log your decisions.
