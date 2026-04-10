# Financial Fragility Project

## What this is

A research project to see whether alternative data sources — GitHub activity, job postings, Glassdoor ratings, and similar — can predict when a public company is about to become financially fragile, before traditional financial metrics catch it.

Think of it as augmenting the Altman Z-score with signals that didn't exist when Altman wrote his paper in 1968.

## Research question

> Do alternative data signals predict large drops in market capitalization among S&P 500 companies, and do they add predictive value beyond what traditional financial ratios already tell us?

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

### Phase 5 — Baseline with Altman Z-score

**What:** Compute the Altman Z-score quarterly for every company in the universe. Backtest: does the Z-score predict the fragility events you labeled in Phase 4? Measure it properly with precision, recall, and lead time (how many days of warning did it give).

**Why:** This is the benchmark every alt-data signal has to beat. The Z-score has been the standard bankruptcy predictor since 1968. If you skip this step, you'll have no way to prove your alt-data signals are adding anything — maybe GitHub activity "predicts" fragility, but maybe the Z-score already predicted it earlier and better. Without the baseline, you can't tell. Every alt-data experiment in Phase 6 gets compared to this number.

**Done when:** you have a Z-score for every company-quarter, a backtest report (precision/recall/lead time), and a clear number to beat.

### Phase 6 — Alt-data experiments

**What:** Start with GitHub activity — commit volume, contributor count, repo activity for public-company repos. Then job postings. Then Glassdoor. For each signal, the test is: *does it add predictive power beyond the Z-score baseline?*

**Why this order:** GitHub has a free, well-documented API and clean data, so you can learn the experiment loop without fighting the data source. Job postings are messier (scraping, API limits). Glassdoor is the hardest (anti-scraping, limited data). Start easy, get the methodology right, then tackle the harder sources.

**The discipline in this phase:** if a signal doesn't beat the baseline, move on. Don't fall in love with a data source just because you spent time on it. "Sunk cost" is the enemy of good research. Log what didn't work in the Decisions section so you don't accidentally re-try it later.

**Done when:** you have at least one alt-data signal tested end-to-end with a clear verdict (beats baseline / doesn't beat baseline / inconclusive and here's why).

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

## Getting started

Once the repo is created and you're in the project directory, open Claude Code and paste this prompt:

> "I'm on macOS and new to Python development. Help me set up a new project called `financial-fragility` using `uv` for package management. I want Jupyter notebooks for exploration and a `src/fragility/` folder for reusable Python modules. Walk me through every command, explain what each one does, and help me verify it's working by running a hello-world notebook that imports pandas and yfinance and plots Apple's stock price for the last year."

Once that works, commit everything to GitHub with a commit message like "initial project setup: uv, jupyter, hello world notebook working." Then upload the two S&P files into `data/raw/`, start on `01_build_universe.ipynb`, and go.

Good luck. Take your time, double-check your work, and log your decisions.
