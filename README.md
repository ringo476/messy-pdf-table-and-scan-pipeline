# Messy PDF Pipeline

Turns messy invoice PDFs, including scans, skewed pages, and difficult tables, into checked JSON. Anything uncertain goes to a person instead of being silently accepted.

It runs entirely on one PC and costs nothing to run. The Claude backup is optional and off by default.

## Run it

You need Docker Desktop running.

1. Create your settings file, then open `.env` and set your own `POSTGRES_PASSWORD`.

```bash
cp .env.example .env
```

2. Build and start everything. The first build downloads the models and takes several minutes.

```bash
docker compose up -d --build
```

3. Create a login for each person. It asks for a password, which must have at least 10 characters.

```bash
docker compose exec web pdfpipe create-user your-name
```

4. Open http://127.0.0.1:8000 and log in.

To stop everything, run this. Your data is kept.

```bash
docker compose stop
```

To let other machines on your network connect, set `PDFPIPE_BIND=0.0.0.0` in `.env`. That traffic is plain HTTP, so only do this on a network you trust.

## How a PDF is processed

Every page goes through the same steps. The only difference between page types is where the words come from in step 6.

1. **Accept the PDF.** The file is checked in its own short-lived process, so a file that crashes or hangs the PDF library is rejected without affecting anyone else.
2. **Classify each page** as digital, scanned, scanned with a bad text layer, or mixed.
3. **Convert every page to an image**, because the layout model only works on images.
4. **Clean up scanned images.** Turn the page upright, straighten it, reduce noise, and fix faded contrast. Digital pages skip this.
5. **Find the regions** with the layout model: headings, paragraphs, tables, charts, and pictures. This step reads no words.
6. **Get the words and their positions.**
   - Digital page: from the PDF's text layer, with no OCR.
   - Scanned page, or one with a bad text layer: OCR reads inside each region.
   - Mixed page: the text layer, plus OCR on the embedded images.
7. **Check for missed text.** If a region has no words, or the image shows text where no word was found, full-page OCR runs. Missing words are added and flagged.
8. **Put each word in its region** by position. Words outside every region are kept and flagged.
9. **Build the tables.** The table model finds rows, columns, and merged cells, and words go into cells by position. If its rows can't explain the visible lines of text, the rows are rebuilt from those lines.
10. **Put the regions in reading order**, including across columns.
11. **Save the charts** as images, marked as not extracted. Their values are never guessed.
12. **Join split tables** when a table continues on the next page and its columns line up.
13. **Extract the fields** with rules. Each value records its page and position, and a field stays empty if it isn't found.
14. **Check the values** with ordinary code: required fields, formats, line and total arithmetic, OCR confidence, look-alike characters in IDs, table structure, and conflicting values.
15. **Score each field and decide.** A field is accepted only if it passes all its checks and its score is at least 0.85. A failed check caps the score at 0.5.
16. **Handle the failed fields.** They go to a person. If Claude is turned on, they first go to Haiku, then once to Sonnet, and the same checks run again after each answer. If Claude disagrees with what was read, a person decides, with Claude's answer shown as a suggestion.
17. **Save the result** as JSON. Chart images are separate files, and the JSON stores their paths.

## Measured results

These come from generated documents only. **Accuracy on real invoices has not been measured.** The generator can't produce every way that real documents go wrong, so expect real-world results to be worse.

Accuracy, on 162 generated documents the pipeline wasn't tuned on (1,620 fields, Claude off):

- **Silent errors (wrong values accepted without review): 0**
- Correct values: 99.38%
- Fields sent to review: 8.46%
- Table cell F1: 0.973
- Crashes: 0
- Documents that must be reviewed but were accepted: 0 of 36
- Documents accepted with no review at all: 80 of the 126 that had no planted problem
- Time per page: 0.82 s median, 3.74 s at the 95th percentile (about 0.6 s for digital pages, 2–3.5 s for scans)
- Claude cost: $0

Four users uploading at the same moment (3 workers, 6 cores):

- 20 documents: all finished in 26 s, and the slowest went from upload to result in 23 s.
- 48 documents, while one worker was killed: no document was lost, and all finished in 68 s. The slowest took 63 s, most of that waiting for the dead worker's lease to expire (20 s, then up to 15 s until another worker notices).
- An earlier run killed one worker and another worker's extraction process: no document was lost, and all finished in 83 s. That was before the lease was cut from 60 s to 20 s.
- 48 documents, while the database was restarted: all three workers reconnected, 3 jobs were recovered, and every document got its own correct result. All finished in 91 s. The web app returned a few 500 errors while the database was down.
- No user could see another user's work.
- Peak memory was about 860 MB per worker.

To repeat these, run the commands below with the system running. Replace `PASSWORD` with the one in your `.env`.

```bash
python -m pdfpipe.evaluation.benchmark --out eval_out/unseen --seeds 3 --first-seed 100
```

```bash
python scripts/load_test.py --db-url postgresql://pdfpipe:PASSWORD@127.0.0.1:5433/pdfpipe
```

## Known limits

- **Not tested on real documents.** This is the biggest gap. Run real invoices through it before trusting the numbers above.
- **The Claude step has never called the real API.** It has only been tested with a fake client, so its real accuracy and cost are unknown.
- **Invoices only.** The field rules, labels, and checks are written for invoices.
- **Some invoice layouts go to review instead of being understood.** Tax split into CGST and SGST, shipping, discounts, and credit notes make the totals fail the sum check, so a person handles them.
- **A digital PDF can show one number and hide another in its text layer.** The pipeline reads the text layer and doesn't compare it with what the page shows.
- **An amount like `1.234` is left for a person.** It is 1234 in some countries and 1.234 in others, so it isn't guessed. European invoices with whole amounts written like `1.250` therefore go to review.
- **Thresholds are hand-set.** The 0.85 minimum and the score formula haven't been calibrated against real error rates.
- **Half of the generated documents still need a person for at least one field.** Some of that is deliberate: 36 of the 162 had planted problems.
- **Line items can't be edited on the web page.** A person can confirm them or correct the document's other fields.
- **Confirming is one click.** Typed values are checked, but a reviewer can still confirm every flagged field without looking at the page.
- **Uploading the same file again reuses the old result** even if settings changed in between, such as turning Claude on.
- **Retries start the whole document again.** Results aren't saved page by page.
- **Monitoring is basic.** There are logs and a `/health` endpoint, but no dashboard.
- **Traffic is plain HTTP**, so it's only suitable for one PC or a trusted network.
- **PDFs that hide scripts or attachments are not cleaned.** Those parts are never run, but they stay in the stored file.

## Turning on Claude

Set these in `.env`, then run `docker compose up -d`. Only failed fields are sent, and each document's estimated cost is recorded in its result.

```
PDFPIPE_LLM_ENABLED=true
PDFPIPE_ANTHROPIC_API_KEY=your-key
```

Spending and failure limits, all set the same way in `.env` and all with defaults:

- `PDFPIPE_LLM_MAX_COST_PER_DOCUMENT_USD=0.25`: one document stops calling Claude once it has cost this much.
- `PDFPIPE_LLM_DAILY_BUDGET_USD=5.0`: all workers together stop for the day at this total.
- `PDFPIPE_LLM_PAUSE_AFTER_FAILURES=5` and `PDFPIPE_LLM_PAUSE_SECONDS=300`: after this many failed calls in a row, a worker stops calling Claude for that long.

After any of these limits, the remaining fields go to a person. The limits are checked before a call, so one document can pass its limit by the cost of its last call. Each worker counts failures on its own.

Sonnet requests use Anthropic's server-side refusal fallback, so if a safety classifier wrongly declines an invoice, another model answers. Remove `fallbacks` in [llm.py](src/pdfpipe/fields/llm.py) if you don't want that.

## Develop

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -c requirements.lock -e ".[service,llm,dev]"
```

Most tests need nothing else:

```bash
.venv/Scripts/python -m pytest
```

The database tests need a database of their own, because **they erase the database they connect to**. Never point them at the one holding your documents.

```bash
docker compose exec db createdb -U pdfpipe pdfpipe_test
```

Then set `PDFPIPE_TEST_DATABASE_URL` to `postgresql://pdfpipe:PASSWORD@127.0.0.1:5433/pdfpipe_test` and run pytest again.
