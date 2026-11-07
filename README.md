# Messy PDF Pipeline

Turns messy invoice PDFs, including scans, skewed pages, and difficult tables, into checked JSON. Anything uncertain goes to a person instead of being silently accepted.

It runs entirely on one PC and costs nothing to run. A language model (OpenAI by default) is optional and off by default. Turned on, it sends failed fields to the provider you choose.

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
13. **Extract the fields** with rules. Each value records its page and position, and a field stays empty if it isn't found. Labels are read in English, German, French, Spanish and Italian. The fields include the subtotal, tax, CGST, SGST, IGST, shipping, discount, and the grand total. A credit note is recognised from its title.
14. **Compare key values with the page.** For a digital page, the invoice number, dates, subtotal, tax and total are read from the text layer. The page is rendered, and only the small area around each of those values is read with OCR. A value that isn't visible on the page, or that the page shows differently, fails a check. This catches text hidden in white, or a text layer that says 400 while the page shows 500.
15. **Check the values** with ordinary code:
    - Required fields, formats, and OCR confidence.
    - Quantity times price equals each line amount.
    - The lines add up to the subtotal. With no subtotal, the lines plus tax and charges must make the total.
    - The subtotal plus tax (or CGST, SGST and IGST) plus shipping minus any discount equals the total.
    - Look-alike characters in IDs, and words that OCR has run together in names and item descriptions.
    - Table structure, and conflicting values.
    - An invoice with no item table, such as a one-line bill, needs no person only if no table that could hold items exists, the missed-text check didn't have to recover anything, and the subtotal and charges add up to the total. A small totals box doesn't count as an item table.
16. **Score each field and decide.** A field is accepted only if it passes all its checks and its score is at least 0.85. A failed check caps the score at 0.5.
17. **Read failed scan fields again.** For a field that failed a check on a scan, the line is read again zoomed in, sharpened, and in black and white. A new reading replaces the old one only if every check then passes and no other field gets worse. A money value needs the arithmetic to confirm it. An invoice number or a date needs two different ways of reading to agree. Nothing is guessed.
18. **Handle what is still failing.** It goes to a person. If the language model is turned on, the failed fields first go to a cheap model that can read images, then once to a second, different model, and the same checks run again after each answer. A value for a field with no arithmetic needs both models to read the same printed value. If a model disagrees with what was read, a person decides, with the model's answer shown as a suggestion.
19. **Check against earlier invoices.** The same user's earlier invoices from the same vendor are looked up. The invoice is flagged if the vendor and invoice number were already processed, if the number doesn't look like that vendor's usual ones (for example `INV-l005` after `INV-1001` to `INV-1004`), or if the currency differs from the vendor's usual one. Nothing from another user is ever used.
20. **Save the result** as JSON. Chart images are separate files, and the JSON stores their paths.

## Searching your invoices

The web page has a search box for the user's own accepted and reviewed invoices, by vendor and date range. The totals are added up per currency by the server, and credit notes are taken off. An invoice that still needs review is not counted.

If the language model is turned on, a question can be typed in plain words, such as "what did Acme invoice us last quarter?". The model only turns the question into three filters: a vendor and a date range. The search and the sums are done by plain code over that user's own invoices, and the filters it chose are shown with the answer. The model never sees the invoices and never writes the answer.

## Measured results

These come from generated documents only. **Accuracy on real invoices has not been measured.** The generator can't produce every way that real documents go wrong, so expect real-world results to be worse.

**Standard set**, on 162 freshly generated documents the code had never been run on (1,620 fields, language model off). It was run once.

- **Silent errors (wrong values accepted without review): 1.** One line-item description had lost a space (`Whiteboardmarkers`) in a 54-item scan, while every number was right. I then added a check for words of 16 or more letters run together. That check has not been measured on unseen documents, and it can't catch a short pair such as `Officechair`.
- Correct values: 99.44%
- Fields sent to review: 8.27%
- Table cell F1: 0.972
- Crashes: 0
- Documents that must be reviewed but were accepted: 0 of 36
- Documents accepted with no review at all: 80 of the 126 that had no planted problem
- The second pass fixed 10 fields on scans, and none of them was wrong.
- Time per page: 1.19 s median and 5.03 s at the 95th percentile. Digital pages take about 1.0 s. Scans take 3.4 to 4.3 s.
- Language model cost: $0 (it was off)

**Variants**, on 48 generated documents (extra shipping and discount lines, split CGST and SGST, credit notes, a one-line bill with no item table, German and French invoices, each as a digital PDF and as a scan). It was run once on new seeds.

- Silent errors: 0
- Correct values: 99.8%
- Documents accepted with no review at all: 42 of 48
- One name with a dropped space on a scanned credit note was caught by the new check and sent to review.
- Spanish and Italian labels are only unit-tested. No generated Spanish or Italian documents exist.

Two cautions about these numbers. The first standard set I used (seeds 100 to 102) is no longer a fair test, because I used its failures to fix the date reading, the vendor guess, and the second pass. And each time a check is added after looking at failures, the set it was tuned on stops being unseen. The numbers above are the ones taken before the last fix.

Four users uploading at the same moment (3 workers, 6 cores):

- 20 documents: all finished in 26 s, and the slowest went from upload to result in 23 s.
- 48 documents, while one worker was killed: no document was lost, and all finished in 68 s. The slowest took 63 s, most of that waiting for the dead worker's lease to expire (20 s, then up to 15 s until another worker notices).
- An earlier run killed one worker and another worker's extraction process: no document was lost, and all finished in 83 s. That was before the lease was cut from 60 s to 20 s.
- 48 documents, while the database was restarted: all three workers reconnected, 3 jobs were recovered, and every document got its own correct result. All finished in 91 s. The web app returned a few 500 errors while the database was down.
- No user could see another user's work.
- Peak memory was about 860 MB per worker.

To repeat these, run the commands below with the system running. Replace `PASSWORD` with the one in your `.env`. Use a new `--first-seed` for a fair test.

```bash
python -m pdfpipe.evaluation.benchmark --out eval_out/standard --seeds 3 --first-seed 400
```

```bash
python -m pdfpipe.evaluation.benchmark --out eval_out/variants --seeds 4 --first-seed 300 --variants
```

```bash
python scripts/load_test.py --db-url postgresql://pdfpipe:PASSWORD@127.0.0.1:5433/pdfpipe
```

## Known limits

- **Not tested on real documents.** This is the biggest gap. Run real invoices through it before trusting the numbers above.
- **The language model step has never made a real call.** It has only been tested with fake clients, so its real accuracy and cost are unknown. The OpenAI model name and prices come from OpenAI's documentation and pricing page, not from a real run.
- **With the model on, invoices leave your PC.** The text and images of failed fields are sent to OpenAI (or Anthropic, if chosen). Without it, nothing does.
- **Invoices only.** The field rules, labels, and checks are written for invoices and credit notes. Other document types get their text and tables, but no meaningful fields.
- **Labels in other languages.** English, German, French, Spanish and Italian are read. A label in any other language is not found, and the field goes to review. Spanish and Italian have unit tests only.
- **Handwriting and photographed pages have not been tested.**
- **Other charges.** Shipping, discount, tax, CGST/SGST/IGST, a rounding line, and a deposit or amount already paid are understood. Any other charge, such as a service fee under a name the labels don't list, or a per-line discount, makes the total fail its check, so a person handles it.
- **Names and descriptions read by OCR have no arithmetic to protect them.** Words that OCR ran together are caught for names, and for item descriptions only when a word is 16 or more letters long. A shorter pair such as `Officechair` passes.
- **The hidden-text check covers six fields** (the invoice number, both dates, subtotal, tax and total), and digital pages only. A different value hidden in the vendor name or an item description isn't compared. It also adds about 0.4 s to each digital page. Set `PDFPIPE_VISIBLE_CHECK_ENABLED=false` to turn it off.
- **The second pass is a bounded retry, not an AI agent.** It uses only the local OCR, on scanned pages, and keeps a new reading only when every check agrees. The language model step is the other reader.
- **Memory of earlier invoices is only as good as what it learned.** It needs three earlier invoices from a vendor before it judges a number's shape. It learns from invoices that were accepted, and a wrong value that slipped through would teach a wrong pattern. It can't catch a wrong digit inside the right pattern, such as `INV-1048` for `INV-1042`.
- **Asking in plain words needs the language model on,** and those small calls are not counted against the daily budget. It has only been tested with a fake client.
- **Without the model, an invoice in a language the labels don't cover** has nothing read from it and is sent to review in full. A vendor guessed from the page top is never accepted on such a page.
- **An amount like `1.234` is left for a person.** It is 1234 in some countries and 1.234 in others, so it isn't guessed. European invoices with whole amounts written like `1.250` therefore go to review.
- **Thresholds are hand-set.** The 0.85 minimum and the score formula haven't been calibrated against real error rates.
- **Half of the generated documents still need a person for at least one field.** Some of that is deliberate: 36 of the 162 had planted problems.
- **Line items can't be edited on the web page.** A person can confirm them or correct the document's other fields.
- **Confirming is one click.** Typed values are checked, but a reviewer can still confirm every flagged field without looking at the page.
- **Uploading the same file again reuses the old result** even if settings changed in between, such as turning the language model on.
- **Retries start the whole document again.** Results aren't saved page by page.
- **Monitoring is basic.** There are logs and a `/health` endpoint, but no dashboard.
- **There is no backup and no retention rule.** Uploaded files and results stay until a user deletes them.
- **Traffic is plain HTTP**, so it's only suitable for one PC or a trusted network.
- **PDFs that hide scripts or attachments are not cleaned.** Those parts are never run, but they stay in the stored file.

## Turning on the language model

It is off by default. With it off, nothing leaves your PC. With it on, the text and page images of **failed fields** are sent to the provider you choose. It is for what the rules and OCR can't do well: an invoice in a language the labels don't cover, handwriting, and photographed pages.

OpenAI is the default provider. Put your key in `.env` yourself (that file is never committed), then run `docker compose up -d`:

```
PDFPIPE_LLM_ENABLED=true
PDFPIPE_OPENAI_API_KEY=your-key
```

The model and the rules around it:

- **A cheap model reads the failed fields.** The default is `gpt-4o-mini`, which accepts images and returns JSON in a fixed shape. Change it with `PDFPIPE_OPENAI_FIRST_MODEL`. Page images are sent at high detail so small digits can be read. `PDFPIPE_OPENAI_IMAGE_DETAIL=low` is cheaper and reads worse.
- **A second, different model gives a second opinion.** Set `PDFPIPE_OPENAI_SECOND_MODEL`, for example to a different small model. A value for a field with no arithmetic to check it (the invoice number, a date, a name, the currency) is filled in only when two different models read the same printed value. With no second model set, or if they disagree, the value is only a suggestion for a person. Turn this off with `PDFPIPE_LLM_REQUIRE_AGREEMENT=false`.
- **An invoice that the rules can't read at all** (three or more of the invoice number, date, vendor and total are empty) has every empty field requested, not only the failed ones.
- **A money value needs no second model**, because the arithmetic checks it. A wrong total from the model is stopped by the sum check.
- **A value is only used if it is printed on the page** as a whole token, and the same checks re-run on every answer.
- **Nothing the model says can change a decision.** The page is sent as untrusted data, the answer must fit a fixed JSON shape, and the model can't call tools or take actions.

Spending and failure limits, all set the same way in `.env` and all with defaults. They count every provider's calls together:

- `PDFPIPE_LLM_MAX_COST_PER_DOCUMENT_USD=0.25`: one document stops calling the model once it has cost this much.
- `PDFPIPE_LLM_DAILY_BUDGET_USD=5.0`: all workers together stop for the day at this total.
- `PDFPIPE_LLM_PAUSE_AFTER_FAILURES=5` and `PDFPIPE_LLM_PAUSE_SECONDS=300`: after this many failed calls in a row, a worker stops calling the model for that long.

After any of these limits, the remaining fields go to a person. The limits are checked before a call, so one document can pass its limit by the cost of its last call. Each worker counts failures on its own. If the model is turned on but the key is empty, the model stays off and the log says so.

The cost is worked out from the tokens each reply reports, using the prices in [llm.py](src/pdfpipe/fields/llm.py). Those OpenAI prices were read from OpenAI's pricing page in November 2026, so check them. A model that isn't in the list is priced high, so the caps stay on the safe side.

To measure it on generated documents before trusting it, run this. It sends data to OpenAI and costs a small amount of money.

```bash
python -m pdfpipe.evaluation.benchmark --out eval_out/model --seeds 3 --first-seed 600 --variants dutch photo --llm
```

Claude is still available. Set `PDFPIPE_LLM_PROVIDER=anthropic` and `PDFPIPE_ANTHROPIC_API_KEY`. Only the provider you choose is ever called. Claude's Sonnet requests use Anthropic's server-side refusal fallback.

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
