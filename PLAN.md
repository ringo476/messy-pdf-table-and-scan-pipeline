# Messy PDF, Table & Scan Pipeline: Plan

Turn messy invoice PDFs (scans, skewed pages, complex tables) into **validated JSON** with confidence scores. Anything uncertain goes to a person.

This plan now matches what was built. [README.md](README.md) has the step-by-step flow, how to run it, and the measured results.

## Design

```
Browser -> FastAPI web app (per-user login) -> files on local disk
                  |
        Postgres (data and job queue)
                  |
Workers: classify -> image -> layout -> words -> tables -> rules -> checks -> score
                  |
        Accepted, or human review (optional Claude first)
```

- Runs on one PC with Docker. There is no server, domain, or Redis.
- Every page goes through the layout model, and words come from the text layer or OCR.
- Field extraction and checks are ordinary code. Claude is off by default and only sees failed fields.
- Postgres `FOR UPDATE SKIP LOCKED` hands out jobs, and leases recover jobs from dead workers.

## Cost

$0 with Claude off, which is the default. With Claude on, the cost depends on how many fields fail, and each document records its estimate.

## Status

Built and tested: extraction, rules, checks, scoring, review, login, workers, evaluation, and the 4-user load test.

Not done yet:

1. Measure accuracy on real invoices. This matters most, because all current numbers come from generated documents.
2. Run the Claude step against the real API, and compare local only, Claude only, and the combination.
3. Calibrate the score threshold on real error rates.
4. Add a CI pipeline.
