"""Four people using the running system at once, over its real HTTP API.

Creates temporary users, uploads a mix of documents from each at the same moment, waits for them all,
then checks accuracy and that no user can see another user's work. Removes its users when finished.

Run the system first with `docker compose up -d`, then:
    python scripts/load_test.py --db-url postgresql://pdfpipe:PASSWORD@127.0.0.1:5433/pdfpipe
"""

import argparse
import secrets
import statistics
import tempfile
import threading
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from pdfpipe.config import Settings
from pdfpipe.evaluation.synth import make_case
from pdfpipe.service import auth, db

DOCUMENT_MIX = [
    ("ruled", "digital"),
    ("borderless", "scan_light"),
    ("two_column", "digital"),
    ("merged_header", "scan_heavy"),
    ("with_chart", "digital"),
    ("ruled", "bad_text_layer"),
    ("long", "digital"),
    ("borderless", "sideways"),
    ("ruled", "wrong_total"),
    ("two_column", "mixed"),
]


class Person(threading.Thread):
    def __init__(self, name: str, password: str, base_url: str, cases: list[Any]) -> None:
        super().__init__(name=name)
        self.password, self.base_url, self.cases = password, base_url, cases
        self.jobs: dict[str, dict[str, Any]] = {}
        self.upload_seconds: list[float] = []
        self.errors: list[str] = []
        self.finished_jobs: dict[str, dict[str, Any]] = {}
        self.client: httpx.Client | None = None

    def run(self) -> None:
        self.client = httpx.Client(base_url=self.base_url, timeout=60)
        reply = self.client.post("/api/login", json={"username": self.name, "password": self.password})
        if reply.status_code != 200:
            self.errors.append(f"login failed: {reply.status_code}")
            return
        for case in self.cases:
            started = time.perf_counter()
            try:
                with case.pdf.open("rb") as handle:
                    reply = self.client.post("/api/jobs", files={"file": (case.pdf.name, handle, "application/pdf")})
            except httpx.HTTPError as error:
                self.errors.append(f"upload failed: {error!r}")
                continue
            self.upload_seconds.append(time.perf_counter() - started)
            if reply.status_code != 202:
                self.errors.append(f"upload failed: {reply.status_code} {reply.text[:120]}")
                continue
            self.jobs[reply.json()["id"]] = {"case": case, "uploaded": time.time()}


def wait_for_all(people: list[Person], limit_seconds: float, during: Any = None) -> float:
    started = time.perf_counter()
    while time.perf_counter() - started < limit_seconds:
        waiting = 0
        for person in people:
            assert person.client is not None
            try:
                jobs = person.client.get("/api/jobs").json()
            except (httpx.HTTPError, ValueError):
                # The server can be briefly unavailable during a failure test, so check again next time.
                waiting += 1
                continue
            for job in jobs:
                if job["status"] in ("queued", "running"):
                    waiting += 1
        if during:
            during(waiting)
        if waiting == 0:
            return time.perf_counter() - started
        time.sleep(2)
    raise TimeoutError(f"documents were still waiting after {limit_seconds} seconds")


def check_results(people: list[Person]) -> dict[str, Any]:
    durations, statuses, wrong_silent, must_review_missed, total_fields, correct_fields = [], {}, 0, 0, 0, 0
    for person in people:
        assert person.client is not None
        for job_id, info in person.jobs.items():
            job = person.client.get(f"/api/jobs/{job_id}").json()
            statuses[job["status"]] = statuses.get(job["status"], 0) + 1
            if job["status"] == "failed":
                person.errors.append(f"{info['case'].name} failed: {job['error']}")
                continue
            from datetime import datetime

            durations.append(
                (datetime.fromisoformat(job["finished_at"]) - datetime.fromisoformat(job["created_at"])).total_seconds()
            )
            truth = info["case"].truth
            fields = job["result"]["machine"]["fields"]
            if truth["must_review"] and job["status"] == "accepted":
                must_review_missed += 1
            for name, want in truth["fields"].items():
                field = fields[name]
                got = field["value"]
                if isinstance(want, float):
                    same = got is not None and Decimal(str(got)) == Decimal(str(want))
                else:
                    same = got == want
                total_fields += 1
                correct_fields += same
                if not same and not field["needs_review"] and got is not None:
                    wrong_silent += 1
    durations.sort()
    return {
        "statuses": statuses,
        "field_accuracy": round(correct_fields / total_fields, 4) if total_fields else None,
        "silent_errors": wrong_silent,
        "must_review_missed": must_review_missed,
        "seconds_upload_to_result_p50": round(statistics.median(durations), 1) if durations else None,
        "seconds_upload_to_result_p95": round(durations[min(len(durations) - 1, int(0.95 * len(durations)))], 1)
        if durations
        else None,
        "seconds_upload_to_result_max": round(durations[-1], 1) if durations else None,
    }


def check_isolation(people: list[Person]) -> list[str]:
    problems = []
    for person in people:
        assert person.client is not None
        own = {job["id"] for job in person.client.get("/api/jobs").json()}
        if own != set(person.jobs):
            problems.append(f"{person.name}'s list doesn't match their uploads")
        for other in people:
            if other is person:
                continue
            for job_id in list(other.jobs)[:3]:
                for path in (f"/api/jobs/{job_id}", f"/api/jobs/{job_id}/result.json"):
                    status = person.client.get(path).status_code
                    if status != 404:
                        problems.append(f"{person.name} got status {status} on {other.name}'s {path}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--users", type=int, default=4)
    parser.add_argument("--documents-each", type=int, default=5)
    parser.add_argument("--wait-seconds", type=float, default=900)
    args = parser.parse_args()

    settings = Settings(_env_file=None, database_url=args.db_url)  # type: ignore[call-arg]
    names = [f"loadtest_{secrets.token_hex(3)}_{index}" for index in range(args.users)]
    password = secrets.token_urlsafe(18)
    with db.connect(settings) as connection:
        for name in names:
            auth.create_user(connection, name, password)

    people: list[Person] = []
    with tempfile.TemporaryDirectory() as temporary:
        try:
            # Documents are made first, one at a time, so everyone's uploads start at the same moment.
            for user_index, name in enumerate(names):
                cases = []
                for index in range(args.documents_each):
                    style, condition = DOCUMENT_MIX[(user_index + index) % len(DOCUMENT_MIX)]
                    cases.append(make_case(style, condition, (user_index + 1) * 1000 + index, Path(temporary)))
                people.append(Person(name, password, args.base_url, cases))
            started = time.perf_counter()
            for person in people:
                person.start()
            for person in people:
                person.join()
            uploads = sorted(seconds for person in people for seconds in person.upload_seconds)
            print(f"uploaded {len(uploads)} documents, slowest upload {uploads[-1]:.2f}s")

            peak = {"waiting": 0}
            wait_for_all(people, args.wait_seconds, lambda waiting: peak.update(waiting=max(peak["waiting"], waiting)))
            total = time.perf_counter() - started
            report = check_results(people)
            problems = check_isolation(people)
            print(f"all finished in {total:.0f}s. Most waiting at once: {peak['waiting']}")
            for key, value in report.items():
                print(f"  {key}: {value}")
            print(f"  isolation problems: {problems or 'none'}")
            errors = [error for person in people for error in person.errors]
            print(f"  errors: {errors or 'none'}")
        finally:
            # Deleting through the API also removes each job's files from disk.
            for person in people:
                if person.client is not None:
                    for job_id in person.jobs:
                        person.client.delete(f"/api/jobs/{job_id}")
            with db.connect(settings) as connection:
                for name in names:
                    connection.execute("DELETE FROM users WHERE username = %s", (name,))
            print("removed the temporary users, their jobs and their files")


if __name__ == "__main__":
    main()
