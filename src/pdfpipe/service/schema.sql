-- Safe to run repeatedly. Every statement checks whether its object already exists.

CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    BIGINT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    expires_at TIMESTAMPTZ NOT NULL
);

-- A document belongs to one user. The same file uploaded by two users is two separate rows,
-- so reusing a result can never show one user's edits to another user.
CREATE TABLE IF NOT EXISTS jobs (
    id           UUID PRIMARY KEY,
    user_id      BIGINT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    filename     TEXT NOT NULL,
    fingerprint  TEXT NOT NULL,
    size_bytes   BIGINT NOT NULL,
    page_count   INTEGER NOT NULL,
    status       TEXT NOT NULL DEFAULT 'queued'
                 CHECK (status IN ('queued', 'running', 'accepted', 'needs_review', 'reviewed', 'failed')),
    attempts     INTEGER NOT NULL DEFAULT 0,
    run_after    TIMESTAMPTZ NOT NULL DEFAULT now(),
    lease_until  TIMESTAMPTZ,
    worker       TEXT,
    error        TEXT,
    result       JSONB,
    reused_from  UUID REFERENCES jobs (id) ON DELETE SET NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS jobs_ready ON jobs (run_after, created_at) WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS jobs_expired_lease ON jobs (lease_until) WHERE status = 'running';
CREATE INDEX IF NOT EXISTS jobs_by_user ON jobs (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS jobs_reuse ON jobs (fingerprint, finished_at DESC) WHERE status IN ('accepted', 'needs_review', 'reviewed');

-- Every change a person makes to a field. Rows are only ever added.
CREATE TABLE IF NOT EXISTS review_log (
    id        BIGSERIAL PRIMARY KEY,
    job_id    UUID NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    user_id   BIGINT NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    field     TEXT NOT NULL,
    old_value JSONB,
    new_value JSONB,
    at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
