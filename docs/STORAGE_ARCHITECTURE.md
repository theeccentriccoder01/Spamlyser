# Storage Architecture

This document explains how Spamlyser Pro persists data — feedback entries, custom
rules, and performance snapshots — and what safeguards are in place to prevent
data loss.

## Components

### `models/storage_manager.py` — `StorageManager`

The single point of truth for all JSON file I/O.  Key behaviours:

| Behaviour | Mechanism |
|---|---|
| **Atomic writes** | Data is written to a `.tmp` sibling file first, then `shutil.move`'d atomically over the target. |
| **Timestamped backups** | Before overwriting an existing file, the original is copied to `.<stem>.backups/<stem>.bak.<timestamp>.json`. |
| **Backup rotation** | `keep_backups` (default 5) most-recent backups are retained; older ones are pruned on every write. |
| **Age-based pruning** | Backups older than `max_backup_age_days` (default 30) are deleted regardless of count. |
| **Corrupt-file recovery** | `load_json` and `load_json_safe` automatically restore from the newest valid backup when the primary file cannot be parsed. |
| **Semantic validation** | `load_json_safe(validate=fn)` lets callers supply a predicate; if the primary file fails it, backups are tried in turn before returning the default. |

### `models/feedback_handler.py` — `FeedbackHandler`

Stores feedback as SQLite rows (WAL mode, 5-second busy timeout).

**Connection pooling** — the `db_connection_pool.ConnectionPool` manages a
bounded set of connections with automatic health checks and idle pruning.
Configuration is via `config.py`:

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `FEEDBACK_DB_MAX_CONNECTIONS` | `SPAMLYSER_FEEDBACK_DB_MAX_CONNS` | 5 | Max simultaneous connections |
| `FEEDBACK_DB_IDLE_TIMEOUT` | `SPAMLYSER_FEEDBACK_DB_IDLE_TIMEOUT` | 300s | Idle connections pruned after this |

**Retry policy** — write operations are wrapped in `_retry_on_db_error()`
which catches transient SQLite errors (`database is locked`, `disk I/O error`)
and retries up to 3 times with exponential backoff. This handles the race
condition where multiple Streamlit sessions write simultaneously.

**Connection health-check policy** — before every write the thread-local
connection is pinged with `SELECT 1`.  If the ping fails (stale descriptor,
deleted DB file, prior transaction error) the connection is silently closed and
a new one is opened for the same thread.  This prevents silent write failures
that would otherwise be indistinguishable from a successful `conn.commit()`.

```
Thread calls save_feedback()
  │
  ▼
_get_connection(db_path)
  │
  ├─ has existing conn?
  │     ├─ YES → ping with SELECT 1
  │     │           ├─ OK → return conn
  │     │           └─ FAIL → close conn, set conn = None
  │     └─ NO
  │
  └─ conn is None → _open_connection(db_path)
                       (WAL + busy_timeout=5000)
```

### `config.py` — centralised paths

All file paths are configured through `config.py` / environment variables so
that tests and deployments can redirect storage without touching application
code:

| Variable | Default |
|---|---|
| `SPAMLYSER_DATA_DIR` | `<project root>/data` |
| `SPAMLYSER_FEEDBACK_DB` | `<data_dir>/feedback.db` |
| `SPAMLYSER_FEEDBACK_JSON` | `<data_dir>/feedback_data.json` |
| `SPAMLYSER_CUSTOM_RULES` | `<data_dir>/custom_rules.json` |
| `SPAMLYSER_PERFORMANCE_DATA` | `<data_dir>/performance_data.json` |

## Data Flow

```
User action
   │
   ▼
FeedbackHandler.save_feedback()   ←── feedback entries (SQLite)
StorageManager.save_json()        ←── custom rules, perf snapshots (JSON)
   │
   ├── write to <file>.tmp
   ├── move atomically to <file>
   └── create timestamped backup in .<stem>.backups/
```

## Recovery Scenarios

| Scenario | Recovery path |
|---|---|
| Primary JSON file corrupted mid-write | `.tmp` write → rename: corruption only possible if OS crashes between write and rename. Backup available from previous write. |
| Primary JSON file manually deleted | `load_json` / `load_json_safe` returns `default`; no backup exists since the file was new. |
| Primary JSON file semantically invalid | `load_json_safe(validate=fn)` falls back through backups until one passes validation. |
| SQLite connection closed externally | Health-check on next write detects the stale handle; reconnection is automatic and transparent to the caller. |
| SQLite file deleted while app is running | Health-check ping raises `OperationalError`; reconnection re-creates the file at the same path via `sqlite3.connect`. |

## Testing

```
tests/test_storage_manager.py   — StorageManager unit tests (14 cases)
tests/test_storage_manager.py   — load_json_safe tests (5 cases)
```

Run with:

```bash
pytest tests/test_storage_manager.py -v
```
