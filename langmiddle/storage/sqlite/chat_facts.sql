-- ==========================================================
-- COMPLETE FACTS SCHEMA WITH PHASE 3 RELEVANCE SCORING
-- SQLite version with sqlite-vec extension
-- For fresh database setup - includes all tables and indexes
-- ==========================================================

BEGIN TRANSACTION;

-- ==========================================================
-- MAIN TABLE: facts
-- Stores user facts with metadata and relevance tracking
-- ==========================================================
CREATE TABLE IF NOT EXISTS facts (
  id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
  user_id TEXT NOT NULL,

  content TEXT NOT NULL,
  namespace TEXT NOT NULL DEFAULT '[]',  -- JSON array stored as TEXT
  language TEXT NOT NULL DEFAULT 'en',
  intensity REAL CHECK (intensity IS NULL OR (intensity >= 0 AND intensity <= 1)),
  confidence REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),

  model_dimension INTEGER NOT NULL,

  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),

  -- Phase 3: Relevance scoring fields
  access_count INTEGER NOT NULL DEFAULT 0,
  last_accessed_at TEXT,
  relevance_score REAL NOT NULL DEFAULT 0.5 CHECK (relevance_score >= 0 AND relevance_score <= 1)
);

-- Basic indexes
CREATE INDEX IF NOT EXISTS facts_user_id_idx ON facts (user_id);
CREATE INDEX IF NOT EXISTS facts_dimension_idx ON facts (model_dimension);
CREATE INDEX IF NOT EXISTS facts_created_at_idx ON facts (created_at DESC);

-- Phase 3: Relevance scoring indexes
CREATE INDEX IF NOT EXISTS facts_relevance_score_idx ON facts (relevance_score DESC);
CREATE INDEX IF NOT EXISTS facts_last_accessed_idx ON facts (last_accessed_at DESC);
CREATE INDEX IF NOT EXISTS facts_access_count_idx ON facts (access_count DESC);
CREATE INDEX IF NOT EXISTS facts_user_relevance_idx ON facts (user_id, relevance_score DESC);

-- ==========================================================
-- HISTORY TABLE: fact_history
-- Tracks all changes to facts (insert, update, delete)
-- ==========================================================
CREATE TABLE IF NOT EXISTS fact_history (
  id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
  fact_id TEXT NOT NULL,
  user_id TEXT NOT NULL,

  operation TEXT NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),

  content TEXT NOT NULL,
  namespace TEXT NOT NULL,  -- JSON array as TEXT
  language TEXT NOT NULL,
  intensity REAL,
  confidence REAL,
  model_dimension INTEGER NOT NULL,

  changed_at TEXT NOT NULL DEFAULT (datetime('now')),
  changed_by TEXT,
  change_reason TEXT,
  changed_fields TEXT,  -- JSON object as TEXT
  previous_version_id TEXT REFERENCES fact_history(id)
);

CREATE INDEX IF NOT EXISTS fact_history_fact_id_idx ON fact_history (fact_id);
CREATE INDEX IF NOT EXISTS fact_history_user_id_idx ON fact_history (user_id);
CREATE INDEX IF NOT EXISTS fact_history_changed_at_idx ON fact_history (changed_at DESC);
CREATE INDEX IF NOT EXISTS fact_history_operation_idx ON fact_history (operation);

-- ==========================================================
-- PHASE 3: FACT ACCESS LOG TABLE
-- Track detailed access patterns for analytics and relevance tuning
-- ==========================================================
CREATE TABLE IF NOT EXISTS fact_access_log (
  id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
  fact_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL,

  accessed_at TEXT NOT NULL DEFAULT (datetime('now')),
  context_type TEXT NOT NULL CHECK (context_type IN ('core', 'context', 'search')),

  query_text TEXT,
  query_embedding_similarity REAL,

  was_used INTEGER DEFAULT NULL,  -- SQLite doesn't have boolean, use 0/1/NULL
  tokens_consumed INTEGER,

  thread_id TEXT,

  retrieval_rank INTEGER,
  combined_score REAL
);

CREATE INDEX IF NOT EXISTS fact_access_log_fact_id_idx ON fact_access_log (fact_id);
CREATE INDEX IF NOT EXISTS fact_access_log_user_id_idx ON fact_access_log (user_id);
CREATE INDEX IF NOT EXISTS fact_access_log_accessed_at_idx ON fact_access_log (accessed_at DESC);
CREATE INDEX IF NOT EXISTS fact_access_log_context_type_idx ON fact_access_log (context_type);

-- ==========================================================
-- PROCESSED MESSAGES TRACKING
-- Tracks which messages have been processed for fact extraction
-- ==========================================================
CREATE TABLE IF NOT EXISTS processed_messages (
  id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
  user_id TEXT NOT NULL,
  message_id TEXT NOT NULL,
  thread_id TEXT,
  processed_at TEXT NOT NULL DEFAULT (datetime('now')),

  UNIQUE (user_id, message_id)
);

CREATE INDEX IF NOT EXISTS processed_messages_user_id_idx ON processed_messages (user_id);
CREATE INDEX IF NOT EXISTS processed_messages_message_id_idx ON processed_messages (message_id);
CREATE INDEX IF NOT EXISTS processed_messages_thread_id_idx ON processed_messages (thread_id);

-- ==========================================================
-- TRIGGERS: Auto-populate fact_history on changes
-- ==========================================================

-- INSERT trigger
CREATE TRIGGER IF NOT EXISTS track_fact_insert
AFTER INSERT ON facts
FOR EACH ROW
BEGIN
  INSERT INTO fact_history (
    fact_id, user_id, operation,
    content, namespace, language, intensity, confidence, model_dimension
  ) VALUES (
    NEW.id, NEW.user_id, 'INSERT',
    NEW.content, NEW.namespace, NEW.language, NEW.intensity, NEW.confidence, NEW.model_dimension
  );
END;

-- UPDATE trigger
CREATE TRIGGER IF NOT EXISTS track_fact_update
AFTER UPDATE ON facts
FOR EACH ROW
BEGIN
  INSERT INTO fact_history (
    fact_id, user_id, operation,
    content, namespace, language, intensity, confidence, model_dimension,
    changed_fields,
    previous_version_id
  )
  VALUES (
    NEW.id, NEW.user_id, 'UPDATE',
    NEW.content, NEW.namespace, NEW.language, NEW.intensity, NEW.confidence, NEW.model_dimension,
    '{}',  -- Simplified: just store empty object for changed_fields
    (SELECT id FROM fact_history WHERE fact_id = OLD.id ORDER BY changed_at DESC LIMIT 1)
  );
END;

-- DELETE trigger
CREATE TRIGGER IF NOT EXISTS track_fact_delete
AFTER DELETE ON facts
FOR EACH ROW
BEGIN
  INSERT INTO fact_history (
    fact_id, user_id, operation,
    content, namespace, language, intensity, confidence, model_dimension,
    previous_version_id
  )
  SELECT
    OLD.id, OLD.user_id, 'DELETE',
    OLD.content, OLD.namespace, OLD.language, OLD.intensity, OLD.confidence, OLD.model_dimension,
    (SELECT id FROM fact_history WHERE fact_id = OLD.id ORDER BY changed_at DESC LIMIT 1);
END;

-- ==========================================================
-- PHASE 3: TRIGGER TO UPDATE ACCESS STATISTICS
-- Auto-increment access_count and update last_accessed_at
-- ==========================================================
CREATE TRIGGER IF NOT EXISTS update_fact_access_stats
AFTER INSERT ON fact_access_log
FOR EACH ROW
BEGIN
  UPDATE facts
  SET
    access_count = access_count + 1,
    last_accessed_at = NEW.accessed_at
  WHERE id = NEW.fact_id;
END;

-- ==========================================================
-- DYNAMIC EMBEDDING TABLES (created on-demand)
-- Tables are created programmatically per dimension size
-- Each table uses sqlite-vec virtual table for vector search
-- ==========================================================

-- Example for dimension N (created by Python code):
--
-- CREATE VIRTUAL TABLE IF NOT EXISTS fact_embeddings_{N} USING vec0(
--   fact_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
--   user_id TEXT NOT NULL,
--   embedding FLOAT[{N}],
--   created_at TEXT NOT NULL DEFAULT (datetime('now')),
--   +fact_id TEXT HIDDEN,  -- Metadata column for filtering
--   +user_id TEXT HIDDEN   -- Metadata column for filtering
-- );
--
-- Note: sqlite-vec uses + prefix for metadata columns (non-vector data)

COMMIT;

-- ==========================================================
-- POST-SETUP: USAGE INSTRUCTIONS
-- ==========================================================
-- After running this script:
--
-- 1. The facts system is fully configured with Phase 3 relevance scoring
-- 2. Embedding tables will be created automatically on first use via Python
-- 3. Relevance scores are calculated in Python (no SQL functions needed)
-- 4. Vector similarity search handled by sqlite-vec extension
--
-- Key differences from PostgreSQL version:
-- - UUIDs replaced with TEXT hex IDs
-- - JSONB replaced with TEXT (JSON strings)
-- - Timestamps use TEXT with datetime('now')
-- - Boolean was_used uses INTEGER (0/1/NULL)
-- - No SQL functions for relevance calculation (done in Python)
-- - sqlite-vec handles vector operations instead of pgvector
-- ==========================================================

-- ==========================================================
-- HARD DELETE (SQLite) - Manual Procedure
-- SQLite does not support CREATE FUNCTION in plain SQL scripts. Provide a manual
-- sequence of statements to permanently remove a fact and all related records.
-- These operations are destructive and irreversible. Run them inside a
-- transaction from your application or sqlite3 shell.
--
-- Example: hard-delete a single fact (replace placeholders)
-- BEGIN TRANSACTION;
-- -- 1) get the embedding dimension for the fact
-- SELECT model_dimension FROM facts WHERE id = 'FACT_ID' AND user_id = 'USER_ID';
-- -- 2) delete embeddings from the dimension-specific embedding table (if created)
-- -- (replace {N} with the returned model_dimension)
-- DELETE FROM fact_embeddings_{N} WHERE fact_id = 'FACT_ID';
-- -- 3) delete access logs and history
-- DELETE FROM fact_access_log WHERE fact_id = 'FACT_ID';
-- DELETE FROM fact_history WHERE fact_id = 'FACT_ID';
-- -- 4) delete the fact itself
-- DELETE FROM facts WHERE id = 'FACT_ID' AND user_id = 'USER_ID';
-- COMMIT;
--
-- Example: batch hard-delete (array of IDs)
-- BEGIN TRANSACTION;
-- -- If you know the set of dimensions involved, delete from each embedding table
-- DELETE FROM fact_embeddings_128 WHERE fact_id IN ('id1','id2',...);
-- DELETE FROM fact_embeddings_1536 WHERE fact_id IN ('id1','id2',...);
-- DELETE FROM fact_access_log WHERE fact_id IN ('id1','id2',...);
-- DELETE FROM fact_history WHERE fact_id IN ('id1','id2',...);
-- DELETE FROM facts WHERE id IN ('id1','id2',...) AND user_id = 'USER_ID';
-- COMMIT;
--
-- Note: In Python code you can automate this by reading the `model_dimension` per
-- fact, checking sqlite_master for the existence of a `fact_embeddings_{N}` table,
-- and issuing the deletes above inside a transaction.
-- ==========================================================
