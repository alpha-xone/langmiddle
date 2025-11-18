-- ==========================================================
-- COMPLETE FACTS SCHEMA WITH PHASE 3 RELEVANCE SCORING
-- PostgreSQL version (no RLS, no auth.users references)
-- For fresh database setup - includes all tables, functions, and features
-- ==========================================================

BEGIN;

-- ==========================================================
-- EXTENSIONS
-- ==========================================================
create extension if not exists "uuid-ossp";
create extension if not exists "vector";

-- ==========================================================
-- MAIN TABLE: facts
-- Stores user facts with metadata and relevance tracking
-- ==========================================================
create table if not exists facts (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,

  content text not null,
  namespace text[] not null default '{}',
  language text not null default 'en',
  intensity float8 check (intensity >= 0 and intensity <= 1),
  confidence float8 check (confidence >= 0 and confidence <= 1),

  model_dimension int not null,

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  -- Phase 3: Relevance scoring fields
  access_count int not null default 0,
  last_accessed_at timestamptz,
  relevance_score float8 not null default 0.5 check (relevance_score >= 0 and relevance_score <= 1)
);

comment on table facts is 'User facts with hierarchical namespaces and relevance scoring';
comment on column facts.model_dimension is 'Embedding dimension - determines which fact_embeddings_N table to use';
comment on column facts.access_count is 'Number of times this fact has been retrieved/injected into context';
comment on column facts.last_accessed_at is 'Timestamp of most recent fact retrieval';
comment on column facts.relevance_score is 'Computed relevance score (0-1) based on recency, access patterns, and usage feedback';

-- Basic indexes
create index if not exists facts_user_id_idx on facts (user_id);
create index if not exists facts_namespace_idx on facts using gin (namespace);
create index if not exists facts_dimension_idx on facts (model_dimension);

-- Phase 3: Relevance scoring indexes
create index if not exists facts_relevance_score_idx on facts (relevance_score desc);
create index if not exists facts_last_accessed_idx on facts (last_accessed_at desc nulls last);
create index if not exists facts_access_count_idx on facts (access_count desc);
create index if not exists facts_user_relevance_idx on facts (user_id, relevance_score desc);

-- ==========================================================
-- HISTORY TABLE: fact_history
-- Tracks all changes to facts (insert, update, delete)
-- ==========================================================
create table if not exists fact_history (
  id uuid primary key default gen_random_uuid(),
  fact_id uuid not null,
  user_id text not null,

  operation text not null check (operation in ('INSERT', 'UPDATE', 'DELETE')),

  content text not null,
  namespace text[] not null,
  language text not null,
  intensity float8,
  confidence float8,
  model_dimension int not null,

  changed_at timestamptz not null default now(),
  changed_by text,
  change_reason text,
  changed_fields jsonb,
  previous_version_id uuid references fact_history(id)
);

comment on table fact_history is 'Immutable audit log of all fact changes';
comment on column fact_history.operation is 'Type of change: INSERT, UPDATE, or DELETE';
comment on column fact_history.changed_fields is 'JSON object showing what changed (null for INSERT/DELETE)';
comment on column fact_history.previous_version_id is 'Links to previous version for version chain traversal';

create index if not exists fact_history_fact_id_idx on fact_history (fact_id);
create index if not exists fact_history_user_id_idx on fact_history (user_id);
create index if not exists fact_history_changed_at_idx on fact_history (changed_at desc);
create index if not exists fact_history_operation_idx on fact_history (operation);

-- ==========================================================
-- PHASE 3: FACT ACCESS LOG TABLE
-- Track detailed access patterns for analytics and relevance tuning
-- ==========================================================
create table if not exists fact_access_log (
  id uuid primary key default gen_random_uuid(),
  fact_id uuid not null references facts(id) on delete cascade,
  user_id text not null,

  accessed_at timestamptz not null default now(),
  context_type text not null check (context_type in ('core', 'context', 'search')),

  query_text text,
  query_embedding_similarity float8,

  was_used boolean default null,
  tokens_consumed int,

  thread_id uuid,

  retrieval_rank int,
  combined_score float8
);

comment on table fact_access_log is 'Detailed tracking of fact retrievals for relevance tuning and analytics';
comment on column fact_access_log.context_type is 'Type of retrieval: core (always loaded), context (query-based), search (explicit search)';
comment on column fact_access_log.was_used is 'Implicit feedback: whether fact appeared in agent response (NULL = unknown)';
comment on column fact_access_log.combined_score is 'Final relevance score at retrieval time (similarity + relevance_score)';

create index if not exists fact_access_log_fact_id_idx on fact_access_log (fact_id);
create index if not exists fact_access_log_user_id_idx on fact_access_log (user_id);
create index if not exists fact_access_log_accessed_at_idx on fact_access_log (accessed_at desc);
create index if not exists fact_access_log_context_type_idx on fact_access_log (context_type);

-- ==========================================================
-- PROCESSED MESSAGES TRACKING
-- Tracks which messages have been processed for fact extraction
-- ==========================================================
create table if not exists processed_messages (
  id uuid primary key default gen_random_uuid(),
  user_id text not null,
  message_id text not null,
  thread_id uuid,
  processed_at timestamptz not null default now(),

  constraint processed_messages_unique unique (user_id, message_id)
);

comment on table processed_messages is 'Tracks messages processed for fact extraction to avoid duplicates';
comment on column processed_messages.message_id is 'Message ID from chat_messages or LangChain message.id';

create index if not exists processed_messages_user_id_idx on processed_messages (user_id);
create index if not exists processed_messages_message_id_idx on processed_messages (message_id);
create index if not exists processed_messages_thread_id_idx on processed_messages (thread_id);

-- ==========================================================
-- TRIGGER: Auto-populate fact_history on changes
-- ==========================================================
create or replace function track_fact_changes()
returns trigger
language plpgsql
as $$
declare
  v_operation text;
  v_changed_fields jsonb := null;
  v_previous_version_id uuid := null;
begin
  if (TG_OP = 'DELETE') then
    v_operation := 'DELETE';

    select id into v_previous_version_id
    from fact_history
    where fact_id = OLD.id
    order by changed_at desc
    limit 1;

    insert into fact_history (
      fact_id, user_id, operation,
      content, namespace, language, intensity, confidence, model_dimension,
      previous_version_id
    ) values (
      OLD.id, OLD.user_id, v_operation,
      OLD.content, OLD.namespace, OLD.language, OLD.intensity, OLD.confidence, OLD.model_dimension,
      v_previous_version_id
    );

    return OLD;

  elsif (TG_OP = 'UPDATE') then
    v_operation := 'UPDATE';

    select id into v_previous_version_id
    from fact_history
    where fact_id = OLD.id
    order by changed_at desc
    limit 1;

    v_changed_fields := jsonb_build_object();

    if OLD.content != NEW.content then
      v_changed_fields := v_changed_fields || jsonb_build_object(
        'content', jsonb_build_object('old', OLD.content, 'new', NEW.content)
      );
    end if;

    if OLD.namespace != NEW.namespace then
      v_changed_fields := v_changed_fields || jsonb_build_object(
        'namespace', jsonb_build_object('old', OLD.namespace, 'new', NEW.namespace)
      );
    end if;

    if OLD.language != NEW.language then
      v_changed_fields := v_changed_fields || jsonb_build_object(
        'language', jsonb_build_object('old', OLD.language, 'new', NEW.language)
      );
    end if;

    if OLD.intensity is distinct from NEW.intensity then
      v_changed_fields := v_changed_fields || jsonb_build_object(
        'intensity', jsonb_build_object('old', OLD.intensity, 'new', NEW.intensity)
      );
    end if;

    if OLD.confidence is distinct from NEW.confidence then
      v_changed_fields := v_changed_fields || jsonb_build_object(
        'confidence', jsonb_build_object('old', OLD.confidence, 'new', NEW.confidence)
      );
    end if;

    if OLD.model_dimension != NEW.model_dimension then
      v_changed_fields := v_changed_fields || jsonb_build_object(
        'model_dimension', jsonb_build_object('old', OLD.model_dimension, 'new', NEW.model_dimension)
      );
    end if;

    insert into fact_history (
      fact_id, user_id, operation,
      content, namespace, language, intensity, confidence, model_dimension,
      changed_fields, previous_version_id
    ) values (
      NEW.id, NEW.user_id, v_operation,
      NEW.content, NEW.namespace, NEW.language, NEW.intensity, NEW.confidence, NEW.model_dimension,
      v_changed_fields, v_previous_version_id
    );

    return NEW;

  elsif (TG_OP = 'INSERT') then
    v_operation := 'INSERT';

    insert into fact_history (
      fact_id, user_id, operation,
      content, namespace, language, intensity, confidence, model_dimension
    ) values (
      NEW.id, NEW.user_id, v_operation,
      NEW.content, NEW.namespace, NEW.language, NEW.intensity, NEW.confidence, NEW.model_dimension
    );

    return NEW;
  end if;

  return null;
end;
$$;

drop trigger if exists track_fact_changes_trigger on facts;
create trigger track_fact_changes_trigger
  after insert or update or delete on facts
  for each row execute function track_fact_changes();

-- ==========================================================
-- PHASE 3: TRIGGER TO UPDATE ACCESS STATISTICS
-- Auto-increment access_count and update last_accessed_at
-- ==========================================================
create or replace function update_fact_access_stats()
returns trigger
language plpgsql
as $$
begin
  update facts
  set
    access_count = access_count + 1,
    last_accessed_at = NEW.accessed_at
  where id = NEW.fact_id;

  return NEW;
end;
$$;

drop trigger if exists update_fact_access_stats_trigger on fact_access_log;
create trigger update_fact_access_stats_trigger
  after insert on fact_access_log
  for each row
  execute function update_fact_access_stats();

-- ==========================================================
-- DYNAMIC EMBEDDING TABLES
-- Tables are created on-demand per dimension size
-- ==========================================================

drop function if exists embedding_table_exists(int);
drop function if exists create_embedding_table(int);
drop function if exists ensure_embedding_table(int);
drop function if exists search_facts(vector, int, text, float8, int, jsonb, float8, float8);
drop function if exists search_facts(vector, int, text, float8, int, text[][]);

create or replace function embedding_table_exists(p_dimension int)
returns boolean
language sql
stable
as $$
  select exists (
    select 1
    from information_schema.tables
    where table_schema = 'public'
      and table_name = 'fact_embeddings_' || p_dimension
  );
$$;

create or replace function create_embedding_table(p_dimension int)
returns void
language plpgsql
as $$
declare
  v_table_name text := 'fact_embeddings_' || p_dimension;
begin
  if embedding_table_exists(p_dimension) then
    return;
  end if;

  execute format(
    'create table if not exists %I (
      id uuid primary key default gen_random_uuid(),
      fact_id uuid not null references facts(id) on delete cascade,
      user_id text not null,
      embedding vector(%s) not null,
      created_at timestamptz not null default now(),
      constraint %I unique (fact_id, user_id)
    )',
    v_table_name, p_dimension, v_table_name || '_unique'
  );

  execute format('drop index if exists %I', v_table_name || '_idx');
  execute format(
    'create index %I on %I using ivfflat (embedding vector_l2_ops) with (lists = 100)',
    v_table_name || '_idx', v_table_name
  );

  execute format('create index %I on %I (fact_id)', v_table_name || '_fact_id_idx', v_table_name);
  execute format('create index %I on %I (user_id, fact_id)', v_table_name || '_user_fact_idx', v_table_name);
  execute format('create index %I on %I (created_at desc)', v_table_name || '_created_at_idx', v_table_name);
end;
$$;

create or replace function ensure_embedding_table(p_dimension int)
returns void
language plpgsql
as $$
begin
  if not embedding_table_exists(p_dimension) then
    perform create_embedding_table(p_dimension);
  end if;
end;
$$;

-- ==========================================================
-- PHASE 3: ENHANCED VECTOR SIMILARITY SEARCH WITH RELEVANCE SCORING
-- Combined score = (similarity * similarity_weight) + (relevance_score * relevance_weight)
-- ==========================================================
create or replace function search_facts(
  p_embedding vector,
  p_dimension int,
  p_user_id text,
  p_threshold float8 default 0.75,
  p_limit int default 10,
  p_namespaces jsonb default null,
  p_similarity_weight float8 default 0.7,
  p_relevance_weight float8 default 0.3
)
returns table (
  id uuid,
  content text,
  namespace text[],
  language text,
  intensity float8,
  confidence float8,
  model_dimension int,
  created_at timestamptz,
  updated_at timestamptz,
  similarity float8,
  relevance_score float8,
  combined_score float8,
  access_count int,
  last_accessed_at timestamptz
)
language plpgsql
stable
as $$
declare
  v_table_name text := 'fact_embeddings_' || p_dimension;
  v_query text;
begin
  if not embedding_table_exists(p_dimension) then
    raise exception 'Embedding table for dimension % does not exist', p_dimension;
  end if;

  v_query := format(
    'with ranked_results as (
      select
        f.id,
        f.content,
        f.namespace,
        f.language,
        f.intensity,
        f.confidence,
        f.model_dimension,
        f.created_at,
        f.updated_at,
        f.relevance_score,
        f.access_count,
        f.last_accessed_at,
        (1 - (e.embedding <=> $1)) as similarity,
        ((1 - (e.embedding <=> $1)) * $7 + f.relevance_score * $8) as combined_score,
        row_number() over (partition by f.id order by (1 - (e.embedding <=> $1)) desc) as rn
      from facts f
      inner join %I e on f.id = e.fact_id
      where e.user_id = $2
        and f.model_dimension = $3',
    v_table_name
  );

  if p_namespaces is not null and jsonb_array_length(p_namespaces) > 0 then
    v_query := v_query || '
        and exists (
          select 1
          from jsonb_array_elements($6) as ns
          where f.namespace @> (select array_agg(elem::text) from jsonb_array_elements_text(ns) as elem)
        )';
  end if;

  v_query := v_query || '
        and (1 - (e.embedding <=> $1)) >= $4
    )
    select
      id, content, namespace, language, intensity, confidence,
      model_dimension, created_at, updated_at,
      similarity, relevance_score, combined_score,
      access_count, last_accessed_at
    from ranked_results
    where rn = 1
    order by combined_score desc, similarity desc
    limit $5';

  return query execute v_query
    using p_embedding, p_user_id, p_dimension, p_threshold, p_limit,
          p_namespaces, p_similarity_weight, p_relevance_weight;
end;
$$;

comment on function search_facts is 'Enhanced vector similarity search with dynamic relevance scoring (70% similarity + 30% relevance by default)';

-- ==========================================================
-- PHASE 3: RELEVANCE SCORE CALCULATION
-- Combines recency (40%), access frequency (30%), and usage rate (30%)
-- ==========================================================
create or replace function calculate_relevance_score(
  p_fact_id uuid,
  p_recency_weight float8 default 0.4,
  p_access_weight float8 default 0.3,
  p_usage_weight float8 default 0.3
)
returns float8
language plpgsql
stable
as $$
declare
  v_created_at timestamptz;
  v_access_count int;
  v_usage_rate float8;
  v_max_age_days float8 := 365;
  v_max_access_count int := 100;

  v_recency_score float8;
  v_access_score float8;
  v_usage_score float8;
  v_final_score float8;
begin
  select f.created_at, f.access_count
  into v_created_at, v_access_count
  from facts f
  where f.id = p_fact_id;

  if not found then
    return 0.5;
  end if;

  v_recency_score := exp(-extract(epoch from (now() - v_created_at)) / (v_max_age_days * 86400));
  v_access_score := least(v_access_count::float8 / v_max_access_count, 1.0);

  select
    coalesce(
      count(*) filter (where was_used = true)::float8 / nullif(count(*), 0),
      0.5
    )
  into v_usage_score
  from fact_access_log
  where fact_id = p_fact_id
    and was_used is not null;

  v_final_score :=
    (v_recency_score * p_recency_weight) +
    (v_access_score * p_access_weight) +
    (v_usage_score * p_usage_weight);

  return least(greatest(v_final_score, 0.0), 1.0);
end;
$$;

comment on function calculate_relevance_score is 'Calculate dynamic relevance score combining recency (40%), access patterns (30%), and usage feedback (30%)';

-- ==========================================================
-- PHASE 3: BATCH RELEVANCE SCORE REFRESH
-- ==========================================================
create or replace function refresh_all_relevance_scores()
returns table (
  updated_count bigint,
  avg_score float8,
  min_score float8,
  max_score float8
)
language plpgsql
as $$
declare
  v_updated_count bigint;
  v_avg_score float8;
  v_min_score float8;
  v_max_score float8;
begin
  with updated as (
    update facts f
    set relevance_score = calculate_relevance_score(f.id)
    returning relevance_score
  )
  select
    count(*),
    avg(relevance_score),
    min(relevance_score),
    max(relevance_score)
  into v_updated_count, v_avg_score, v_min_score, v_max_score
  from updated;

  return query select v_updated_count, v_avg_score, v_min_score, v_max_score;
end;
$$;

comment on function refresh_all_relevance_scores is 'Recalculate relevance scores for all facts (run periodically via scheduler)';

-- ==========================================================
-- PHASE 3: HELPER FUNCTIONS FOR ACCESS LOGGING
-- ==========================================================
create or replace function log_fact_access(
  p_fact_ids uuid[],
  p_user_id text,
  p_context_type text,
  p_query_text text default null,
  p_thread_id uuid default null
)
returns int
language plpgsql
as $$
declare
  v_fact_id uuid;
  v_count int := 0;
begin
  foreach v_fact_id in array p_fact_ids
  loop
    insert into fact_access_log (
      fact_id, user_id, context_type, query_text, thread_id
    )
    values (
      v_fact_id, p_user_id, p_context_type, p_query_text, p_thread_id
    );

    v_count := v_count + 1;
  end loop;

  return v_count;
end;
$$;

comment on function log_fact_access is 'Bulk insert fact access records for tracking';

create or replace function update_fact_usage_feedback(
  p_access_log_ids uuid[],
  p_was_used boolean
)
returns int
language plpgsql
as $$
declare
  v_updated int;
begin
  update fact_access_log
  set was_used = p_was_used
  where id = any(p_access_log_ids);

  get diagnostics v_updated = row_count;
  return v_updated;
end;
$$;

comment on function update_fact_usage_feedback is 'Update usage feedback for fact access log entries';

-- ==========================================================
-- HELPER FUNCTIONS: Query fact history
-- ==========================================================
create or replace function get_fact_history(p_fact_id uuid, p_user_id text)
returns table (
  id uuid,
  fact_id uuid,
  operation text,
  content text,
  namespace text[],
  language text,
  intensity float8,
  confidence float8,
  model_dimension int,
  changed_at timestamptz,
  changed_by text,
  change_reason text,
  changed_fields jsonb,
  previous_version_id uuid
)
language sql
stable
as $$
  select
    id, fact_id, operation,
    content, namespace, language, intensity, confidence, model_dimension,
    changed_at, changed_by, change_reason, changed_fields, previous_version_id
  from fact_history
  where fact_id = p_fact_id
    and user_id = p_user_id
  order by changed_at desc;
$$;

create or replace function get_recent_fact_changes(
  p_user_id text,
  p_limit int default 50,
  p_operation text default null
)
returns table (
  id uuid,
  fact_id uuid,
  operation text,
  content text,
  namespace text[],
  changed_at timestamptz,
  changed_fields jsonb
)
language sql
stable
as $$
  select
    id, fact_id, operation,
    content, namespace,
    changed_at, changed_fields
  from fact_history
  where user_id = p_user_id
    and (p_operation is null or operation = p_operation)
  order by changed_at desc
  limit p_limit;
$$;

create or replace function get_fact_change_stats(p_user_id text)
returns table (
  total_changes bigint,
  inserts bigint,
  updates bigint,
  deletes bigint,
  oldest_change timestamptz,
  newest_change timestamptz
)
language sql
stable
as $$
  select
    count(*) as total_changes,
    count(*) filter (where operation = 'INSERT') as inserts,
    count(*) filter (where operation = 'UPDATE') as updates,
    count(*) filter (where operation = 'DELETE') as deletes,
    min(changed_at) as oldest_change,
    max(changed_at) as newest_change
  from fact_history
  where user_id = p_user_id;
$$;

COMMIT;

-- ==========================================================
-- POST-SETUP: USAGE INSTRUCTIONS
-- ==========================================================
-- After running this script:
--
-- 1. The facts system is fully configured with Phase 3 relevance scoring
-- 2. Embedding tables will be created automatically on first use
-- 3. To calculate initial relevance scores:
--    SELECT * FROM refresh_all_relevance_scores();
--
-- 4. Optional: Schedule periodic score refresh with pg_cron:
--    SELECT cron.schedule(
--      'refresh-relevance',
--      '0 3 * * *',
--      'SELECT refresh_all_relevance_scores()'
--    );
--
-- Features included:
-- - Facts storage with hierarchical namespaces
-- - Dynamic embedding tables (auto-created per dimension)
-- - Fact history tracking (audit log)
-- - Phase 3: Relevance scoring (access tracking, usage feedback)
-- - Phase 3: Enhanced search with combined scoring
-- - PostgreSQL-compatible (no RLS, text user_id)
-- ==========================================================
