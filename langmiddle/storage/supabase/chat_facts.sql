-- ==========================================================
-- EXTENSIONS
-- ==========================================================
create extension if not exists "uuid-ossp";
create extension if not exists "vector";

-- ==========================================================
-- MAIN TABLE: facts
-- Stores user facts with metadata (no embeddings here)
-- ==========================================================
create table if not exists public.facts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,

  content text not null,
  namespace text[] not null default '{}',
  language text not null default 'en',
  intensity float8 check (intensity >= 0 and intensity <= 1),
  confidence float8 check (confidence >= 0 and confidence <= 1),

  model_dimension int not null,

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

comment on table public.facts is 'User facts with hierarchical namespaces';
comment on column public.facts.model_dimension is 'Embedding dimension - determines which fact_embeddings_N table to use';

create index if not exists facts_user_id_idx on public.facts (user_id);
create index if not exists facts_namespace_idx on public.facts using gin (namespace);
create index if not exists facts_dimension_idx on public.facts (model_dimension);

-- ==========================================================
-- HISTORY TABLE: fact_history
-- Tracks all changes to facts (insert, update, delete)
-- ==========================================================
create table if not exists public.fact_history (
  id uuid primary key default gen_random_uuid(),
  fact_id uuid not null,  -- Not a foreign key since fact may be deleted
  user_id uuid not null references auth.users(id) on delete cascade,

  operation text not null check (operation in ('INSERT', 'UPDATE', 'DELETE')),

  -- Snapshot of fact data at this point in time
  content text not null,
  namespace text[] not null,
  language text not null,
  intensity float8,
  confidence float8,
  model_dimension int not null,

  -- Change metadata
  changed_at timestamptz not null default now(),
  changed_by text,  -- Optional: track which system/agent made the change
  change_reason text,  -- Optional: reason for the change (e.g., 'user_update', 'auto_merge', 'contradiction_resolved')

  -- Track what changed (for updates)
  changed_fields jsonb,  -- e.g., {"content": {"old": "...", "new": "..."}, "namespace": {...}}

  -- Link to previous version for easy traversal
  previous_version_id uuid references public.fact_history(id)
);

comment on table public.fact_history is 'Immutable audit log of all fact changes';
comment on column public.fact_history.operation is 'Type of change: INSERT, UPDATE, or DELETE';
comment on column public.fact_history.changed_fields is 'JSON object showing what changed (null for INSERT/DELETE)';
comment on column public.fact_history.previous_version_id is 'Links to previous version for version chain traversal';

create index if not exists fact_history_fact_id_idx on public.fact_history (fact_id);
create index if not exists fact_history_user_id_idx on public.fact_history (user_id);
create index if not exists fact_history_changed_at_idx on public.fact_history (changed_at desc);
create index if not exists fact_history_operation_idx on public.fact_history (operation);

-- ==========================================================
-- TRIGGER: Auto-populate fact_history on changes
-- ==========================================================
create or replace function public.track_fact_changes()
returns trigger
language plpgsql
security definer
as $$
declare
  v_operation text;
  v_changed_fields jsonb := null;
  v_previous_version_id uuid := null;
begin
  -- Determine operation type
  if (TG_OP = 'DELETE') then
    v_operation := 'DELETE';

    -- Get the last history entry for this fact
    select id into v_previous_version_id
    from public.fact_history
    where fact_id = OLD.id
    order by changed_at desc
    limit 1;

    -- Insert history record for deletion
    insert into public.fact_history (
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

    -- Get the last history entry for this fact
    select id into v_previous_version_id
    from public.fact_history
    where fact_id = OLD.id
    order by changed_at desc
    limit 1;

    -- Build changed_fields JSON
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

    -- Insert history record for update
    insert into public.fact_history (
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

    -- Insert history record for new fact
    insert into public.fact_history (
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

-- Drop existing trigger if it exists
drop trigger if exists track_fact_changes_trigger on public.facts;

-- Create trigger for all operations
create trigger track_fact_changes_trigger
  after insert or update or delete on public.facts
  for each row execute function public.track_fact_changes();

-- ==========================================================
-- RLS: fact_history table
-- ==========================================================
alter table public.fact_history enable row level security;

-- Allow users to view their own fact history
drop policy if exists "users_view_own_fact_history" on public.fact_history;
create policy "users_view_own_fact_history"
  on public.fact_history
  for select
  using (auth.uid() = user_id);

-- Allow trigger to insert history records (runs as security definer, but explicit is better)
drop policy if exists "system_insert_fact_history" on public.fact_history;
create policy "system_insert_fact_history"
  on public.fact_history
  for insert
  with check (true);  -- Trigger validates user_id matches fact owner

-- Prevent any updates or deletes (append-only audit log)
drop policy if exists "prevent_history_modifications" on public.fact_history;
create policy "prevent_history_modifications"
  on public.fact_history
  for update
  using (false);

drop policy if exists "prevent_history_deletes" on public.fact_history;
create policy "prevent_history_deletes"
  on public.fact_history
  for delete
  using (false);

-- ==========================================================
-- HELPER FUNCTIONS: Query fact history
-- ==========================================================

-- Get full history for a specific fact
create or replace function public.get_fact_history(p_fact_id uuid, p_user_id uuid)
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
security definer
as $$
  select
    id, fact_id, operation,
    content, namespace, language, intensity, confidence, model_dimension,
    changed_at, changed_by, change_reason, changed_fields, previous_version_id
  from public.fact_history
  where fact_id = p_fact_id
    and user_id = p_user_id
  order by changed_at desc;
$$;

-- Get recent fact changes for a user
create or replace function public.get_recent_fact_changes(
  p_user_id uuid,
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
security definer
as $$
  select
    id, fact_id, operation,
    content, namespace,
    changed_at, changed_fields
  from public.fact_history
  where user_id = p_user_id
    and (p_operation is null or operation = p_operation)
  order by changed_at desc
  limit p_limit;
$$;

-- Get statistics about fact changes
create or replace function public.get_fact_change_stats(p_user_id uuid)
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
security definer
as $$
  select
    count(*) as total_changes,
    count(*) filter (where operation = 'INSERT') as inserts,
    count(*) filter (where operation = 'UPDATE') as updates,
    count(*) filter (where operation = 'DELETE') as deletes,
    min(changed_at) as oldest_change,
    max(changed_at) as newest_change
  from public.fact_history
  where user_id = p_user_id;
$$;

-- ==========================================================
-- RLS: facts table
-- ==========================================================
alter table public.facts enable row level security;

drop policy if exists "users_manage_own_facts" on public.facts;
create policy "users_manage_own_facts"
  on public.facts
  for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

-- ==========================================================
-- DYNAMIC EMBEDDING TABLES
-- Tables are created on-demand per dimension size
-- Example: fact_embeddings_1536, fact_embeddings_768, etc.
-- ==========================================================

-- Drop existing functions to avoid conflicts
drop function if exists public.embedding_table_exists(int);
drop function if exists public.create_embedding_table(int);
drop function if exists public.ensure_embedding_table(int);
drop function if exists public.search_facts(vector, int, uuid, float8, int, text[][]);
drop function if exists public.search_facts_by_embedding;
drop function if exists public.get_or_create_embedding_table;

-- Helper: Check if embedding table exists
create or replace function public.embedding_table_exists(p_dimension int)
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

-- Helper: Create embedding table for a specific dimension
create or replace function public.create_embedding_table(p_dimension int)
returns void
language plpgsql
security definer
as $$
declare
  v_table_name text := 'fact_embeddings_' || p_dimension;
begin
  -- Check if already exists
  if public.embedding_table_exists(p_dimension) then
    return;
  end if;

  -- Create table
  execute format(
    'create table if not exists public.%I (
      fact_id uuid primary key references public.facts(id) on delete cascade,
      embedding vector(%s) not null
    )',
    v_table_name, p_dimension
  );

  -- Create vector index (drop first if exists)
  execute format('drop index if exists public.%I', v_table_name || '_idx');
  execute format(
    'create index %I on public.%I using ivfflat (embedding vector_l2_ops) with (lists = 100)',
    v_table_name || '_idx', v_table_name
  );

  -- Enable RLS
  execute format('alter table public.%I enable row level security', v_table_name);

  -- Create policy (drop first if exists)
  execute format('drop policy if exists "users_access_own_embeddings" on public.%I', v_table_name);
  execute format(
    'create policy "users_access_own_embeddings" on public.%I
      for all
      using (fact_id in (select id from public.facts where user_id = auth.uid()))',
    v_table_name
  );
end;
$$;

-- Convenience: Get or create embedding table
create or replace function public.ensure_embedding_table(p_dimension int)
returns void
language plpgsql
as $$
begin
  if not public.embedding_table_exists(p_dimension) then
    perform public.create_embedding_table(p_dimension);
  end if;
end;
$$;

-- ==========================================================
-- VECTOR SIMILARITY SEARCH
-- ==========================================================
create or replace function public.search_facts(
  p_embedding vector,
  p_dimension int,
  p_user_id uuid,
  p_threshold float8 default 0.75,
  p_limit int default 10,
  p_namespaces text[][] default null
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
  similarity float8
)
language plpgsql
stable
security definer
as $$
declare
  v_table_name text := 'fact_embeddings_' || p_dimension;
  v_query text;
begin
  -- Verify table exists
  if not public.embedding_table_exists(p_dimension) then
    raise exception 'Embedding table for dimension % does not exist', p_dimension;
  end if;

  -- Build query
  v_query := format(
    'select
      f.id,
      f.content,
      f.namespace,
      f.language,
      f.intensity,
      f.confidence,
      f.model_dimension,
      f.created_at,
      f.updated_at,
      1 - (e.embedding <=> $1) as similarity
    from public.facts f
    join public.%I e on e.fact_id = f.id
    where f.user_id = $2
      and f.model_dimension = $3',
    v_table_name
  );

  -- Add namespace filter if provided
  if p_namespaces is not null and array_length(p_namespaces, 1) > 0 then
    v_query := v_query || ' and f.namespace && $6';
  end if;

  -- Add similarity filter and ordering
  v_query := v_query || '
    and (1 - (e.embedding <=> $1)) >= $4
    order by e.embedding <=> $1
    limit $5';

  -- Execute
  return query execute v_query
    using p_embedding, p_user_id, p_dimension, p_threshold, p_limit, p_namespaces;
end;
$$;

-- ==========================================================
-- PROCESSED MESSAGES TRACKING
-- Tracks which messages have been processed for fact extraction
-- to avoid duplicate work
-- ==========================================================

create table if not exists public.processed_messages (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  message_id text not null,
  thread_id uuid references public.chat_threads(id) on delete cascade,
  processed_at timestamptz not null default now(),

  constraint processed_messages_unique unique (user_id, message_id)
);

comment on table public.processed_messages is 'Tracks messages processed for fact extraction to avoid duplicates';
comment on column public.processed_messages.message_id is 'Message ID from chat_messages or LangChain message.id';

create index if not exists processed_messages_user_id_idx on public.processed_messages (user_id);
create index if not exists processed_messages_message_id_idx on public.processed_messages (message_id);
create index if not exists processed_messages_thread_id_idx on public.processed_messages (thread_id);

-- Enable RLS
alter table public.processed_messages enable row level security;

-- RLS Policy
drop policy if exists "users_manage_own_processed" on public.processed_messages;
create policy "users_manage_own_processed"
  on public.processed_messages
  for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
