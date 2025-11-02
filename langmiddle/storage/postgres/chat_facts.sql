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
  user_id uuid not null,

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
  user_id uuid not null,
  message_id text not null,
  thread_id uuid,
  processed_at timestamptz not null default now(),

  constraint processed_messages_unique unique (user_id, message_id)
);

comment on table public.processed_messages is 'Tracks messages processed for fact extraction to avoid duplicates';
comment on column public.processed_messages.message_id is 'Message ID from chat_messages or LangChain message.id';

create index if not exists processed_messages_user_id_idx on public.processed_messages (user_id);
create index if not exists processed_messages_message_id_idx on public.processed_messages (message_id);
create index if not exists processed_messages_thread_id_idx on public.processed_messages (thread_id);

-- Add foreign key constraint if chat_threads table exists
do $$
begin
  if exists (select 1 from information_schema.tables where table_schema = 'public' and table_name = 'chat_threads') then
    execute 'alter table public.processed_messages
             add constraint processed_messages_thread_id_fkey
             foreign key (thread_id) references public.chat_threads(id) on delete cascade';
  end if;
exception
  when duplicate_object then null;
end $$;
