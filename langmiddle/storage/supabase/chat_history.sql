-- ==========================================================
-- CHAT HISTORY TABLES
-- Stores conversation threads and messages
-- ==========================================================

-- Create trigger function (if not exists)
create or replace function public.update_updated_at_column()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = timezone('utc'::text, now());
  return new;
end;
$$;

-- ==========================================================
-- TABLE: chat_threads
-- ==========================================================
create table if not exists public.chat_threads (
  id uuid not null default gen_random_uuid (),
  user_id uuid not null,
  title text not null default ''::text,
  metadata jsonb null default '{}'::jsonb,
  created_at timestamp with time zone not null default timezone ('utc'::text, now()),
  updated_at timestamp with time zone not null default timezone ('utc'::text, now()),
  custom_state jsonb null default '{}'::jsonb,
  constraint chat_threads_pkey primary key (id),
  constraint chat_threads_user_id_fkey foreign key (user_id) references auth.users (id) on delete cascade
) tablespace pg_default;

-- Indexes for chat_threads
create index if not exists idx_chat_threads_user_id on public.chat_threads using btree (user_id) tablespace pg_default;
create index if not exists idx_chat_threads_updated_at on public.chat_threads using btree (updated_at) tablespace pg_default;

-- Trigger for chat_threads
drop trigger if exists update_chat_threads_updated_at on public.chat_threads;
create trigger update_chat_threads_updated_at before update on public.chat_threads
  for each row execute function public.update_updated_at_column();

-- Enable RLS for chat_threads
alter table public.chat_threads enable row level security;

-- RLS Policy for chat_threads
drop policy if exists "users_manage_own_threads" on public.chat_threads;
create policy "users_manage_own_threads"
  on public.chat_threads
  for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

-- ==========================================================
-- TABLE: chat_messages
-- ==========================================================
create table if not exists public.chat_messages (
  id text not null,
  user_id uuid not null,
  thread_id uuid not null,
  content text not null,
  role text not null,
  metadata jsonb null default '{}'::jsonb,
  created_at timestamp with time zone not null default timezone ('utc'::text, now()),
  usage_metadata jsonb null,
  constraint chat_messages_pkey primary key (id),
  constraint chat_messages_thread_id_fkey foreign key (thread_id) references public.chat_threads (id) on delete cascade,
  constraint chat_messages_user_id_fkey foreign key (user_id) references auth.users (id) on delete cascade,
  constraint chat_messages_role_check check (
    (
      role = any (
        array[
          'user'::text,
          'human'::text,
          'assistant'::text,
          'ai'::text,
          'tool'::text,
          'system'::text
        ]
      )
    )
  )
) tablespace pg_default;

-- Indexes for chat_messages
create index if not exists idx_chat_messages_user_id on public.chat_messages using btree (user_id) tablespace pg_default;
create index if not exists idx_chat_messages_thread_id on public.chat_messages using btree (thread_id) tablespace pg_default;
create index if not exists idx_chat_messages_created_at on public.chat_messages using btree (created_at) tablespace pg_default;

-- Enable RLS for chat_messages
alter table public.chat_messages enable row level security;

-- RLS Policy for chat_messages
drop policy if exists "users_manage_own_messages" on public.chat_messages;
create policy "users_manage_own_messages"
  on public.chat_messages
  for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
