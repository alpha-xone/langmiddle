create table public.chat_messages (
  id text not null,
  user_id uuid not null,
  thread_id uuid not null,
  content text not null,
  role text not null,
  metadata jsonb null default '{}'::jsonb,
  created_at timestamp with time zone not null default timezone ('utc'::text, now()),
  usage_metadata jsonb null,
  constraint chat_messages_pkey primary key (id),
  constraint chat_messages_thread_id_fkey foreign KEY (thread_id) references chat_threads (id) on delete CASCADE,
  constraint chat_messages_user_id_fkey foreign KEY (user_id) references auth.users (id) on delete CASCADE,
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
) TABLESPACE pg_default;

create index IF not exists idx_chat_messages_user_id on public.chat_messages using btree (user_id) TABLESPACE pg_default;

create index IF not exists idx_chat_messages_thread_id on public.chat_messages using btree (thread_id) TABLESPACE pg_default;

create index IF not exists idx_chat_messages_created_at on public.chat_messages using btree (created_at) TABLESPACE pg_default;

-- Enable RLS
alter table public.chat_messages enable row level security;

-- RLS Policy
drop policy if exists "users_manage_own_messages" on public.chat_messages;
create policy "users_manage_own_messages"
  on public.chat_messages
  for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
