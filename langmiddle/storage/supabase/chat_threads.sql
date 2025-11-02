create table public.chat_threads (
  id uuid not null default gen_random_uuid (),
  user_id uuid not null,
  title text not null default ''::text,
  metadata jsonb null default '{}'::jsonb,
  created_at timestamp with time zone not null default timezone ('utc'::text, now()),
  updated_at timestamp with time zone not null default timezone ('utc'::text, now()),
  custom_state jsonb null default '{}'::jsonb,
  constraint chat_threads_pkey primary key (id),
  constraint chat_threads_user_id_fkey foreign KEY (user_id) references auth.users (id) on delete CASCADE
) TABLESPACE pg_default;

create index IF not exists idx_chat_threads_user_id on public.chat_threads using btree (user_id) TABLESPACE pg_default;

create index IF not exists idx_chat_threads_updated_at on public.chat_threads using btree (updated_at) TABLESPACE pg_default;

create trigger update_chat_threads_updated_at BEFORE
update on chat_threads for EACH row
execute FUNCTION update_updated_at_column ();

-- Enable RLS
alter table public.chat_threads enable row level security;

-- RLS Policy
drop policy if exists "users_manage_own_threads" on public.chat_threads;
create policy "users_manage_own_threads"
  on public.chat_threads
  for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
