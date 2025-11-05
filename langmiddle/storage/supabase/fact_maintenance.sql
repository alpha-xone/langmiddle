-- ==========================================================
-- FACT MAINTENANCE AND DEDUPLICATION
-- Server-side functions for periodic fact cleanup and merging
-- Run this alongside chat_facts.sql when enable_facts == True
-- ==========================================================

-- ==========================================================
-- HELPER: Find similar memories based on embedding similarity
-- ==========================================================
create or replace function public.find_similar_memories(
  p_user_id uuid,
  p_dimension int,
  dream_depth float8 default 0.90,
  p_limit int default 100
)
returns table (
  fact_id_1 uuid,
  fact_id_2 uuid,
  content_1 text,
  content_2 text,
  namespace_1 text[],
  namespace_2 text[],
  similarity float8,
  created_at_1 timestamptz,
  created_at_2 timestamptz
)
language plpgsql
stable
security definer
as $$
declare
  v_table_name text := 'fact_embeddings_' || p_dimension;
begin
  -- Verify table exists
  if not public.embedding_table_exists(p_dimension) then
    raise exception 'Embedding table for dimension % does not exist', p_dimension;
  end if;

  -- Find pairs of facts with high similarity
  return query execute format(
    'select
      f1.id as fact_id_1,
      f2.id as fact_id_2,
      f1.content as content_1,
      f2.content as content_2,
      f1.namespace as namespace_1,
      f2.namespace as namespace_2,
      1 - (e1.embedding <=> e2.embedding) as similarity,
      f1.created_at as created_at_1,
      f2.created_at as created_at_2
    from public.facts f1
    join public.%I e1 on e1.fact_id = f1.id
    join public.facts f2 on f2.user_id = f1.user_id and f2.id > f1.id
    join public.%I e2 on e2.fact_id = f2.id
    where f1.user_id = $1
      and f1.model_dimension = $2
      and f2.model_dimension = $2
      and (1 - (e1.embedding <=> e2.embedding)) >= $3
    order by similarity desc
    limit $4',
    v_table_name, v_table_name
  ) using p_user_id, p_dimension, dream_depth, p_limit;
end;
$$;

comment on function public.find_similar_memories is 'Find pairs of memories with high embedding similarity (like finding related thoughts during dreaming)';


-- ==========================================================
-- CORE: Merge two facts and combine their history
-- ==========================================================
create or replace function public.merge_facts(
  p_fact_id_keep uuid,
  p_fact_id_merge uuid,
  p_user_id uuid,
  p_merge_strategy text default 'keep_older',  -- 'keep_older', 'keep_newer', 'combine'
  p_merge_reason text default 'automatic_deduplication'
)
returns jsonb
language plpgsql
security definer
as $$
declare
  v_fact_keep record;
  v_fact_merge record;
  v_merged_content text;
  v_merged_namespace text[];
  v_merged_intensity float8;
  v_merged_confidence float8;
  v_history_count int;
  v_embedding_table text;
  v_result jsonb;
begin
  -- Fetch both facts
  select * into v_fact_keep from public.facts where id = p_fact_id_keep and user_id = p_user_id;
  select * into v_fact_merge from public.facts where id = p_fact_id_merge and user_id = p_user_id;

  -- Validate both facts exist
  if v_fact_keep is null or v_fact_merge is null then
    return jsonb_build_object(
      'success', false,
      'error', 'One or both facts not found or not owned by user'
    );
  end if;

  -- Validate same dimension
  if v_fact_keep.model_dimension != v_fact_merge.model_dimension then
    return jsonb_build_object(
      'success', false,
      'error', 'Facts have different embedding dimensions'
    );
  end if;

  -- Determine merged values based on strategy
  case p_merge_strategy
    when 'keep_older' then
      if v_fact_keep.created_at <= v_fact_merge.created_at then
        v_merged_content := v_fact_keep.content;
        v_merged_namespace := v_fact_keep.namespace;
        v_merged_intensity := v_fact_keep.intensity;
        v_merged_confidence := v_fact_keep.confidence;
      else
        v_merged_content := v_fact_merge.content;
        v_merged_namespace := v_fact_merge.namespace;
        v_merged_intensity := v_fact_merge.intensity;
        v_merged_confidence := v_fact_merge.confidence;
      end if;
    when 'keep_newer' then
      if v_fact_keep.created_at >= v_fact_merge.created_at then
        v_merged_content := v_fact_keep.content;
        v_merged_namespace := v_fact_keep.namespace;
        v_merged_intensity := v_fact_keep.intensity;
        v_merged_confidence := v_fact_keep.confidence;
      else
        v_merged_content := v_fact_merge.content;
        v_merged_namespace := v_fact_merge.namespace;
        v_merged_intensity := v_fact_merge.intensity;
        v_merged_confidence := v_fact_merge.confidence;
      end if;
    when 'combine' then
      -- Combine content (longer one wins)
      v_merged_content := case
        when length(v_fact_keep.content) >= length(v_fact_merge.content)
        then v_fact_keep.content
        else v_fact_merge.content
      end;
      -- Use more specific namespace (longer array)
      v_merged_namespace := case
        when array_length(v_fact_keep.namespace, 1) >= array_length(v_fact_merge.namespace, 1)
        then v_fact_keep.namespace
        else v_fact_merge.namespace
      end;
      -- Average intensity and confidence
      v_merged_intensity := (coalesce(v_fact_keep.intensity, 0.5) + coalesce(v_fact_merge.intensity, 0.5)) / 2.0;
      v_merged_confidence := (coalesce(v_fact_keep.confidence, 0.5) + coalesce(v_fact_merge.confidence, 0.5)) / 2.0;
    else
      return jsonb_build_object(
        'success', false,
        'error', 'Invalid merge strategy'
      );
  end case;

  -- Begin transaction-like operations
  -- Step 1: Update fact_history to link merged fact's history to kept fact
  update public.fact_history
  set fact_id = p_fact_id_keep,
      change_reason = coalesce(change_reason, '') || ' [merged from ' || p_fact_id_merge || ']'
  where fact_id = p_fact_id_merge;

  get diagnostics v_history_count = row_count;

  -- Step 2: Create history record for the merge operation on kept fact
  insert into public.fact_history (
    fact_id, user_id, operation,
    content, namespace, language, intensity, confidence, model_dimension,
    change_reason,
    changed_fields
  ) values (
    p_fact_id_keep, p_user_id, 'UPDATE',
    v_merged_content, v_merged_namespace, v_fact_keep.language,
    v_merged_intensity, v_merged_confidence, v_fact_keep.model_dimension,
    p_merge_reason,
    jsonb_build_object(
      'merged_from', p_fact_id_merge,
      'merge_strategy', p_merge_strategy,
      'original_content', v_fact_keep.content,
      'merged_content', v_merged_content
    )
  );

  -- Step 3: Update the kept fact with merged values
  update public.facts
  set content = v_merged_content,
      namespace = v_merged_namespace,
      intensity = v_merged_intensity,
      confidence = v_merged_confidence,
      updated_at = now()
  where id = p_fact_id_keep;

  -- Step 4: Delete the merged fact (will cascade to embeddings and trigger history)
  delete from public.facts where id = p_fact_id_merge;

  -- Return success result
  v_result := jsonb_build_object(
    'success', true,
    'fact_id_kept', p_fact_id_keep,
    'fact_id_merged', p_fact_id_merge,
    'history_records_moved', v_history_count,
    'merged_content', v_merged_content,
    'merged_namespace', v_merged_namespace
  );

  return v_result;

exception
  when others then
    return jsonb_build_object(
      'success', false,
      'error', SQLERRM
    );
end;
$$;

comment on function public.merge_facts is 'Merge two duplicate facts, combining their history and preserving lineage';


-- ==========================================================
-- BATCH: Process multiple duplicate pairs
-- ==========================================================
create or replace function public.consolidate_memories(
  p_user_id uuid,
  p_dimension int,
  dream_depth float8 default 0.92,
  memories_per_cycle int default 50,
  p_merge_strategy text default 'keep_older'
)
returns jsonb
language plpgsql
security definer
as $$
declare
  v_duplicate record;
  v_merge_result jsonb;
  v_merged_count int := 0;
  v_failed_count int := 0;
  v_errors jsonb := '[]'::jsonb;
  v_merged_pairs jsonb := '[]'::jsonb;
begin
  -- Find and merge duplicates
  for v_duplicate in (
    select * from public.find_similar_memories(
      p_user_id,
      p_dimension,
      dream_depth,
      memories_per_cycle * 2  -- Get more candidates since some may fail
    )
  ) loop
    -- Stop if we've hit the limit
    exit when v_merged_count >= memories_per_cycle;

    -- Decide which fact to keep (older one by default)
    declare
      v_fact_keep uuid;
      v_fact_merge uuid;
    begin
      if v_duplicate.created_at_1 <= v_duplicate.created_at_2 then
        v_fact_keep := v_duplicate.fact_id_1;
        v_fact_merge := v_duplicate.fact_id_2;
      else
        v_fact_keep := v_duplicate.fact_id_2;
        v_fact_merge := v_duplicate.fact_id_1;
      end if;

      -- Attempt merge
      v_merge_result := public.merge_facts(
        v_fact_keep,
        v_fact_merge,
        p_user_id,
        p_merge_strategy,
        'batch_deduplication similarity=' || round(v_duplicate.similarity::numeric, 3)::text
      );

      if (v_merge_result->>'success')::boolean then
        v_merged_count := v_merged_count + 1;
        v_merged_pairs := v_merged_pairs || jsonb_build_object(
          'kept', v_fact_keep,
          'merged', v_fact_merge,
          'similarity', v_duplicate.similarity
        );
      else
        v_failed_count := v_failed_count + 1;
        v_errors := v_errors || jsonb_build_object(
          'fact_id_1', v_duplicate.fact_id_1,
          'fact_id_2', v_duplicate.fact_id_2,
          'error', v_merge_result->>'error'
        );
      end if;
    end;
  end loop;

  return jsonb_build_object(
    'success', true,
    'merged_count', v_merged_count,
    'failed_count', v_failed_count,
    'merged_pairs', v_merged_pairs,
    'errors', v_errors
  );

exception
  when others then
    return jsonb_build_object(
      'success', false,
      'error', SQLERRM,
      'merged_count', v_merged_count,
      'failed_count', v_failed_count
    );
end;
$$;

comment on function public.consolidate_memories is 'Consolidate multiple similar memories (like brain organizing thoughts during sleep)';


-- ==========================================================
-- SCHEDULED JOB: Day Dreaming - consolidate memories for all users and all dimensions
-- Like the brain reorganizing memories during sleep, this automatically consolidates similar memories
-- Designed to be called by pg_cron or external scheduler
-- ==========================================================
create or replace function public.day_dreaming(
  dream_depth float8 default 0.92,
  memories_per_cycle int default 20
)
returns jsonb
language plpgsql
security definer
as $$
declare
  v_dimension_record record;
  v_user_id uuid;
  v_user_result jsonb;
  v_total_merged int := 0;
  v_total_failed int := 0;
  v_users_processed int := 0;
  v_dimensions_processed int := 0;
  v_dimension_results jsonb := '[]'::jsonb;
  v_user_results jsonb := '[]'::jsonb;
  v_current_dimension int;
begin
  -- Get all unique dimensions that have embedding tables
  for v_dimension_record in (
    select distinct model_dimension
    from public.facts
    where public.embedding_table_exists(model_dimension)
    order by model_dimension
  ) loop
    v_current_dimension := v_dimension_record.model_dimension;
    v_dimensions_processed := v_dimensions_processed + 1;

    declare
      v_dimension_merged int := 0;
      v_dimension_failed int := 0;
      v_dimension_users int := 0;
    begin
      -- Process each user with facts for this dimension
      for v_user_id in (
        select distinct user_id
        from public.facts
        where model_dimension = v_current_dimension
        limit 1000  -- Safety limit per dimension
      ) loop
        -- Run batch merge for this user and dimension
        v_user_result := public.consolidate_memories(
          v_user_id,
          v_current_dimension,
          dream_depth,
          memories_per_cycle,
          'combine'  -- Use combine strategy for automatic merges
        );

        declare
          v_merged int := (v_user_result->>'merged_count')::int;
          v_failed int := (v_user_result->>'failed_count')::int;
        begin
          v_dimension_merged := v_dimension_merged + v_merged;
          v_dimension_failed := v_dimension_failed + v_failed;
          v_dimension_users := v_dimension_users + 1;
          v_total_merged := v_total_merged + v_merged;
          v_total_failed := v_total_failed + v_failed;

          -- Store result if any merges happened
          if v_merged > 0 then
            v_user_results := v_user_results || jsonb_build_object(
              'user_id', v_user_id,
              'dimension', v_current_dimension,
              'merged', v_merged,
              'failed', v_failed
            );
          end if;
        end;
      end loop;

      -- Store dimension summary
      v_dimension_results := v_dimension_results || jsonb_build_object(
        'dimension', v_current_dimension,
        'users_processed', v_dimension_users,
        'total_merged', v_dimension_merged,
        'total_failed', v_dimension_failed
      );
    end;
  end loop;

  v_users_processed := (
    select count(distinct user_id)
    from public.facts
    where public.embedding_table_exists(model_dimension)
  );

  return jsonb_build_object(
    'success', true,
    'timestamp', now(),
    'dimensions_processed', v_dimensions_processed,
    'users_processed', v_users_processed,
    'total_merged', v_total_merged,
    'total_failed', v_total_failed,
    'dimension_results', v_dimension_results,
    'user_results', v_user_results
  );

exception
  when others then
    return jsonb_build_object(
      'success', false,
      'error', SQLERRM,
      'timestamp', now(),
      'dimensions_processed', v_dimensions_processed,
      'total_merged', v_total_merged,
      'total_failed', v_total_failed
    );
end;
$$;

comment on function public.day_dreaming is 'Day Dreaming - like the brain during sleep, automatically consolidates similar memories for all users across all embedding dimensions';


-- ==========================================================
-- STATISTICS: Get deduplication stats
-- ==========================================================
create or replace function public.get_deduplication_stats(
  p_user_id uuid,
  p_dimension int
)
returns jsonb
language plpgsql
stable
security definer
as $$
declare
  v_total_facts int;
  v_potential_duplicates int;
  v_merged_facts int;
  v_result jsonb;
begin
  -- Count total facts
  select count(*) into v_total_facts
  from public.facts
  where user_id = p_user_id and model_dimension = p_dimension;

  -- Count potential duplicates (similarity >= 0.90)
  select count(*) into v_potential_duplicates
  from public.find_similar_memories(p_user_id, p_dimension, 0.90, 1000);

  -- Count merged facts from history
  select count(distinct fact_id) into v_merged_facts
  from public.fact_history
  where user_id = p_user_id
    and change_reason like '%merged from%';

  v_result := jsonb_build_object(
    'user_id', p_user_id,
    'dimension', p_dimension,
    'total_facts', v_total_facts,
    'potential_duplicate_pairs', v_potential_duplicates,
    'facts_with_merge_history', v_merged_facts
  );

  return v_result;
end;
$$;

comment on function public.get_deduplication_stats is 'Get statistics about fact deduplication for a user';


-- ==========================================================
-- SETUP pg_cron JOB (requires pg_cron extension)
-- Uncomment and modify schedule as needed
-- ==========================================================

/*
-- Enable pg_cron extension (run as superuser)
create extension if not exists pg_cron;

-- Schedule day dreaming job to run daily at 2 AM UTC
-- Like the brain during sleep, automatically consolidates memories across ALL dimensions
select cron.schedule(
  'day-dreaming',                         -- Job name
  '0 2 * * *',                            -- Cron schedule (2 AM daily)
  $$
  select public.day_dreaming(
    dream_depth := 0.92,
    memories_per_cycle := 20
  );
  $$
);

-- View scheduled jobs
select * from cron.job;

-- View job run history
select * from cron.job_run_details order by start_time desc limit 10;

-- To unschedule the job:
-- select cron.unschedule('day-dreaming');
*/


-- ==========================================================
-- MANUAL TRIGGER EXAMPLE
-- Use this to manually trigger deduplication for testing
-- ==========================================================

/*
-- Example 1: Find similar memories for a user (specific dimension)
select * from public.find_similar_memories(
  'YOUR_USER_ID'::uuid,
  1536,
  0.90,
  50
);

-- Example 2: Merge specific facts
select public.merge_facts(
  'FACT_ID_TO_KEEP'::uuid,
  'FACT_ID_TO_MERGE'::uuid,
  'YOUR_USER_ID'::uuid,
  'combine',
  'manual_merge'
);

-- Example 3: Consolidate memories for one user and dimension
select public.consolidate_memories(
  'YOUR_USER_ID'::uuid,
  1536,
  0.92,
  20,
  'combine'
);

-- Example 4: Get deduplication stats for a dimension
select public.get_deduplication_stats(
  'YOUR_USER_ID'::uuid,
  1536
);

-- Example 5: Day Dreaming - consolidate memories for ALL users and ALL dimensions (manual trigger)
-- This is the same function called by the cron job
select public.day_dreaming(0.92, 20);

-- Example 6: Check what dimensions are in use
select distinct model_dimension, count(*) as fact_count
from public.facts
group by model_dimension
order by model_dimension;
*/
