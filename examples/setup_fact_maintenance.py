"""
Example: Setting up automatic fact deduplication and maintenance.

This example shows how to:
1. Initialize Supabase with fact maintenance enabled
2. Manually trigger deduplication
3. View deduplication statistics
4. Set up automatic pg_cron jobs (server-side)

🧠 Day Dreaming: Like the brain reorganizing memories during sleep, this system
   automatically consolidates similar facts to maintain a clean memory structure.

Requirements:
- Supabase project with facts enabled
- Direct PostgreSQL connection string
- pg_cron extension (for automatic scheduling)
"""

from langmiddle.storage import ChatStorage

# =============================================================================
# Option 1: Initialize with maintenance enabled (creates functions on startup)
# =============================================================================

storage = ChatStorage.create(
    backend_type="supabase",
    supabase_url="YOUR_SUPABASE_URL",
    supabase_key="YOUR_SUPABASE_ANON_KEY",
    connection_string="YOUR_DIRECT_CONNECTION_STRING",
    auto_create_tables=True,
    enable_facts=True,
    enable_day_dreaming=True,  # <-- Enable maintenance functions
)

print("✓ Storage initialized with fact maintenance functions")


# =============================================================================
# Option 2: Manual SQL execution (if not using auto_create_tables)
# =============================================================================

"""
If you don't use auto_create_tables, run these SQL files in Supabase SQL Editor:
1. chat_history.sql
2. chat_facts.sql
3. fact_maintenance.sql  <-- The new maintenance file
"""


# =============================================================================
# Manual Deduplication Examples
# =============================================================================

# Example 1: Find similar memories for a user
find_similar_memories_sql = """
select * from public.find_similar_memories(
  'YOUR_USER_ID'::uuid,
  1536,              -- embedding dimension
  0.90,              -- dream_depth: similarity threshold (0-1)
  50                 -- max results
);
"""

# Example 2: Merge two specific facts
merge_facts_sql = """
select public.merge_facts(
  'FACT_ID_TO_KEEP'::uuid,
  'FACT_ID_TO_MERGE'::uuid,
  'YOUR_USER_ID'::uuid,
  'combine',         -- merge strategy: 'keep_older', 'keep_newer', 'combine'
  'manual_merge'     -- reason for merge
);
"""

# Example 3: Consolidate memories for one user
consolidate_memories_sql = """
select public.consolidate_memories(
  'YOUR_USER_ID'::uuid,
  1536,              -- embedding dimension
  0.92,              -- dream_depth: similarity threshold (higher = more conservative)
  20,                -- memories_per_cycle: max merges per run
  'combine'          -- merge strategy
);
"""

# Example 4: Get deduplication statistics
# Example 4: Get deduplication statistics for a specific dimension
stats_sql = """
select public.get_deduplication_stats(
  'YOUR_USER_ID'::uuid,
  1536              -- specific dimension
);
"""

# Example 5: 🧠 Day Dreaming - consolidate memories for all users and ALL dimensions (manual trigger)
day_dreaming_sql = """
select public.day_dreaming(
  0.92,             -- dream_depth: similarity threshold
  20                -- memories_per_cycle: max merges per user per dimension
);
"""

# Example 6: Check what dimensions are in use
check_dimensions_sql = """
select distinct model_dimension, count(*) as fact_count
from public.facts
group by model_dimension
order by model_dimension;
"""


# =============================================================================
# Automatic Scheduling with pg_cron (Server-Side)
# =============================================================================

"""
🧠 Day Dreaming: Like the brain during REM sleep, automatically consolidate memories

To enable automatic memory consolidation, run this SQL in Supabase SQL Editor
(requires superuser access or enabled pg_cron extension):

-- 1. Enable pg_cron extension (if not already enabled)
create extension if not exists pg_cron;

-- 2. Schedule daily memory consolidation at 2 AM UTC
-- Like the brain during sleep, this automatically processes ALL dimensions
select cron.schedule(
  'day-dreaming',
  '0 2 * * *',  -- Cron schedule (2 AM daily)
  $$
  select public.day_dreaming(
    dream_depth := 0.92,
    memories_per_cycle := 20
  );
  $$
);

-- 3. View scheduled jobs
select * from cron.job;

-- 4. View job run history
select * from cron.job_run_details
order by start_time desc
limit 10;

-- 5. Check what dimensions are active
select distinct model_dimension, count(*) as fact_count
from public.facts
group by model_dimension
order by model_dimension;

-- 6. To unschedule the job:
-- select cron.unschedule('day-dreaming');
"""


# =============================================================================
# Merge Strategies Explained
# =============================================================================

"""
'keep_older':
  - Keeps the content/namespace from the older fact
  - Use when older facts are more reliable

'keep_newer':
  - Keeps the content/namespace from the newer fact
  - Use when newer facts contain corrections

'combine':
  - Takes longer content, more specific namespace
  - Averages intensity and confidence
  - Best for automatic merges (recommended)
"""


# =============================================================================
# Best Practices
# =============================================================================

"""
1. Dream Depth (Similarity Thresholds):
   - 0.95+: Very conservative, only exact duplicates
   - 0.92: Recommended for automatic consolidation (balanced)
   - 0.85-0.90: More aggressive, review manually first

2. Scheduling:
   - Run during low-traffic hours (e.g., 2 AM) - like actual sleep!
   - Start with daily, adjust based on usage
   - Monitor job run history for failures

3. Monitoring:
   - Check deduplication stats regularly
   - Review merge history in fact_history table
   - Watch for unexpected merge patterns

4. Safety:
   - Start with low memories_per_cycle (10-20)
   - Test on staging environment first
   - Keep fact_history for audit trail
   - Can always query history to see what was merged

5. Performance:
   - Limit memories_per_cycle to avoid long-running jobs
   - Use higher dream_depth for automatic consolidation
   - Consider batch size for large user bases
"""


# =============================================================================
# Query Fact History After Merges
# =============================================================================

view_merge_history_sql = """
-- View all facts that were merged
select
  fact_id,
  operation,
  content,
  change_reason,
  changed_fields->>'merged_from' as merged_from_id,
  changed_fields->>'merge_strategy' as strategy,
  changed_at
from public.fact_history
where change_reason like '%merge%'
order by changed_at desc
limit 20;

-- Get full history for a specific fact (including merges)
select * from public.get_fact_history(
  'FACT_ID'::uuid,
  'USER_ID'::uuid
);
"""


if __name__ == "__main__":
    print("🧠 Day Dreaming - Fact Memory Consolidation Setup")
    print("Like the brain during sleep, automatically organize and consolidate memories")
    print("\nKey functions available:")
    print("  - find_similar_memories(user_id, dimension, dream_depth, limit)")
    print("  - merge_facts(keep_id, merge_id, user_id, strategy, reason)")
    print("  - consolidate_memories(user_id, dimension, dream_depth, memories_per_cycle, strategy)")
    print("  - day_dreaming(dream_depth, memories_per_cycle)  # 🧠 Main function")
    print("  - get_deduplication_stats(user_id, dimension)")
    print("\nMerge strategies: 'keep_older', 'keep_newer', 'combine'")
    print("Recommended dream_depth for auto-consolidation: 0.92")
    print("\n💡 Run 'day_dreaming' daily at 2 AM for automatic memory consolidation")
