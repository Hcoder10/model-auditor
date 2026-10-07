-- Apply once with the Supabase migration CLI or SQL editor.
-- Service-role/secret credentials stay on the server. No anonymous writes.
begin;

create table if not exists public.audit_runs (
  id uuid primary key,
  owner_id uuid references auth.users(id),
  created_at timestamptz not null default now(),
  status text not null,
  verdict text not null,
  mode text,
  report_sha256 text not null check (report_sha256 ~ '^[0-9a-f]{64}$'),
  chain_head_sha256 text not null check (chain_head_sha256 ~ '^[0-9a-f]{64}$'),
  report jsonb not null
);

create table if not exists public.audit_evidence (
  run_id uuid not null references public.audit_runs(id),
  seq integer not null check (seq >= 0),
  sha256 text not null check (sha256 ~ '^[0-9a-f]{64}$'),
  event jsonb not null,
  primary key (run_id, seq)
);

create table if not exists public.audit_artifacts (
  run_id uuid not null references public.audit_runs(id),
  path text not null,
  sha256 text not null check (sha256 ~ '^[0-9a-f]{64}$'),
  storage_key text not null,
  size_bytes bigint not null check (size_bytes >= 0),
  primary key (run_id, path)
);

create index if not exists audit_runs_owner_idx on public.audit_runs(owner_id, created_at desc);
alter table public.audit_runs enable row level security;
alter table public.audit_evidence enable row level security;
alter table public.audit_artifacts enable row level security;
revoke all on public.audit_runs, public.audit_evidence, public.audit_artifacts from anon, authenticated;
revoke all on public.audit_runs, public.audit_evidence, public.audit_artifacts from service_role;
grant select on public.audit_runs, public.audit_evidence, public.audit_artifacts to authenticated;
grant select, insert on public.audit_runs, public.audit_evidence, public.audit_artifacts to service_role;

drop policy if exists audit_runs_owner_read on public.audit_runs;
create policy audit_runs_owner_read on public.audit_runs for select to authenticated
  using (owner_id = (select auth.uid()));
drop policy if exists audit_evidence_owner_read on public.audit_evidence;
create policy audit_evidence_owner_read on public.audit_evidence for select to authenticated
  using (exists (select 1 from public.audit_runs r where r.id = run_id and r.owner_id = (select auth.uid())));
drop policy if exists audit_artifacts_owner_read on public.audit_artifacts;
create policy audit_artifacts_owner_read on public.audit_artifacts for select to authenticated
  using (exists (select 1 from public.audit_runs r where r.id = run_id and r.owner_id = (select auth.uid())));

insert into storage.buckets(id, name, public)
values ('audit-evidence', 'audit-evidence', false)
on conflict (id) do nothing;
-- No storage.objects policies are granted for this bucket; server-only access.
commit;
