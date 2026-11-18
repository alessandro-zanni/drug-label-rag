-- One row per answered question. Questions can be health data: rows older than 30 days are
-- purged on every insert (main.audit). client_net is the client IP truncated to /24 or /48.
create table audit_log (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  request_id text not null,
  endpoint text not null,
  question text not null,
  sources jsonb not null,
  refused boolean not null,
  latency_ms int not null,
  client_net cidr
);

create index on audit_log (created_at);
