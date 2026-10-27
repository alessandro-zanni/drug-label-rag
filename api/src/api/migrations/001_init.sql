create extension if not exists vector;

create table labels (
  set_id text primary key,
  label_id text not null,
  generic_name text not null,
  brand_name text,
  manufacturer text,
  product_type text,
  effective_date date not null,
  fetched_at timestamptz not null default now()
);

-- content carries a "Drug (Brand) — Section" header so both the embedding and the
-- full-text index know which drug and section a chunk belongs to.
create table chunks (
  id bigint generated always as identity primary key,
  set_id text not null references labels on delete cascade,
  section text not null,
  ordinal int not null,
  content text not null,
  embedding vector(1024),
  tsv tsvector generated always as (to_tsvector('english', content)) stored,
  unique (set_id, section, ordinal)
);

create index on chunks using hnsw (embedding vector_cosine_ops);
create index on chunks using gin (tsv);
