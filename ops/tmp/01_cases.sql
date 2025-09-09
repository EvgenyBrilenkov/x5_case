-- ==== Справочники оргструктуры ====
CREATE TABLE IF NOT EXISTS public.org_units (
  org_unit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  parent_id   UUID REFERENCES public.org_units(org_unit_id) ON DELETE SET NULL,
  name        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS public.job_titles (
  job_title_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name         TEXT NOT NULL UNIQUE
);

-- ==== Пользователи ====
DO $$ BEGIN
  CREATE TYPE user_role AS ENUM ('employee','methodist','admin');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS public.users (
  user_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  full_name    TEXT NOT NULL,
  org_unit_id  UUID REFERENCES public.org_units(org_unit_id),
  job_title_id UUID REFERENCES public.job_titles(job_title_id),
  role         user_role NOT NULL DEFAULT 'employee',
  email        TEXT UNIQUE,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ==== Навыки и критерии ====
CREATE TABLE IF NOT EXISTS public.skills (
  skill_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name       TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS public.criteria (
  criterion_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  skill_id     UUID NOT NULL REFERENCES public.skills(skill_id) ON DELETE CASCADE,
  name         TEXT NOT NULL,
  default_recommendation TEXT,
  UNIQUE (skill_id, name)
);

CREATE TABLE IF NOT EXISTS public.criterion_levels (
  criterion_id UUID NOT NULL REFERENCES public.criteria(criterion_id) ON DELETE CASCADE,
  level        INT  NOT NULL CHECK (level BETWEEN 0 AND 5),
  description  TEXT NOT NULL,
  PRIMARY KEY (criterion_id, level)
);

-- ==== Кейсы (нормализовано) ====
CREATE TABLE IF NOT EXISTS public.cases (
  case_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  external_id    TEXT UNIQUE,                 -- сюда кладём исходный case_id из CSV
  title          TEXT,
  case_text      TEXT NOT NULL,
  best_solution  TEXT NOT NULL,
  created_by     UUID REFERENCES public.users(user_id),
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ключевые слова по одному слову
CREATE TABLE IF NOT EXISTS public.case_keywords (
  case_id  UUID NOT NULL REFERENCES public.cases(case_id) ON DELETE CASCADE,
  keyword  TEXT NOT NULL,
  PRIMARY KEY (case_id, keyword)
);

-- критерии, применяемые к кейсу
CREATE TABLE IF NOT EXISTS public.case_criteria (
  case_id       UUID NOT NULL REFERENCES public.cases(case_id) ON DELETE CASCADE,
  criterion_id  UUID NOT NULL REFERENCES public.criteria(criterion_id) ON DELETE CASCADE,
  custom_recommendation TEXT,
  PRIMARY KEY (case_id, criterion_id)
);

-- ==== Назначения и сессии ====
DO $$ BEGIN
  CREATE TYPE assignment_status AS ENUM ('assigned','in_progress','submitted','graded','returned');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS public.assignments (
  assignment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  case_id       UUID NOT NULL REFERENCES public.cases(case_id) ON DELETE CASCADE,
  user_id       UUID NOT NULL REFERENCES public.users(user_id) ON DELETE CASCADE,
  assigned_by   UUID REFERENCES public.users(user_id),
  due_at        TIMESTAMPTZ,
  status        assignment_status NOT NULL DEFAULT 'assigned',
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.sessions (
  session_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID NOT NULL REFERENCES public.users(user_id) ON DELETE CASCADE,
  assignment_id UUID REFERENCES public.assignments(assignment_id) ON DELETE SET NULL,
  started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at   TIMESTAMPTZ,
  duration_sec  INTEGER GENERATED ALWAYS AS
    (CASE WHEN finished_at IS NOT NULL THEN EXTRACT(EPOCH FROM (finished_at - started_at))::INT END) STORED
);

-- ==== Подачи решения ====
DO $$ BEGIN
  CREATE TYPE submission_status AS ENUM ('submitted','evaluated','returned');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS public.submissions (
  submission_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  assignment_id  UUID NOT NULL REFERENCES public.assignments(assignment_id) ON DELETE CASCADE,
  session_id     UUID REFERENCES public.sessions(session_id),
  case_id        UUID NOT NULL REFERENCES public.cases(case_id) ON DELETE CASCADE,
  user_id        UUID NOT NULL REFERENCES public.users(user_id) ON DELETE CASCADE,
  solution_text  TEXT NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  status         submission_status NOT NULL DEFAULT 'submitted'
);

-- ==== Оценивание ====
CREATE TABLE IF NOT EXISTS public.evaluations (
  evaluation_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  submission_id     UUID NOT NULL UNIQUE REFERENCES public.submissions(submission_id) ON DELETE CASCADE,
  model_name        TEXT NOT NULL,
  model_version     TEXT,
  prompt_template_id TEXT,
  overall_comment   TEXT,
  overall_score     NUMERIC(4,2),
  review_mode       TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.evaluation_items (
  evaluation_id  UUID NOT NULL REFERENCES public.evaluations(evaluation_id) ON DELETE CASCADE,
  criterion_id   UUID NOT NULL REFERENCES public.criteria(criterion_id),
  score          INT  NOT NULL CHECK (score BETWEEN 0 AND 5),
  explanation    TEXT NOT NULL,
  recommendation TEXT NOT NULL,
  PRIMARY KEY (evaluation_id, criterion_id)
);

-- ==== Трейсы и аудит ====
CREATE TABLE IF NOT EXISTS public.traces (
  trace_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  submission_id      UUID NOT NULL REFERENCES public.submissions(submission_id) ON DELETE CASCADE,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  model_name         TEXT,
  prompt_template_id TEXT,
  review_mode        TEXT,
  weights_json       JSONB,
  retrieved_ids      JSONB,
  similarities_json  JSONB,
  rules_results_json JSONB,
  llm_summary_json   JSONB,
  final_status       TEXT,
  confidence         NUMERIC(4,3),
  latency_ms         INTEGER,
  token_usage        JSONB
);

CREATE TABLE IF NOT EXISTS public.activity_log (
  log_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  submission_id UUID REFERENCES public.submissions(submission_id) ON DELETE CASCADE,
  actor         TEXT NOT NULL,
  action        TEXT NOT NULL,
  ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
  diff          JSONB
);

-- ==== Эмбеддинги (pgvector) ====
-- Подставь нужную размерность (1024 пример)
CREATE TABLE IF NOT EXISTS public.case_embeddings (
  case_id     UUID PRIMARY KEY REFERENCES public.cases(case_id) ON DELETE CASCADE,
  kind        TEXT NOT NULL CHECK (kind IN ('case_text','best_solution')),
  embedding   vector(1024) NOT NULL,
  model_name  TEXT NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS case_embeddings_vec_idx
  ON public.case_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS public.submission_embeddings (
  submission_id UUID PRIMARY KEY REFERENCES public.submissions(submission_id) ON DELETE CASCADE,
  embedding     vector(1024) NOT NULL,
  model_name    TEXT NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS submission_embeddings_vec_idx
  ON public.submission_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ==== Полнотекстовый поиск по кейсам ====
ALTER TABLE public.cases
  ADD COLUMN IF NOT EXISTS tsv tsvector
  GENERATED ALWAYS AS (
    to_tsvector('russian', coalesce(case_text,'') || ' ' || coalesce(best_solution,''))
  ) STORED;

CREATE INDEX IF NOT EXISTS cases_tsv_idx ON public.cases USING GIN (tsv);
