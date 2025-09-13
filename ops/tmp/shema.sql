-- Расширения
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
-- если образ без pgvector, то:
-- CREATE EXTENSION IF NOT EXISTS vector;

-- ==== Справочники оргструктуры ====
CREATE TABLE org_units (
  org_unit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  parent_id   UUID REFERENCES org_units(org_unit_id) ON DELETE SET NULL,
  name        TEXT NOT NULL
);

CREATE TABLE job_titles (
  job_title_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name         TEXT NOT NULL UNIQUE
);

-- ==== Пользователи ====
CREATE TYPE user_role AS ENUM ('employee','methodist','admin');

CREATE TABLE users (
  user_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  full_name    TEXT NOT NULL,
  org_unit_id  UUID REFERENCES org_units(org_unit_id),
  job_title_id UUID REFERENCES job_titles(job_title_id),
  role         user_role NOT NULL DEFAULT 'employee',
  email        TEXT UNIQUE,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ==== Навыки и критерии ====
CREATE TABLE skills (
  skill_id   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name       TEXT NOT NULL UNIQUE
);

CREATE TABLE criteria (
  criterion_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  skill_id     UUID NOT NULL REFERENCES skills(skill_id) ON DELETE CASCADE,
  name         TEXT NOT NULL,
  default_recommendation TEXT,        -- базовая рекомендация методиста
  UNIQUE (skill_id, name)
);

-- Шкала 0–5 для каждого критерия (нормализовано)
CREATE TABLE criterion_levels (
  criterion_id UUID NOT NULL REFERENCES criteria(criterion_id) ON DELETE CASCADE,
  level        INT  NOT NULL CHECK (level BETWEEN 0 AND 5),
  description  TEXT NOT NULL,
  PRIMARY KEY (criterion_id, level)
);

-- ==== Кейсы ====
CREATE TABLE cases (
  case_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  title          TEXT,                         -- если понадобится
  case_text      TEXT NOT NULL,                -- задание
  best_solution  TEXT NOT NULL,                -- эталон
  created_by     UUID REFERENCES users(user_id),
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Ключевые слова (по одному слову, чтобы можно было искать)
CREATE TABLE case_keywords (
  case_id  UUID REFERENCES cases(case_id) ON DELETE CASCADE,
  keyword  TEXT NOT NULL,
  PRIMARY KEY (case_id, keyword)
);

-- Какие критерии применяются к кейсу (M:N) + опциональные оверрайды
CREATE TABLE case_criteria (
  case_id       UUID NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
  criterion_id  UUID NOT NULL REFERENCES criteria(criterion_id) ON DELETE CASCADE,
  custom_recommendation TEXT,    -- если для кейса рекомендация своя
  PRIMARY KEY (case_id, criterion_id)
);

-- ==== Назначения и сессии ====
CREATE TYPE assignment_status AS ENUM ('assigned','in_progress','submitted','graded','returned');

CREATE TABLE assignments (
  assignment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  case_id       UUID NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
  user_id       UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  assigned_by   UUID REFERENCES users(user_id),
  due_at        TIMESTAMPTZ,
  status        assignment_status NOT NULL DEFAULT 'assigned',
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE sessions (
  session_id   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id      UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  assignment_id UUID REFERENCES assignments(assignment_id) ON DELETE SET NULL,
  started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at  TIMESTAMPTZ,
  duration_sec INTEGER GENERATED ALWAYS AS
    (CASE WHEN finished_at IS NOT NULL THEN EXTRACT(EPOCH FROM (finished_at - started_at))::INT END) STORED
);

-- ==== Подачи решения (попытки) ====
CREATE TYPE submission_status AS ENUM ('submitted','evaluated','returned');

CREATE TABLE submissions (
  submission_id  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  assignment_id  UUID NOT NULL REFERENCES assignments(assignment_id) ON DELETE CASCADE,
  session_id     UUID REFERENCES sessions(session_id),
  case_id        UUID NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
  user_id        UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
  solution_text  TEXT NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  status         submission_status NOT NULL DEFAULT 'submitted'
);

-- ==== Оценивание ====
CREATE TABLE evaluations (
  evaluation_id   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  submission_id   UUID NOT NULL UNIQUE REFERENCES submissions(submission_id) ON DELETE CASCADE,
  model_name      TEXT NOT NULL,               -- LLM (gpt/deepseek/…)
  model_version   TEXT,
  prompt_template_id TEXT,                     -- id шаблона промпта/режима
  overall_comment TEXT,                        -- общий вердикт
  overall_score   NUMERIC(4,2),                -- если нужна агрегированная оценка
  review_mode     TEXT,                        -- например, 'soft'/'hard'
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Точечные оценки по критериям
CREATE TABLE evaluation_items (
  evaluation_id  UUID NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
  criterion_id   UUID NOT NULL REFERENCES criteria(criterion_id),
  score          INT  NOT NULL CHECK (score BETWEEN 0 AND 5),
  explanation    TEXT NOT NULL,                -- объяснение оценке
  recommendation TEXT NOT NULL,                -- сгенерированная рекомендация
  PRIMARY KEY (evaluation_id, criterion_id)
);

-- ==== Трейсы и аудит ====
CREATE TABLE traces (
  trace_id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  submission_id      UUID NOT NULL REFERENCES submissions(submission_id) ON DELETE CASCADE,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  model_name         TEXT,
  prompt_template_id TEXT,
  review_mode        TEXT,
  weights_json       JSONB,                    -- {alpha,beta,gamma}
  retrieved_ids      JSONB,                    -- [case_id, ...] или фрагменты
  similarities_json  JSONB,                    -- [{id, sim}, ...]
  rules_results_json JSONB,                    -- результаты линтеров/регексов
  llm_summary_json   JSONB,                    -- краткий итог
  final_status       TEXT,                     -- ok / needs_fix / …
  confidence         NUMERIC(4,3),
  latency_ms         INTEGER,
  token_usage        JSONB                     -- {prompt, completion, total}
);

CREATE TABLE activity_log (
  log_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  submission_id UUID REFERENCES submissions(submission_id) ON DELETE CASCADE,
  actor        TEXT NOT NULL,                  -- user:<id> / system:llm / methodist:<id>
  action       TEXT NOT NULL,
  ts           TIMESTAMPTZ NOT NULL DEFAULT now(),
  diff         JSONB                            -- опционально патч состояния
);

-- ==== Эмбеддинги (pgvector) ====
-- Выбери размер под модель (пример: 768). При другой модели поменяй размер.
CREATE TABLE case_embeddings (
  case_id     UUID PRIMARY KEY REFERENCES cases(case_id) ON DELETE CASCADE,
  kind        TEXT NOT NULL CHECK (kind IN ('case_text','best_solution')),
  embedding   vector(768) NOT NULL,
  model_name  TEXT NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON case_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE submission_embeddings (
  submission_id UUID PRIMARY KEY REFERENCES submissions(submission_id) ON DELETE CASCADE,
  embedding     vector(768) NOT NULL,
  model_name    TEXT NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON submission_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ==== Поисковые индексы (опц., для быстрых фильтров) ====
-- Полнотекстовый поиск по кейсам
ALTER TABLE cases ADD COLUMN tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('russian', coalesce(case_text,'') || ' ' || coalesce(best_solution,''))) STORED;
CREATE INDEX cases_tsv_idx ON cases USING GIN (tsv);
