-- staging под CSV как есть
CREATE TABLE IF NOT EXISTS public._cases_csv (
  case_id       TEXT,
  case_text     TEXT,
  best_solution TEXT,
  keywords      TEXT,
  skills        TEXT
);

TRUNCATE public._cases_csv;

COPY public._cases_csv (case_id, case_text, best_solution, keywords, skills)
FROM '/docker-entrypoint-initdb.d/data/cases.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '"');

-- переносим в нормализованную таблицу cases (external_id <- case_id из CSV)
INSERT INTO public.cases (external_id, case_text, best_solution)
SELECT c.case_id, c.case_text, c.best_solution
FROM public._cases_csv c
ON CONFLICT (external_id) DO UPDATE
SET case_text = EXCLUDED.case_text,
    best_solution = EXCLUDED.best_solution,
    updated_at = now();

-- keywords → public.case_keywords (очистим для переливаемых записей)
DELETE FROM public.case_keywords ck
USING public.cases cs
WHERE ck.case_id = cs.case_id
  AND cs.external_id IN (SELECT case_id FROM public._cases_csv);

WITH src AS (
  SELECT cs.case_id,
         regexp_split_to_table(trim(both '[]' from c.keywords), '\s*,\s*') AS raw_kw
  FROM public._cases_csv c
  JOIN public.cases cs ON cs.external_id = c.case_id
),
clean AS (
  SELECT case_id, trim(both '''"' from raw_kw) AS keyword
  FROM src
)
INSERT INTO public.case_keywords (case_id, keyword)
SELECT case_id, keyword
FROM clean
WHERE keyword IS NOT NULL AND keyword <> ''
ON CONFLICT DO NOTHING;
