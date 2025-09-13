import re
import time
import json
import random
import logging
from typing import List, Dict, Any

import torch
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from huggingface_hub import login


# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MistralSynthesizer:
    """Синтезатор кейсов с использованием Mistral 7B через HuggingFace"""
    
    def __init__(self, hf_token=None):
        logger.info("Инициализация Mistral 7B через HuggingFace...")
        
        if hf_token:
            login(token=hf_token)
        
        try:
            self.openai_client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def read_csv(self, file_path: str) -> pd.DataFrame:
        """Чтение CSV файла"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Прочитано {len(df)} кейсов из {file_path}")
            return df
        except Exception as e:
            logger.error(f"Ошибка чтения файла {file_path}: {e}")
            return pd.DataFrame()
    
    def save_csv(self, df: pd.DataFrame, file_path: str):
        """Сохранение данных в CSV файл"""
        try:
            df.to_csv(file_path, index=False, encoding='utf-8')
            logger.info(f"Сохранено {len(df)} кейсов в {file_path}")
        except Exception as e:
            logger.error(f"Ошибка записи в файл {file_path}: {e}")
    
    def generate_with_mistral(self, prompt: str, max_length: int = 2048) -> str:
        """Генерация текста с помощью Mistral 7B"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
            response = self.openai_client.chat.completions.create(
                model='qwen2.5vl:72b',
                n=1,
                temperature=0.8,
                messages=messages
            )
            
            generated_text = response.choices[0].message.content

            # Убираем промпт из результата
            if generated_text.startswith(prompt):
                generated_text = generated_text.replace(prompt, '')

            return generated_text
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return ""
    
    def create_case_prompt(self, example_case: Dict[str, Any]) -> str:
        """Создание промпта для генерации кейса"""
        prompt = f"""<s>[INST]Ты эксперт по созданию обучающих кейсов для корпоративного обучения в X5 Group. 

            На основе приведенного примера создай новый уникальный кейс в ТОЧНОМ ТОМ ЖЕ ФОРМАТЕ.

            Пример кейса:
            case_id: {example_case['case_id']}
            case_text: {example_case['case_text']}
            best_solution: {example_case['best_solution']}
            keywords: {example_case['keywords']}
            skills: {example_case['skills']}

            Создай новый кейс со следующими требованиями:
            1. case_id в формате X5-[НОВАЯ_РОЛЬ]-[НОВАЯ_ТЕМА]-[НОМЕР]
            2. case_text должен описывать реалистичную рабочую ситуацию в X5 Group
            3. best_solution должен содержать конкретные, практические рекомендации
            4. keywords - список релевантных ключевых слов в JSON формате
            5. skills - список словарей с навыками в JSON формате
            6. Кейс должен быть уникальным и отличаться от примера
            7. Сохраняй профессиональный тон и уровень детализации

            Верни ответ ТОЛЬКО в формате JSON без каких-либо пояснений:
            {{
            "case_id": "X5-НОВАЯ_РОЛЬ-НОВАЯ_ТЕМА-001",
            "case_text": "текст кейса",
            "best_solution": "решение",
            "keywords": ["ключевые", "слова"],
            "skills": [{{"skill": "навык", "criterion": "критерий", "rubric": ["шкала"], "recommendation": "рекомендация"}}]
            }}
            [/INST]</s>
        """

        return prompt
    
    def parse_generated_case(self, generated_text: str) -> Dict[str, Any]:
        """Парсинг сгенерированного кейса"""
        try:
            # Ищем JSON в тексте
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                case_data = json.loads(json_str)
                
                # Валидация обязательных полей
                required_fields = ['case_id', 'case_text', 'best_solution', 'keywords', 'skills']
                if all(field in case_data for field in required_fields):
                    return case_data
            
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Ошибка парсинга: {e}")
            return None
    
    def generate_case_with_llm(self, example_case: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация одного кейса с помощью Mistral"""
        prompt = self.create_case_prompt(example_case)
        
        for attempt in range(3):  # 3 попытки генерации
            try:
                generated_text = self.generate_with_mistral(prompt)
                
                if generated_text:
                    case_data = self.parse_generated_case(generated_text)
                    if case_data and self.validate_case(case_data):
                        return case_data
                
                logger.warning(f"Попытка {attempt + 1}: не удалось сгенерировать валидный кейс")
                time.sleep(1)  # Пауза между попытками
                
            except Exception as e:
                logger.error(f"Ошибка генерации на попытке {attempt + 1}: {e}")
                time.sleep(2)
        
        return None
    
    def validate_case(self, case_data: Dict[str, Any]) -> bool:
        """Валидация сгенерированного кейса"""
        try:
            # Проверка обязательных полей
            required_fields = ['case_id', 'case_text', 'best_solution', 'keywords', 'skills']
            if not all(field in case_data for field in required_fields):
                return False
            
            # Проверка формата case_id
            if not case_data['case_id'].startswith('X5-'):
                return False
            
            # Проверка длины текстов
            if len(case_data['case_text']) < 50 or len(case_data['best_solution']) < 50:
                return False
            
            # Проверка типов данных
            if not isinstance(case_data['keywords'], list):
                return False
            if not isinstance(case_data['skills'], list):
                return False
            
            return True
            
        except Exception:
            return False
    
    def synthesize_with_mistral(self, input_file: str, output_file: str, num_samples: int = 10) -> pd.DataFrame:
        """Синтез кейсов с использованием Mistral 7B"""
        
        # Чтение исходных данных
        original_df = self.read_csv(input_file)
        if original_df.empty:
            raise ValueError("Исходный файл пуст или не найден")
        
        # Берем пример для генерации
        example_row = original_df.iloc[0]
        example_case = {
            'case_id': example_row['case_id'],
            'case_text': example_row['case_text'],
            'best_solution': example_row['best_solution'],
            'keywords': example_row['keywords'],
            'skills': example_row['skills']
        }
        
        # Генерация новых кейсов
        synthetic_data = []
        
        for i in tqdm(range(num_samples), desc="Генерация кейсов с Mistral 7B"):
            try:
                case_data = self.generate_case_with_llm(example_case)
                
                if case_data:
                    # Конвертируем списки в строки для CSV
                    case_data['keywords'] = json.dumps(case_data['keywords'], ensure_ascii=False)
                    case_data['skills'] = json.dumps(case_data['skills'], ensure_ascii=False)
                    synthetic_data.append(case_data)
                    logger.info(f"Сгенерирован кейс: {case_data['case_id']}")
                else:
                    logger.warning(f"Не удалось сгенерировать кейс {i+1}")
                    
                # Пауза между запросами
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Ошибка при генерации кейса {i+1}: {e}")
                continue
        
        # Создание DataFrame из новых данных
        synthetic_df = pd.DataFrame(synthetic_data)
        
        # Объединение с исходными данными
        combined_df = pd.concat([original_df, synthetic_df], ignore_index=True)
        
        # Сохранение
        self.save_csv(combined_df, output_file)
        
        return combined_df

# Альтернатива: шаблонный генератор на случай проблем с моделью
class TemplateSynthesizer:
    """Резервный синтезатор на основе шаблонов"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Any]:
        """Загрузка шаблонов для генерации"""
        return {
            'roles': ['BARISTA', 'MANAGER', 'IT', 'LOGISTICS', 'HR', 'FINANCE', 'MARKETING'],
            'themes': ['UPSELL', 'QUALITY', 'CUSTOMER', 'TRAINING', 'OPTIMIZATION', 'ANALYTICS']
        }
    
    def generate_template_case(self, index: int) -> Dict[str, Any]:
        """Генерация кейса по шаблону"""
        role = random.choice(self.templates['roles'])
        theme = random.choice(self.templates['themes'])
        
        return {
            'case_id': f"X5-{role}-{theme}-{index:03d}",
            'case_text': f"Вы — {role.lower()} в X5 Group. Реалистичная рабочая ситуация требующая решения.",
            'best_solution': "Конкретный план действий с учетом лучших практик и доступных ресурсов.",
            'keywords': json.dumps([role.lower(), theme.lower(), "x5", "кейс"], ensure_ascii=False),
            'skills': json.dumps([{
                "skill": "Профессиональные знания",
                "criterion": "Глубина понимания предметной области",
                "rubric": ["0 — отсутствует", "5 — экспертное понимание"],
                "recommendation": "Изучите дополнительные материалы по теме"
            }], ensure_ascii=False)
        }

def main():
    """Основная функция"""
    
    # ==================== НАСТРОЙКИ ====================
    INPUT_CSV_PATH = "/Users/nikolya/Downloads/ai_hack_ivan/cases_fixed_preview.csv"    # Ваш исходный CSV файл
    OUTPUT_CSV_PATH = "mistral_generated_cases.csv" # Результат
    NUM_SAMPLES_TO_GENERATE = 38                  # Количество новых кейсов
    HF_TOKEN = ""      # Ваш HF token (опционально)
    USE_MISTRAL = True                            # Использовать Mistral или шаблоны
    # ===================================================
    
    try:
        if USE_MISTRAL:
            # Использование Mistral 7B
            synthesizer = MistralSynthesizer(hf_token=HF_TOKEN if HF_TOKEN != "your_huggingface_token_here" else None)
            result_df = synthesizer.synthesize_with_mistral(
                input_file=INPUT_CSV_PATH,
                output_file=OUTPUT_CSV_PATH,
                num_samples=NUM_SAMPLES_TO_GENERATE
            )
        else:
            # Резервный шаблонный метод
            synthesizer = TemplateSynthesizer()
            synthetic_data = [synthesizer.generate_template_case(i + 100) for i in range(NUM_SAMPLES_TO_GENERATE)]
            original_df = pd.read_csv(INPUT_CSV_PATH)
            synthetic_df = pd.DataFrame(synthetic_data)
            result_df = pd.concat([original_df, synthetic_df], ignore_index=True)
            result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        
        # Статистика
        print(f"\n✅ Генерация завершена!")
        print(f"📊 Всего кейсов: {len(result_df)}")
        print(f"🎯 Новых сгенерировано: {NUM_SAMPLES_TO_GENERATE}")
        print(f"💾 Сохранено в: {OUTPUT_CSV_PATH}")
        
        if USE_MISTRAL:
            print(f"🤖 Использована модель: Mistral 7B v0.3")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("🔄 Переключаюсь на шаблонный метод...")
        
        # Резервный метод
        synthesizer = TemplateSynthesizer()
        synthetic_data = [synthesizer.generate_template_case(i + 100) for i in range(NUM_SAMPLES_TO_GENERATE)]
        original_df = pd.read_csv(INPUT_CSV_PATH)
        synthetic_df = pd.DataFrame(synthetic_data)
        result_df = pd.concat([original_df, synthetic_df], ignore_index=True)
        result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
        
        print(f"✅ Сгенерировано {NUM_SAMPLES_TO_GENERATE} кейсов шаблонным методом")

if __name__ == "__main__":
    main()