import csv
import os

def txt_to_csv(txt_filepath, csv_filepath, delimiter=',', quotechar='"'):
    """
    Конвертирует текстовый файл (txt) в CSV файл.

    Args:
        txt_filepath (str): Путь к входному txt файлу.
        csv_filepath (str): Путь к выходному csv файлу.
        delimiter (str): Разделитель между полями в txt файле (по умолчанию ',').
        quotechar (str): Символ заключения для полей, содержащих разделитель (по умолчанию '"').
    """
    try:
        with open(txt_filepath, 'r', encoding='utf-8') as infile, \
             open(csv_filepath, 'w', newline='', encoding='utf-8') as outfile:


            writer = csv.writer(outfile, delimiter=delimiter, quotechar=quotechar, quoting=csv.QUOTE_MINIMAL)


            for line in infile:

                line = line.strip()

    
                fields = line.split(delimiter)

                # Записываем строку в CSV
                writer.writerow(fields)

        print(f"Файл '{txt_filepath}' успешно конвертирован в '{csv_filepath}'")

    except FileNotFoundError:
        print(f"Ошибка: Файл '{txt_filepath}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


# Пример использования
if __name__ == "__main__":
    txt_file = "/Users/ivangolovkin/VSCode/MyProjects/LLM_applications/deepseek_csv_20250908_73e8d7.txt"  # Замените на путь к вашему txt файлу
    csv_file = "/Users/ivangolovkin/VSCode/MyProjects/LLM_applications/output3.csv"  # Замените на желаемый путь к csv файлу
    my_delimiter = ";" 
    txt_to_csv(txt_file, csv_file, delimiter=my_delimiter)


    if os.path.exists(csv_file):
        print(f"CSV файл '{csv_file}' создан.")
    else:
        print(f"Ошибка: CSV файл '{csv_file}' не создан.")