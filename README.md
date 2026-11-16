# ДЗ №1. Анализ и ускорение Qwen3-4B

В этой работе я сравниваю исходную Qwen3-4B (BF16) и её 4-битную версию на MacBook с M3 Pro и 18 ГиБ памяти.

**[Читать отчёт](reports/report.md)**

Квантование ускорило генерацию в 3,4–3,5 раза и снизило пиковую память MLX на 63–70%. Это результаты двух запросов, по три измерения после прогрева; подробная методика и ограничения приведены в отчёте.

| Содержимое | Файлы |
|---|---|
| Отчёт, архитектура, расчёты и выводы | [reports/report.md](reports/report.md) |
| Скрипт измерений | [benchmarks/run.py](benchmarks/run.py) |
| Исходные замеры и ответы | [BF16](results/baseline.jsonl), [Q4](results/q4.jsonl) |
| Коэффициенты ускорения и экономии памяти | [results/summary.json](results/summary.json) |
| Версии библиотек | [requirements.txt](requirements.txt) |
| Материалы курса | [references/README.md](references/README.md) |

## Воспроизведение

Использовались Python 3.11, macOS 26.6.2 и MLX-LM. Нужен Mac с Apple Silicon и достаточно свободной памяти и диска для исходных весов и Q4-копии. Команды выполняются из корня репозитория.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

HF_HOME="$PWD/.cache/huggingface" .venv/bin/python benchmarks/run.py \
  --model Qwen/Qwen3-4B --out results/baseline.jsonl

HF_HOME="$PWD/.cache/huggingface" .venv/bin/python -m mlx_lm convert \
  --hf-path Qwen/Qwen3-4B --mlx-path models/qwen3-4b-q4 -q --q-bits 4

.venv/bin/python benchmarks/run.py \
  --model models/qwen3-4b-q4 --out results/q4.jsonl
```

Первый запуск скачивает модель. Конвертация также может обращаться к Hugging Face за карточкой модели. Повторный запуск измерений перезаписывает указанный выходной JSONL; для отдельного эксперимента следует задать другое имя файла.

В исходных результатах Q4 путь начинается с `work/`: так называлась рабочая папка при проведении эксперимента. На результаты это не влияет.

Веса, кеш загрузок и виртуальное окружение исключены из Git.
