# ФИНАЛЬНАЯ СВОДКА ОБНОВЛЕННОГО АНАЛИЗА

## Обновленные Данные
- **Curated Dataset (ST2_v2.csv):** 195 статей (было 192) ✅
  - PubMed: 131 (было 128) ✅
  - Preprints: 64 (без изменений) ✅
- **Full Dataset (cleaned.csv):** 51,613 статей ✅

## Overlap Проценты
| Анализ | Было | Стало | Статус |
|--------|------|-------|--------|
| **Standard** | 57% (17/30) | 57% (17/30) | ✅ Без изменений |
| **Filtered** | 23% (7/30) | 23% (7/30) | ✅ Без изменений |
| **Fine-Tuned** | 27% (8/30) | **23% (7/30)** | ⚠️ Изменилось! |

**Важно:** Fine-tuned overlap теперь такой же как filtered (23%)

## Union Terms
| Анализ | Было | Стало |
|--------|------|-------|
| Standard | 43 | 43 ✅ |
| Filtered | 52/53 | 53 ✅ |
| Fine-Tuned | 52 | 53 ⚠️ |

## Топ-Термины (Filtered Analysis)
1. precision medicine (0.0172)
2. gene expression (0.0151)
3. pre trained (0.0151)
4. fine tuned (0.0142)
5. open source (0.0127)

## Топ-Термины (Fine-Tuned Analysis)
1. precision medicine (0.0215)
2. pre trained (0.0181)
3. fine tuned (0.0177)
4. open source (0.0168)
5. gene expression (0.0166)

## Generic Terms Removed
- Было: 206 terms
- Стало: **193 terms** ⚠️ (изменилось из-за новых данных)

## Фигуры
### ❌ Удалено из Main Text:
- Fig3_selected_filtered_comparison_top30.pdf (перемещена в supplement)

### ✅ Supplement Figures (5 фигур):
- SFig1_progression_top30.pdf (3 panels: Full → Selected → Filtered)
- SFig2_selected_comparison_top30.pdf (Standard, 57% overlap, 43 union)
- SFig3_selected_filtered_comparison_top30.pdf (Filtered, 23% overlap, 53 union)
- SFig4_topic_scatter.pdf (LDA, 8 topics)
- SFig5_finetuned_comparison_top30.pdf (Fine-tuned, 23% overlap, 53 union)

## Обновленные Файлы
✅ tfidf_analysis.py - убрана генерация Fig3
✅ UPDATED_METHODS_SECTION.tex - все цифры обновлены
✅ UPDATED_APPENDIX.tex - все цифры обновлены
✅ FIGURE_CAPTIONS.md - Fig3 удалена из main text
✅ results/ANALYSIS_SUMMARY.md - автоматически обновлен
✅ Все фигуры скопированы в ../imgs/

## Ключевые Изменения для Рукописи
1. **195 статей** в curated dataset (не 192)
2. **131 PubMed** articles (не 128)
3. **Fine-tuned overlap = 23%** (не 27%)
4. **193 generic terms** удалены (не 206)
5. **Все TF-IDF фигуры только в supplement** (нет в main text)

## Следующие Шаги
- [ ] Обновить основную рукопись с цифрами из UPDATED_METHODS_SECTION.tex
- [ ] Обновить supplement с текстом из UPDATED_APPENDIX.tex
- [ ] Убедиться что Fig3 не упоминается в main text
- [ ] Проверить что все 5 SFigs правильно пронумерованы в рукописи
