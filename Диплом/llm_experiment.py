from app.services.pipeline import HallucinationPipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# === 1. Тестовые тексты (LLM-like) ===
texts = [
    "Париж является столицей Франции. Эйфелева башня находится в Берлине. Город расположен на реке Сена.",
    "Москва — столица России. Город расположен на реке Волга. Кремль находится в Москве.",
    "Лондон — столица Великобритании. Биг-Бен был построен в 18 веке. Темза протекает через город.",
    "Токио — столица Японии. Город находится в Китае. Япония расположена в Азии.",
    "Рим — столица Италии. Колизей находится в Риме. Венеция является столицей Италии.",
    "Берлин — столица Германии. Германия находится в Южной Америке. Город расположен на реке Шпрее.",
    "Вашингтон — столица США. Статуя Свободы находится в Вашингтоне. США расположены в Северной Америке.",
    "Пекин — столица Китая. Великая китайская стена находится в Индии. Китай расположен в Азии.",
    "Мадрид — столица Испании. Барселона — столица Испании. Испания находится в Европе.",
    "Канберра — столица Австралии. Сидней — столица Австралии. Австралия является страной."
]

# === 2. Ручная разметка (ground truth) ===
# 0 = корректно, 1 = галлюцинация

ground_truth = [
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
]

# === 3. Запуск pipeline ===
pipeline = HallucinationPipeline()
pipeline.initialize()

y_true = []
y_pred = []

for text, gt_list in zip(texts, ground_truth):
    results = pipeline.analyze_text(text)

    for res, gt in zip(results, gt_list):
        pred = 1 if res["hallucination"] else 0

        y_true.append(gt)
        y_pred.append(pred)

# === 4. Метрики ===
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
accuracy = accuracy_score(y_true, y_pred)

print("=== LLM Experiment ===")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")
print(f"Accuracy: {accuracy:.3f}")