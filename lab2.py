import re
import numpy as np
import pandas as pd
import pymorphy3
import nltk
import nltk.pathsec
nltk.pathsec.ALLOW_PROXIED_FETCH = True
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка стоп-слов nltk, если уже есть — пропуск
nltk.download('stopwords', quiet=True)
russian_stopwords = set(stopwords.words('russian'))

# Инициализация морфологического анализатора
morph = pymorphy3.MorphAnalyzer()

# Кэш лемматизации для ускорения работы pymorphy3
lemma_cache = {}


def get_lemma(word):
    if word not in lemma_cache:
        lemma_cache[word] = morph.parse(word)[0].normal_form
    return lemma_cache[word]


# Чтение документов и исследование данных
print("=== Загрузка и проверка данных ===")

docs_df = pd.read_csv('documents.csv', encoding='utf-8-sig')
queries_df = pd.read_csv('queries.csv', encoding='utf-8-sig')
dev_rel_df = pd.read_csv('dev_relevance.csv', encoding='utf-8-sig')

print(f"Количество документов: {len(docs_df)}")
print(f"Количество запросов: {len(queries_df)}")
print(f"Открытых эталонных пар: {len(dev_rel_df)}\n")

print("Типы данных в documents.csv:")
print(docs_df.dtypes)

print("\nПропуски в documents.csv:")
print(docs_df.isnull().sum())

print(f"\nУникальных ID документов: {docs_df['document_id'].nunique()} из {len(docs_df)}")
print(f"Полных дубликатов строк: {docs_df.duplicated().sum()}")
print(f"Дубликатов по полю text: {docs_df['text'].duplicated().sum()}\n")

# Формирование полного текста документа: заголовок + раздел + текст
docs_df['full_text'] = docs_df['title'] + " " + docs_df['section'] + " " + docs_df['text']


# Одинаковая предобработка текста для документов и запросов
def preprocess_text(text, use_lemmatization=False, remove_stopwords=False):
    if not isinstance(text, str) or not text.strip():
        return ""

    # Приведение к нижнему регистру и фильтрация
    tokens = re.findall(r'[а-яёa-z0-9]+', text.lower())

    if use_lemmatization:
        tokens = [get_lemma(t) for t in tokens]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in russian_stopwords]

    return " ".join(tokens)


# Базовый поиск по пересечению слов
print("=== Базовый поиск по пересечению слов ===")


def baseline_word_search(query_text, docs_df, top_k=5):
    # Обработка запроса
    q_clean = preprocess_text(query_text, use_lemmatization=True, remove_stopwords=True)
    q_words = set(q_clean.split())

    # Обработка пустого запроса
    if not q_words:
        return pd.DataFrame(columns=['document_id', 'score'])

    scores = []
    for idx, row in docs_df.iterrows():
        d_clean = preprocess_text(row['full_text'], use_lemmatization=True, remove_stopwords=True)
        d_words = set(d_clean.split())

        # Подсчет совпавших уникальных слов
        overlap = len(q_words.intersection(d_words))
        scores.append(overlap)

    res = pd.DataFrame({
        'document_id': docs_df['document_id'],
        'score': scores
    })

    # Сортировка по убыванию совпадений, при равенстве — по ID
    res = res.sort_values(by=['score', 'document_id'], ascending=[False, True]).head(top_k)
    return res


# Проверка базового поиска
sample_q = queries_df.iloc[0]['query']
print(f"Тест базового поиска на запросе: '{sample_q}'")
print(baseline_word_search(sample_q, docs_df, top_k=5).to_string(index=False))
print(f"Проверка пустого запроса, строк возвращено: {len(baseline_word_search('   ', docs_df))}\n")


# Поисковая система TF-IDF + косинусное сходство
print("=== Сравнение конфигураций A, B, C, D и расчёт метрик ===")

configs = {
    'A': {
        'description': 'без лемматизации, стоп-слова остаются, униграммы',
        'lemmatize': False,
        'remove_stops': False,
        'ngram_range': (1, 1)
    },
    'B': {
        'description': 'с лемматизацией, стоп-слова остаются, униграммы',
        'lemmatize': True,
        'remove_stops': False,
        'ngram_range': (1, 1)
    },
    'C': {
        'description': 'с лемматизацией, стоп-слова удаляются, униграммы',
        'lemmatize': True,
        'remove_stops': True,
        'ngram_range': (1, 1)
    },
    'D': {
        'description': 'с лемматизацией, стоп-слова удаляются, униграммы + биграммы',
        'lemmatize': True,
        'remove_stops': True,
        'ngram_range': (1, 2)
    }
}


def run_tfidf_pipeline(config_params, docs_df, queries_df, top_k=5):
    # Единая обработка корпуса и запросов
    processed_docs = [
        preprocess_text(t, config_params['lemmatize'], config_params['remove_stops'])
        for t in docs_df['full_text']
    ]
    processed_queries = [
        preprocess_text(q, config_params['lemmatize'], config_params['remove_stops'])
        for q in queries_df['query']
    ]

    # Обучение векторизатора на документах
    vectorizer = TfidfVectorizer(ngram_range=config_params['ngram_range'])
    doc_matrix = vectorizer.fit_transform(processed_docs)

    # Перевод запросов в то же пространство
    query_matrix = vectorizer.transform(processed_queries)

    # Расчет косинусного сходства
    sim_matrix = cosine_similarity(query_matrix, doc_matrix)

    # Формирование ТОП-5 для каждого запроса
    all_results = []
    for q_idx, q_row in queries_df.iterrows():
        q_id = q_row['query_id']
        q_text_clean = processed_queries[q_idx]

        if not q_text_clean.strip():
            scores = np.zeros(len(docs_df))
        else:
            scores = sim_matrix[q_idx]

        top_indices = np.argsort(scores)[::-1][:top_k]

        for rank, doc_idx in enumerate(top_indices, start=1):
            all_results.append({
                'query_id': q_id,
                'rank': rank,
                'document_id': docs_df.iloc[doc_idx]['document_id'],
                'score': round(float(scores[doc_idx]), 6)
            })

    return pd.DataFrame(all_results)


# Оценка Hit@5 и MRR@5
def evaluate_predictions(results_df, dev_rel_df):
    hits = []
    reciprocal_ranks = []

    for _, row in dev_rel_df.iterrows():
        q_id = row['query_id']
        true_doc_id = row['relevant_document_id']

        sub = results_df[results_df['query_id'] == q_id].sort_values('rank')
        predicted_docs = list(sub['document_id'])

        if true_doc_id in predicted_docs:
            hits.append(1.0)
            rank_pos = predicted_docs.index(true_doc_id) + 1
            reciprocal_ranks.append(1.0 / rank_pos)
        else:
            hits.append(0.0)
            reciprocal_ranks.append(0.0)

    return np.mean(hits), np.mean(reciprocal_ranks)


metrics_rows = []
results_by_config = {}

for cfg_name, cfg_params in configs.items():
    res_df = run_tfidf_pipeline(cfg_params, docs_df, queries_df, top_k=5)
    results_by_config[cfg_name] = res_df

    hit_5, mrr_5 = evaluate_predictions(res_df, dev_rel_df)
    metrics_rows.append({
        'configuration': cfg_name,
        'hit_at_5': round(hit_5, 4),
        'mrr_at_5': round(mrr_5, 4)
    })
    print(f"Конфигурация {cfg_name} {cfg_params['description']}: Hit@5 = {hit_5:.4f}, MRR@5 = {mrr_5:.4f}")

# Сохранение метрик
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv('experiment_metrics.csv', index=False, encoding='utf-8-sig')
print("\nФайл experiment_metrics.csv успешно сохранён.")

# Анализ ошибок для конфигурации C
print("\n=== Анализ ошибок на открытых запросах, Конфигурация C ===")
best_res = results_by_config['C']
for _, row in dev_rel_df.iterrows():
    q_id = row['query_id']
    true_doc = row['relevant_document_id']
    q_text = queries_df[queries_df['query_id'] == q_id]['query'].values[0]

    sub = best_res[best_res['query_id'] == q_id].sort_values('rank')
    top1_doc = sub.iloc[0]['document_id']

    if top1_doc != true_doc:
        true_rank = list(sub['document_id']).index(true_doc) + 1 if true_doc in list(sub['document_id']) else -1
        top1_title = docs_df[docs_df['document_id'] == top1_doc]['title'].values[0]
        true_title = docs_df[docs_df['document_id'] == true_doc]['title'].values[0]
        print(f"\nЗапрос {q_id}: {q_text}")
        print(f"  Ожидался документ: {true_doc} {true_title} -> оказался на позиции {true_rank}")
        print(f"  Топ-1 выдачи:      {top1_doc} {top1_title}, score={sub.iloc[0]['score']:.4f}")

# Сохранение результатов для конфигурации C
best_res.to_csv('search_results.csv', index=False, encoding='utf-8-sig')
print(f"\nФайл search_results.csv сохранён, строк: {len(best_res)}.")