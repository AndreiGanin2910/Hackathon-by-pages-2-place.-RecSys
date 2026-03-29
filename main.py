"""
================================================================================
  Book Recommendation System
================================================================================

  Гибридная рекомендательная система для предсказания книг, которые пользователь
  захочет прочитать или добавить в список желаний.

  Архитектура решения:
  ────────────────────
  1. КОЛЛАБОРАТИВНАЯ ФИЛЬТРАЦИЯ (6 моделей):
     • ALS (Alternating Least Squares) — основная модель, 96 факторов
     • ALS-Recent — ALS на данных за последние 90 дней
     • ALS-2 — облегчённая версия ALS (48 факторов, другая регуляризация)
     • ALS-TimedDecay — ALS с экспоненциальным затуханием весов по времени
     • BPR (Bayesian Personalized Ranking) — 64 фактора, 100 итераций
     • SVD-CF (TruncatedSVD) — SVD на матрице взаимодействий, 32 компоненты

  2. КОНТЕНТНАЯ ФИЛЬТРАЦИЯ:
     • TF-IDF + TruncatedSVD (dim=24) для текстовых эмбеддингов
       (название + описание + имя автора)
     • Item2Vec (Word2Vec на последовательностях взаимодействий)

  3. FEATURE ENGINEERING (~219 признаков):
     • Пользовательские агрегаты (активность, предпочтения, статистики)
     • Жанровые признаки (TF-IDF, новизна, Jaccard, рецентность, лояльность)
     • Языковые признаки (доля, энтропия, совпадение с предпочтениями)
     • Авторские признаки (популярность, лояльность, target encoding)
     • Издательские признаки (target encoding, доля)
     • Временные паттерны (тренды, рецентность, день недели)
     • CF-скоры и их ранги, z-нормализации, кросс-фичи

  4. МОДЕЛИ РАНЖИРОВАНИЯ (ансамбль):
     • CatBoostRanker (YetiRank) — 4 модели с разных временных окон
     • CatBoostClassifier (MultiClass: no_event / wish / read) — 4 модели

  5. ПОСТОБРАБОТКА:
     • Калибровка вероятностей (temperature scaling + per-user scaling)
     • Diversity-aware reranking (MMR-подобный) с учётом:
       - nDCG-оптимальности
       - Жанрового покрытия (coverage)
       - Внутрисписковой разнообразности (ILD)
       - Ограничений на дубликаты книг и авторов

  Схема обучения:
  ───────────────
  Используется 5 временных окон (cutoffs), сдвинутых назад на 7/10/15/20/25 дней
  от максимальной даты в данных. Первое окно (7 дней) — валидация,
  остальные 4 — обучение. Для каждого окна:
    past  = все взаимодействия до cutoff
    future = взаимодействия в [cutoff, cutoff + 30 дней)
    label  = 3 (read), 1 (wish), 0 (нет события)
"""

# ══════════════════════════════════════════════════════════════════════════════
#  ИМПОРТЫ И НАСТРОЙКА ОКРУЖЕНИЯ
# ══════════════════════════════════════════════════════════════════════════════

import os
import time
import gc
import re
import warnings
import argparse

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from catboost import CatBoostRanker, CatBoostClassifier, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from scipy import sparse
from threadpoolctl import threadpool_limits

import implicit

# Ограничиваем число потоков для воспроизводимости BLAS-операций
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


# ══════════════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def now():
    """Текущее время в формате HH:MM:SS для логирования."""
    return time.strftime("%H:%M:%S")


def sigmoid(x):
    """Сигмоидная функция с клиппингом для численной стабильности."""
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p, eps=1e-6):
    """Обратная сигмоида (logit) с защитой от крайних значений."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def jaccard_distance(a, b):
    """
    Расстояние Жаккара между двумя множествами.
    Возвращает 0, если оба множества пусты; иначе 1 - |A∩B| / |A∪B|.
    """
    if (not a) and (not b):
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


# ══════════════════════════════════════════════════════════════════════════════
#  ПОДГОТОВКА ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════════

def build_editions_item(editions, book_genres):
    """
    Создаёт обогащённую таблицу изданий с жанрами.

    Объединяет метаданные изданий (автор, издатель, год, язык, возраст)
    со списком жанров из book_genres. Результат — основная item-таблица,
    используемая во всём пайплайне.

    Returns:
        DataFrame с колонками: edition_id, book_id, author_id, publisher_id,
        publication_year, age_restriction, language_id, genre_list, genre_cnt
    """
    # Собираем жанры в список для каждой книги
    bg = (book_genres
          .groupby("book_id")["genre_id"]
          .apply(list)
          .rename("genre_list")
          .reset_index())

    # Берём ключевые атрибуты изданий и присоединяем жанры
    item = editions[[
        "edition_id", "book_id", "author_id", "publisher_id",
        "publication_year", "age_restriction", "language_id"
    ]].merge(bg, on="book_id", how="left")

    # Заполняем пропуски пустыми списками и считаем число жанров
    item["genre_list"] = item["genre_list"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    item["genre_cnt"] = item["genre_list"].apply(len).astype(np.int16)

    # Приводим типы, заполняем NaN значением -1
    for c in ["author_id", "publisher_id", "language_id",
              "age_restriction", "publication_year", "book_id"]:
        item[c] = item[c].fillna(-1).astype(np.int32)

    return item


def make_label_from_future(future):
    """
    Создаёт метки из будущих взаимодействий.

    Для каждой пары (user_id, edition_id) в будущем окне:
      - label = 3, если пользователь прочитал (event_type == 2)
      - label = 1, если только добавил в «хочу прочитать» (event_type == 1)
      - label = 0, если нет событий (используется для негативных примеров)

    Чтение приоритетнее — если есть и wish, и read, ставим label=3.
    """
    g = future.groupby(["user_id", "edition_id"])["event_type"].agg(
        has_read=lambda x: int((x == 2).any()),
        has_wish=lambda x: int((x == 1).any())
    ).reset_index()

    g["label"] = np.where(
        g["has_read"] == 1, 3,
        np.where(g["has_wish"] == 1, 1, 0)
    ).astype(np.int8)

    return g[["user_id", "edition_id", "label"]]


def ensure_grouped(df, group_col="user_id"):
    """
    Гарантирует, что данные отсортированы по group_col.
    Необходимо для корректной работы CatBoost с group_id.
    Используем mergesort для стабильности порядка.
    """
    return df.sort_values(group_col, kind="mergesort").reset_index(drop=True)


def downsample_negatives_per_user(df, neg_per_user=120, seed=42, verbose=True):
    """
    Ограничивает число негативных примеров на пользователя.

    При большом числе кандидатов (~200 на пользователя) класс 0 (нет события)
    доминирует. Для баланса оставляем не более neg_per_user негативов
    на каждого пользователя, сохраняя все положительные примеры.

    Args:
        df: DataFrame с колонкой 'label' (0 = neg, >0 = pos)
        neg_per_user: максимум негативов на пользователя
        seed: random seed для воспроизводимости сэмплирования
    """
    rng = np.random.RandomState(seed)
    pos = df[df["label"] > 0]
    neg = df[df["label"] == 0]
    parts = [pos]

    for uid, g in neg.groupby("user_id", sort=False):
        if len(g) > neg_per_user:
            parts.append(g.sample(neg_per_user, random_state=rng))
        else:
            parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    if verbose:
        print(f"[{now()}][DS] before={len(df):,} after={len(out):,} "
              f"pos={len(pos):,} neg={len(out) - len(pos):,}")
    return out


def precompute_user_demo(users):
    """
    Подготавливает демографические признаки пользователей.

    Создаёт:
      - gender: пол (0 = не указан)
      - age_num: числовой возраст (-1 если не указан)
      - age_bucket: возрастная группа (бакет) для категориальной фичи
        Группы: [0, 12, 17, 24, 34, 44, 54, 200]
    """
    u_demo = users[["user_id", "gender", "age"]].copy()
    u_demo["gender"] = u_demo["gender"].fillna(0).astype(np.int8)
    u_demo["age_num"] = u_demo["age"].fillna(-1).astype(np.float32)

    age = u_demo["age"].fillna(-1)
    bins = [-2, 0, 12, 17, 24, 34, 44, 54, 200]
    u_demo["age_bucket"] = pd.cut(age, bins=bins, labels=False).fillna(0).astype(np.int8)

    return u_demo[["user_id", "gender", "age_bucket", "age_num"]]


# ══════════════════════════════════════════════════════════════════════════════
#  ТЕКСТОВЫЕ ПРИЗНАКИ (TF-IDF + SVD)
# ══════════════════════════════════════════════════════════════════════════════

_html_re = re.compile(r"<[^>]+>")


def clean_text(s):
    """Очищает текст: удаляет HTML-теги, лишние пробелы, приводит к нижнему регистру."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = _html_re.sub(" ", s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def build_text_embeddings(editions, authors, text_dim=24, max_features=35000,
                          ngram_range=(1, 2), min_df=3,
                          cache_path="text_svd_embeddings.parquet", verbose=True):
    """
    Строит текстовые эмбеддинги для всех изданий.

    Пайплайн:
      1. Объединяет название + описание + имя автора
      2. TF-IDF (max_features=35000, uni+bigrams)
      3. TruncatedSVD для снижения размерности до text_dim

    Результат кэшируется в parquet-файл для ускорения повторных запусков.

    Дополнительно вычисляет текстовые мета-признаки:
      - title_len: длина названия в символах
      - desc_len: длина описания в символах
      - desc_word_cnt: число слов в описании
      - title_has_digit: содержит ли название цифру

    Returns:
        DataFrame: [edition_id, text_svd_0..text_svd_{dim-1},
                     title_len, desc_len, desc_word_cnt, title_has_digit]
    """
    # Проверяем наличие кэша
    if os.path.exists(cache_path):
        if verbose:
            print(f"[{now()}][TEXT] load cache: {cache_path}")
        emb = pd.read_parquet(cache_path)
        need = ({"edition_id"}
                | {f"text_svd_{i}" for i in range(text_dim)}
                | {"title_len", "desc_len", "desc_word_cnt", "title_has_digit"})
        if need.issubset(set(emb.columns)):
            return emb

    t0 = time.time()

    # Подготовка данных
    ed = editions[["edition_id", "author_id", "title", "description"]].copy()
    ed["author_id"] = ed["author_id"].fillna(-1).astype(np.int64)

    au = authors[["author_id", "author_name"]].copy()
    au["author_id"] = au["author_id"].fillna(-1).astype(np.int64)
    ed = ed.merge(au, on="author_id", how="left")
    ed["author_name"] = ed["author_name"].fillna("")

    # Очистка текстов
    for c in ["title", "description", "author_name"]:
        ed[c] = ed[c].fillna("").map(clean_text)

    # Мета-признаки из текста
    ed["title_len"] = ed["title"].str.len().astype(np.int16)
    ed["desc_len"] = ed["description"].str.len().astype(np.int32)
    ed["desc_word_cnt"] = ed["description"].str.split().map(len).astype(np.int16)
    ed["title_has_digit"] = (ed["title"]
                             .str.contains(r"\d", regex=True)
                             .fillna(False)
                             .astype(np.int8))

    # Конкатенация текстовых полей
    corpus = (ed["title"] + " " + ed["description"] + " " + ed["author_name"]).astype(str)
    if verbose:
        print(f"[{now()}][TEXT] fit TFIDF on {len(corpus):,} editions...")

    # TF-IDF векторизация
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        token_pattern=r"(?u)\b\w\w+\b",
        dtype=np.float32
    )
    X = vec.fit_transform(corpus)

    if verbose:
        print(f"[{now()}][TEXT] TFIDF shape={X.shape}, nnz={X.nnz:,}. SVD dim={text_dim} ...")

    # Снижение размерности
    svd = TruncatedSVD(n_components=text_dim, random_state=42)
    Z = svd.fit_transform(X).astype(np.float32)

    # Формируем результат
    emb_cols = [f"text_svd_{i}" for i in range(text_dim)]
    emb = pd.DataFrame(Z, columns=emb_cols)
    emb.insert(0, "edition_id", ed["edition_id"].values)
    emb["title_len"] = ed["title_len"].values
    emb["desc_len"] = ed["desc_len"].values
    emb["desc_word_cnt"] = ed["desc_word_cnt"].values
    emb["title_has_digit"] = ed["title_has_digit"].values

    # Сохраняем кэш
    emb.to_parquet(cache_path, index=False)
    if verbose:
        print(f"[{now()}][TEXT] saved {cache_path} in {time.time() - t0:.1f}s")

    return emb


# ══════════════════════════════════════════════════════════════════════════════
#  КОЛЛАБОРАТИВНАЯ ФИЛЬТРАЦИЯ — МОДЕЛИ
# ══════════════════════════════════════════════════════════════════════════════

def als_score_pairs(past, pairs, factors=96, iters=18, reg=0.05, alpha=15.0):
    """
    ALS (Alternating Least Squares) — основная CF-модель.

    Построение матрицы взаимодействий:
      - read (event_type=2): вес = 3.0 + 0.1 * rating
      - wish (event_type=1): вес = 1.0
    Матрица умножается на alpha перед подачей в ALS.

    Args:
        past: DataFrame взаимодействий (user_id, edition_id, event_type, rating)
        pairs: DataFrame пар (user_id, edition_id) для скоринга
        factors: число латентных факторов
        iters: число итераций ALS
        reg: коэффициент регуляризации
        alpha: множитель уверенности (confidence scaling)

    Returns:
        np.array скоров (dot product user_factors @ item_factors) для каждой пары
    """
    # Создаём маппинги ID → индекс
    user_ids = pd.Index(pairs["user_id"].unique())
    item_ids = pd.Index(past["edition_id"].unique()).union(pairs["edition_id"].unique())
    uid2idx = pd.Series(np.arange(len(user_ids), dtype=np.int32), index=user_ids)
    iid2idx = pd.Series(np.arange(len(item_ids), dtype=np.int32), index=item_ids)

    # Фильтруем и маппим
    df = past[["user_id", "edition_id", "event_type", "rating"]].copy()
    df = df[df["user_id"].isin(uid2idx.index) & df["edition_id"].isin(iid2idx.index)]
    u = uid2idx.loc[df["user_id"]].to_numpy(np.int32)
    i = iid2idx.loc[df["edition_id"]].to_numpy(np.int32)

    # Формируем веса: чтение = 3 + 0.1*rating, остальное = 1
    rating = np.nan_to_num(df["rating"].to_numpy(), nan=0.0).astype(np.float32)
    evt = df["event_type"].to_numpy()
    w = np.where(evt == 2, 3.0 + 0.1 * rating, 1.0).astype(np.float32)

    # Строим разреженную матрицу и обучаем
    X = sparse.csr_matrix((w, (u, i)), shape=(len(user_ids), len(item_ids)))
    model = implicit.als.AlternatingLeastSquares(
        factors=factors, iterations=iters, regularization=reg,
        use_gpu=False, random_state=42
    )
    with threadpool_limits(limits=1, user_api="blas"):
        model.fit(X.T.tocsr() * alpha)

    # Извлекаем факторы (implicit может менять порядок user/item)
    UF = model.user_factors
    IF = model.item_factors
    UF = UF.to_numpy() if hasattr(UF, "to_numpy") else np.asarray(UF)
    IF = IF.to_numpy() if hasattr(IF, "to_numpy") else np.asarray(IF)

    if UF.shape[0] == len(item_ids) and IF.shape[0] == len(user_ids):
        uf, itf = IF, UF
    elif UF.shape[0] == len(user_ids) and IF.shape[0] == len(item_ids):
        uf, itf = UF, IF
    else:
        raise RuntimeError("ALS factor shape mismatch")

    # Скорим пары через dot product
    cu = uid2idx.loc[pairs["user_id"]].to_numpy(np.int32)
    ci = iid2idx.loc[pairs["edition_id"]].to_numpy(np.int32)
    return np.sum(uf[cu] * itf[ci], axis=1).astype(np.float32)


def als2_score_pairs(past, pairs, factors=48, iters=25, reg=0.1, alpha=8.0):
    """
    ALS-2 — облегчённая версия ALS с другими гиперпараметрами.

    Отличия от основного ALS:
      - Меньше факторов (48 vs 96) — захватывает другие паттерны
      - Более сильная регуляризация (0.1 vs 0.05)
      - Упрощённые веса: read=2.0, wish=1.0 (без учёта рейтинга)
      - Другой random_state (123) — для разнообразия ансамбля
    """
    user_ids = pd.Index(pairs["user_id"].unique())
    item_ids = pd.Index(past["edition_id"].unique()).union(pairs["edition_id"].unique())
    uid2idx = pd.Series(np.arange(len(user_ids), dtype=np.int32), index=user_ids)
    iid2idx = pd.Series(np.arange(len(item_ids), dtype=np.int32), index=item_ids)

    df = past[past["user_id"].isin(uid2idx.index) & past["edition_id"].isin(iid2idx.index)]
    u = uid2idx.loc[df["user_id"]].to_numpy(np.int32)
    i = iid2idx.loc[df["edition_id"]].to_numpy(np.int32)
    w = np.where(df["event_type"].to_numpy() == 2, 2.0, 1.0).astype(np.float32)

    X = sparse.csr_matrix((w, (u, i)), shape=(len(user_ids), len(item_ids)))
    model = implicit.als.AlternatingLeastSquares(
        factors=factors, iterations=iters, regularization=reg,
        use_gpu=False, random_state=123
    )
    with threadpool_limits(limits=1, user_api="blas"):
        model.fit(X.T.tocsr() * alpha)

    UF = model.user_factors
    IF = model.item_factors
    UF = UF.to_numpy() if hasattr(UF, "to_numpy") else np.asarray(UF)
    IF = IF.to_numpy() if hasattr(IF, "to_numpy") else np.asarray(IF)

    if UF.shape[0] == len(item_ids) and IF.shape[0] == len(user_ids):
        uf, itf = IF, UF
    elif UF.shape[0] == len(user_ids) and IF.shape[0] == len(item_ids):
        uf, itf = UF, IF
    else:
        raise RuntimeError("ALS2 factor shape mismatch")

    cu = uid2idx.loc[pairs["user_id"]].to_numpy(np.int32)
    ci = iid2idx.loc[pairs["edition_id"]].to_numpy(np.int32)
    return np.sum(uf[cu] * itf[ci], axis=1).astype(np.float32)


def bpr_score_pairs(past, pairs, factors=64, iters=100, reg=0.01, lr=0.05):
    """
    BPR (Bayesian Personalized Ranking) — модель попарного ранжирования.

    В отличие от ALS (pointwise), BPR оптимизирует порядок:
    для каждого пользователя модель учится ранжировать наблюдаемые элементы
    выше случайных (негативных). Бинарная матрица (есть взаимодействие / нет).

    Дополняет ALS, т.к. оптимизирует другую целевую функцию (AUC vs MSE).
    """
    user_ids = pd.Index(pairs["user_id"].unique())
    item_ids = pd.Index(past["edition_id"].unique()).union(pairs["edition_id"].unique())
    uid2idx = pd.Series(np.arange(len(user_ids), dtype=np.int32), index=user_ids)
    iid2idx = pd.Series(np.arange(len(item_ids), dtype=np.int32), index=item_ids)

    df = past[past["user_id"].isin(uid2idx.index) & past["edition_id"].isin(iid2idx.index)]
    u = uid2idx.loc[df["user_id"]].to_numpy(np.int32)
    i = iid2idx.loc[df["edition_id"]].to_numpy(np.int32)

    # BPR работает с бинарной матрицей
    X = sparse.csr_matrix(
        (np.ones(len(u), dtype=np.float32), (u, i)),
        shape=(len(user_ids), len(item_ids))
    )

    model = implicit.bpr.BayesianPersonalizedRanking(
        factors=factors, iterations=iters, regularization=reg,
        learning_rate=lr, use_gpu=False, random_state=42
    )
    with threadpool_limits(limits=1, user_api="blas"):
        model.fit(X)

    UF = model.user_factors
    IF = model.item_factors
    UF = UF.to_numpy() if hasattr(UF, "to_numpy") else np.asarray(UF)
    IF = IF.to_numpy() if hasattr(IF, "to_numpy") else np.asarray(IF)

    if UF.shape[0] == len(user_ids) and IF.shape[0] == len(item_ids):
        uf, itf = UF, IF
    elif UF.shape[0] == len(item_ids) and IF.shape[0] == len(user_ids):
        itf, uf = UF, IF
    else:
        raise RuntimeError("BPR factor shape mismatch")

    cu = uid2idx.loc[pairs["user_id"]].to_numpy(np.int32)
    ci = iid2idx.loc[pairs["edition_id"]].to_numpy(np.int32)
    return np.sum(uf[cu] * itf[ci], axis=1).astype(np.float32)


def als_timedecay_score_pairs(past, pairs, ref_ts,
                              factors=64, iters=20, reg=0.08,
                              alpha=12.0, half_life=60):
    """
    ALS с экспоненциальным затуханием весов по времени.

    Идея: недавние взаимодействия важнее старых. Вес каждого события
    умножается на exp(-ln(2) * days_ago / half_life), где half_life=60 дней.
    Через 60 дней вес события уменьшается вдвое.

    Это помогает модели адаптироваться к смене интересов пользователя.
    """
    user_ids = pd.Index(pairs["user_id"].unique())
    item_ids = pd.Index(past["edition_id"].unique()).union(pairs["edition_id"].unique())
    uid2idx = pd.Series(np.arange(len(user_ids), dtype=np.int32), index=user_ids)
    iid2idx = pd.Series(np.arange(len(item_ids), dtype=np.int32), index=item_ids)

    df = past[past["user_id"].isin(uid2idx.index) &
              past["edition_id"].isin(iid2idx.index)].copy()
    u = uid2idx.loc[df["user_id"]].to_numpy(np.int32)
    i = iid2idx.loc[df["edition_id"]].to_numpy(np.int32)

    # Вычисляем временной decay
    days = (pd.Timestamp(ref_ts) - df["event_ts"]).dt.total_seconds().to_numpy() / 86400.0
    decay = np.exp(-np.log(2) * days / half_life).astype(np.float32)

    # Базовые веса (как в основном ALS) × decay
    rating = np.nan_to_num(df["rating"].to_numpy(), nan=0.0).astype(np.float32)
    evt = df["event_type"].to_numpy()
    w = (np.where(evt == 2, 3.0 + 0.1 * rating, 1.0) * decay).astype(np.float32)

    X = sparse.csr_matrix((w, (u, i)), shape=(len(user_ids), len(item_ids)))
    model = implicit.als.AlternatingLeastSquares(
        factors=factors, iterations=iters, regularization=reg,
        use_gpu=False, random_state=77
    )
    with threadpool_limits(limits=1, user_api="blas"):
        model.fit(X.T.tocsr() * alpha)

    UF = model.user_factors
    IF = model.item_factors
    UF = UF.to_numpy() if hasattr(UF, "to_numpy") else np.asarray(UF)
    IF = IF.to_numpy() if hasattr(IF, "to_numpy") else np.asarray(IF)

    if UF.shape[0] == len(item_ids) and IF.shape[0] == len(user_ids):
        uf, itf = IF, UF
    elif UF.shape[0] == len(user_ids) and IF.shape[0] == len(item_ids):
        uf, itf = UF, IF
    else:
        raise RuntimeError("ALS-TimedDecay factor shape mismatch")

    cu = uid2idx.loc[pairs["user_id"]].to_numpy(np.int32)
    ci = iid2idx.loc[pairs["edition_id"]].to_numpy(np.int32)
    return np.sum(uf[cu] * itf[ci], axis=1).astype(np.float32)


def svd_collab_score_pairs(past, pairs, n_components=32, verbose=True):
    """
    SVD-CF — коллаборативная фильтрация через TruncatedSVD.

    Строит матрицу взаимодействий (read=3.0, wish=1.0), затем
    TruncatedSVD с 32 компонентами. Скор = dot(user_embedding, item_embedding).

    Отличие от implicit ALS: sklearn SVD — это прямое разложение матрицы,
    без итеративной оптимизации. Даёт другой «взгляд» на данные.
    """
    t0 = time.time()
    user_ids = pd.Index(pairs["user_id"].unique())
    item_ids = pd.Index(past["edition_id"].unique()).union(pairs["edition_id"].unique())
    uid2idx = pd.Series(np.arange(len(user_ids), dtype=np.int32), index=user_ids)
    iid2idx = pd.Series(np.arange(len(item_ids), dtype=np.int32), index=item_ids)

    df = past[past["user_id"].isin(uid2idx.index) & past["edition_id"].isin(iid2idx.index)]
    u = uid2idx.loc[df["user_id"]].to_numpy(np.int32)
    i = iid2idx.loc[df["edition_id"]].to_numpy(np.int32)
    w = np.where(df["event_type"].to_numpy() == 2, 3.0, 1.0).astype(np.float32)

    X = sparse.csr_matrix((w, (u, i)), shape=(len(user_ids), len(item_ids)))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    uf = svd.fit_transform(X).astype(np.float32)
    itf = svd.components_.T.astype(np.float32)

    cu = uid2idx.loc[pairs["user_id"]].to_numpy(np.int32)
    ci = iid2idx.loc[pairs["edition_id"]].to_numpy(np.int32)
    scores = np.sum(uf[cu] * itf[ci], axis=1).astype(np.float32)

    if verbose:
        print(f"[{now()}][SVD-CF] done in {time.time() - t0:.1f}s, k={n_components}")

    del X, uf, itf
    gc.collect()
    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  ITEM2VEC (Word2Vec на последовательностях взаимодействий)
# ══════════════════════════════════════════════════════════════════════════════

def build_item2vec_model(interactions, dim=64, window=10, epochs=15,
                         min_count=2, verbose=True):
    """
    Обучает Item2Vec модель (Word2Vec на «предложениях» из edition_id).

    Каждый пользователь — «документ», его хронологически упорядоченные
    взаимодействия (edition_id) — «слова». Word2Vec (Skip-gram) учит
    эмбеддинги, в которых items из похожих контекстов близки.

    Позволяет вычислять косинусную близость между:
      - кандидатом и средним профилем пользователя
      - кандидатом и последним прочитанным/добавленным
    """
    from gensim.models import Word2Vec

    t0 = time.time()

    # Формируем последовательности: для каждого юзера — список edition_id в хронологии
    df = interactions.sort_values(["user_id", "event_ts"])
    seqs = (df.groupby("user_id")["edition_id"]
            .apply(lambda x: [str(e) for e in x.tolist()])
            .tolist())

    if verbose:
        print(f"[{now()}][I2V] {len(seqs)} sequences, dim={dim} window={window} ...")

    model = Word2Vec(
        sentences=seqs,
        vector_size=dim,
        window=window,
        min_count=min_count,
        sg=1,           # Skip-gram
        workers=1,
        epochs=epochs,
        seed=42
    )

    if verbose:
        print(f"[{now()}][I2V] vocab={len(model.wv)} trained in {time.time() - t0:.1f}s")

    return model


def item2vec_user_scores(i2v_model, past, pairs, top_k=20):
    """
    Вычисляет Item2Vec-based скоры для пар (user, item).

    Для каждого пользователя берём последние top_k взаимодействий,
    получаем их векторы из Item2Vec и вычисляем:
      - cos_mean: косинус между кандидатом и средним вектором профиля
      - cos_max: максимальный косинус между кандидатом и любым из top_k элементов

    cos_max хорошо ловит точечное совпадение (нужна одна похожая книга),
    cos_mean — общую тематическую близость.
    """
    wv = i2v_model.wv
    dim = wv.vector_size

    # Строим пользовательские профили из последних top_k взаимодействий
    user_items = (past.sort_values(["user_id", "event_ts"])
                  .groupby("user_id")["edition_id"]
                  .apply(lambda x: x.tail(top_k).tolist()))

    user_vecs, user_item_vecs = {}, {}
    for uid, items in user_items.items():
        vecs = [wv[str(e)] for e in items if str(e) in wv]
        if vecs:
            mat = np.array(vecs, dtype=np.float32)
            user_vecs[uid] = mat.mean(axis=0)
            user_item_vecs[uid] = mat
        else:
            user_vecs[uid] = np.zeros(dim, dtype=np.float32)
            user_item_vecs[uid] = None

    # Скорим каждую пару
    n = len(pairs)
    cos_mean = np.zeros(n, dtype=np.float32)
    cos_max = np.zeros(n, dtype=np.float32)
    uids = pairs["user_id"].to_numpy()
    eids = pairs["edition_id"].to_numpy()

    for idx in range(n):
        s = str(eids[idx])
        if s not in wv:
            continue
        iv = wv[s]
        inrm = np.linalg.norm(iv)
        if inrm < 1e-8:
            continue

        uid = uids[idx]

        # Косинус со средним вектором
        uv = user_vecs.get(uid)
        if uv is not None:
            un = np.linalg.norm(uv)
            if un > 1e-8:
                cos_mean[idx] = float(np.dot(iv, uv) / (inrm * un))

        # Максимальный косинус с отдельными элементами
        um = user_item_vecs.get(uid)
        if um is not None:
            dots = um @ iv
            nrms = np.maximum(np.linalg.norm(um, axis=1) * inrm, 1e-8)
            cos_max[idx] = float((dots / nrms).max())

    return cos_mean, cos_max


def item2vec_last_item_cos(i2v_model, past, pairs, event_type_filter=None):
    """
    Косинусная близость между кандидатом и последним элементом пользователя.

    Если event_type_filter задан (2=read, 1=wish), берём последний элемент
    указанного типа. Иначе — последний элемент любого типа.

    Полезно для sequential-рекомендаций: если пользователь только что
    прочитал фантастику, следующая книга скорее тоже будет похожей.
    """
    wv = i2v_model.wv
    filt = past[past["event_type"] == event_type_filter] if event_type_filter else past

    if len(filt) == 0:
        return np.zeros(len(pairs), dtype=np.float32)

    # Находим последний элемент каждого пользователя
    last = (filt.sort_values(["user_id", "event_ts"])
            .drop_duplicates("user_id", keep="last")[["user_id", "edition_id"]])
    ld = dict(zip(last["user_id"], last["edition_id"]))

    n = len(pairs)
    out = np.zeros(n, dtype=np.float32)
    uids = pairs["user_id"].to_numpy()
    eids = pairs["edition_id"].to_numpy()

    for idx in range(n):
        le = ld.get(uids[idx])
        if le is None:
            continue
        cs, ls = str(eids[idx]), str(le)
        if cs not in wv or ls not in wv:
            continue
        v1, v2 = wv[cs], wv[ls]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-8 and n2 > 1e-8:
            out[idx] = float(np.dot(v1, v2) / (n1 * n2))

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  ПОЛЬЗОВАТЕЛЬСКИЕ АГРЕГАТЫ
# ══════════════════════════════════════════════════════════════════════════════

def precompute_user_aggs(past, editions_item, ed_genre_long, ref_ts):
    """
    Вычисляет агрегированные признаки по каждому пользователю.

    Группы признаков:
      1. Базовая активность: u_cnt, u_read_cnt, u_wish_cnt, u_n_editions
      2. Рейтинги: u_mean_rating, u_median_rating, u_rating_std, u_rating_cnt
      3. Временные: u_days_since_last, u_active_days, u_activity_rate
      4. Оконные (7/14/30 дней): u_cnt_7, u_cnt_14, u_cnt_30, u_activity_trend
      5. Diversity: u_n_authors, u_n_genres, u_genre_entropy, u_genre_concentration
      6. Предпочтения: u_avg_pub_year, u_read_wish_ratio
      7. Скорость чтения: u_reads_30, u_read_velocity
      8. Паттерны: u_median_gap_hours, u_active_days_cnt
    """
    past_end = past["event_ts"].max()
    if pd.isna(past_end):
        past_end = pd.Timestamp(ref_ts)
    reads = past[past["event_type"] == 2]

    # ── 1. Базовая активность ──
    u = past.groupby("user_id").agg(
        u_cnt=("edition_id", "size"),
        u_read_cnt=("event_type", lambda x: int((x == 2).sum())),
        u_wish_cnt=("event_type", lambda x: int((x == 1).sum())),
        u_first=("event_ts", "min"),
        u_last=("event_ts", "max"),
        u_n_editions=("edition_id", "nunique"),
    ).reset_index()

    u["u_read_share"] = (u["u_read_cnt"] / u["u_cnt"].replace(0, np.nan)).fillna(0).astype(np.float32)
    u["u_days_since_last"] = (
        (pd.Timestamp(ref_ts) - u["u_last"]).dt.total_seconds() / 86400.0
    ).fillna(9999).astype(np.float32)

    uad = (u["u_last"] - u["u_first"]).dt.total_seconds() / 86400.0 + 1
    u["u_activity_rate"] = (u["u_cnt"] / uad.replace(0, np.nan)).fillna(0).astype(np.float32)
    u["u_active_days"] = uad.fillna(0).astype(np.float32)
    u.drop(columns=["u_last", "u_first"], inplace=True)

    for c in ["u_cnt", "u_read_cnt", "u_wish_cnt", "u_n_editions"]:
        u[c] = u[c].astype(np.int32)

    # ── 2. Рейтинги ──
    ur = reads.groupby("user_id")["rating"].agg(
        u_mean_rating="mean",
        u_rating_cnt="size",
        u_rating_std="std"
    ).reset_index()
    ur["u_mean_rating"] = ur["u_mean_rating"].fillna(0).astype(np.float32)
    ur["u_rating_cnt"] = ur["u_rating_cnt"].fillna(0).astype(np.int32)
    ur["u_rating_std"] = ur["u_rating_std"].fillna(0).astype(np.float32)
    u = u.merge(ur, on="user_id", how="left")
    u["u_mean_rating"] = u["u_mean_rating"].fillna(0).astype(np.float32)
    u["u_rating_cnt"] = u["u_rating_cnt"].fillna(0).astype(np.int32)
    u["u_rating_std"] = u["u_rating_std"].fillna(0).astype(np.float32)

    ur_med = reads.groupby("user_id")["rating"].median().rename("u_median_rating").reset_index()
    u = u.merge(ur_med, on="user_id", how="left")
    u["u_median_rating"] = u["u_median_rating"].fillna(0).astype(np.float32)

    u["u_read_wish_ratio"] = (
        u["u_read_cnt"] / u["u_wish_cnt"].replace(0, np.nan)
    ).fillna(0).clip(0, 100).astype(np.float32)

    # ── 3. Число активных дней ──
    u_adc = (past.groupby("user_id")["event_ts"]
             .apply(lambda x: x.dt.date.nunique())
             .rename("u_active_days_cnt")
             .reset_index())
    u = u.merge(u_adc, on="user_id", how="left")
    u["u_active_days_cnt"] = u["u_active_days_cnt"].fillna(0).astype(np.int32)

    # ── 4. Оконные счётчики (30/14/7 дней) ──
    for days, nm in [(30, "30"), (14, "14"), (7, "7")]:
        pw = past[past["event_ts"] >= (past_end - pd.Timedelta(days=days))]
        uw = pw.groupby("user_id").size().rename(f"u_cnt_{nm}").reset_index()
        u = u.merge(uw, on="user_id", how="left")
        u[f"u_cnt_{nm}"] = u[f"u_cnt_{nm}"].fillna(0).astype(np.int32)

    # Тренд активности: нормализованная 7-дневная активность к 30-дневной
    u["u_activity_trend"] = (
        (u["u_cnt_7"] * 4.28 + 1) / (u["u_cnt_30"] + 1)
    ).astype(np.float32)

    # ── 5. Медианный интервал между событиями ──
    def _median_gap(g):
        s = g.sort_values()
        if len(s) < 2:
            return np.nan
        return s.diff().dropna().dt.total_seconds().median() / 3600.0

    ug = (past.groupby("user_id")["event_ts"]
          .apply(_median_gap)
          .rename("u_median_gap_hours")
          .reset_index())
    u = u.merge(ug, on="user_id", how="left")
    u["u_median_gap_hours"] = u["u_median_gap_hours"].fillna(-1).astype(np.float32)

    # ── 6. Разнообразие: авторы ──
    pba = past.merge(editions_item[["edition_id", "author_id"]], on="edition_id", how="left")
    ua = pba.groupby("user_id")["author_id"].nunique().rename("u_n_authors").reset_index()
    u = u.merge(ua, on="user_id", how="left")
    u["u_n_authors"] = u["u_n_authors"].fillna(0).astype(np.int32)

    # ── 7. Разнообразие: жанры ──
    ugdf = (past[["user_id", "edition_id"]]
            .merge(ed_genre_long, on="edition_id", how="left")
            .dropna(subset=["genre_id"]))
    ugn = ugdf.groupby("user_id")["genre_id"].nunique().rename("u_n_genres").reset_index()
    u = u.merge(ugn, on="user_id", how="left")
    u["u_n_genres"] = u["u_n_genres"].fillna(0).astype(np.int32)

    # ── 8. Средний год публикации предпочтений ──
    ppy = past.merge(editions_item[["edition_id", "publication_year"]], on="edition_id")
    apy = ppy.groupby("user_id")["publication_year"].mean().rename("u_avg_pub_year").reset_index()
    u = u.merge(apy, on="user_id", how="left")
    u["u_avg_pub_year"] = u["u_avg_pub_year"].fillna(2020).astype(np.float32)

    # ── 9. Жанровая энтропия и концентрация ──
    if len(ugdf) > 0:
        ug_cnt = ugdf.groupby(["user_id", "genre_id"]).size().rename("w").reset_index()
        ug_total = ug_cnt.groupby("user_id")["w"].sum().rename("total")
        ug_cnt = ug_cnt.merge(ug_total, on="user_id")
        ug_cnt["p"] = (ug_cnt["w"] / ug_cnt["total"]).astype(np.float32)

        # Энтропия Шеннона по жанрам
        u_genre_ent = (ug_cnt.groupby("user_id")["p"]
                       .apply(lambda x: float(-np.sum(x.to_numpy() * np.log(x.to_numpy() + 1e-12))))
                       .rename("u_genre_entropy")
                       .reset_index())
        # Концентрация — доля самого частого жанра
        u_genre_conc = (ug_cnt.groupby("user_id")["p"]
                        .max()
                        .rename("u_genre_concentration")
                        .reset_index())
        u = u.merge(u_genre_ent, on="user_id", how="left")
        u = u.merge(u_genre_conc, on="user_id", how="left")
        del ug_cnt, ug_total, u_genre_ent, u_genre_conc
    else:
        u["u_genre_entropy"] = 0.0
        u["u_genre_concentration"] = 0.0

    u["u_genre_entropy"] = u["u_genre_entropy"].fillna(0).astype(np.float32)
    u["u_genre_concentration"] = u["u_genre_concentration"].fillna(0).astype(np.float32)

    # ── 10. Скорость чтения (за последние 30 дней) ──
    u_reads_30 = (
        past[(past["event_type"] == 2) &
             (past["event_ts"] >= (past_end - pd.Timedelta(days=30)))]
        .groupby("user_id")["edition_id"]
        .nunique()
        .rename("u_reads_30")
        .reset_index()
    )
    u = u.merge(u_reads_30, on="user_id", how="left")
    u["u_reads_30"] = u["u_reads_30"].fillna(0).astype(np.int32)
    u["u_read_velocity"] = (u["u_reads_30"] / 4.28).astype(np.float32)  # книг/неделю

    del pba, ua, ugdf, ugn, ppy, apy, u_reads_30, u_adc, ur_med
    gc.collect()
    return u


# ══════════════════════════════════════════════════════════════════════════════
#  ЖАНРОВЫЕ ПРИЗНАКИ
# ══════════════════════════════════════════════════════════════════════════════

def build_genre_novelty_features(df, past, ed_genre_long):
    """
    Вычисляет жанровую новизну и взвешенное совпадение жанров.

    Признаки:
      - genre_novelty: доля жанров кандидата, которых нет в истории юзера
      - genre_overlap: доля жанров кандидата, совпадающих с историей
      - weighted_genre_match: сумма долей жанров кандидата в профиле юзера
        (взвешенных по частоте, read=3x, wish=1x)

    Новизна высокая → пользователь не читал такие жанры раньше.
    weighted_genre_match высокий → кандидат точно в его вкусе.
    """
    # Множество всех жанров пользователя
    user_genres_all = (
        past[["user_id", "edition_id"]]
        .merge(ed_genre_long, on="edition_id", how="left")
        .dropna(subset=["genre_id"])
        .groupby("user_id")["genre_id"]
        .apply(set)
        .to_dict()
    )

    # Взвешенный профиль жанров: read=3.0, wish=1.0, нормализация по юзеру
    past_wg = (
        past[["user_id", "edition_id", "event_type"]]
        .merge(ed_genre_long, on="edition_id", how="left")
        .dropna(subset=["genre_id"])
    )
    past_wg["w"] = np.where(past_wg["event_type"].to_numpy() == 2, 3.0, 1.0).astype(np.float32)
    ugw_raw = past_wg.groupby(["user_id", "genre_id"])["w"].sum().reset_index()
    ugw_total = ugw_raw.groupby("user_id")["w"].sum().rename("total")
    ugw_raw = ugw_raw.merge(ugw_total, on="user_id")
    ugw_raw["share"] = (ugw_raw["w"] / ugw_raw["total"]).astype(np.float32)

    # Словарь user_id → {genre_id: share}
    ugw_dict = {}
    for _, row in ugw_raw.iterrows():
        uid = row["user_id"]
        if uid not in ugw_dict:
            ugw_dict[uid] = {}
        ugw_dict[uid][int(row["genre_id"])] = float(row["share"])

    # Жанры каждого кандидата
    cand_genres = (
        df[["edition_id"]].drop_duplicates()
        .merge(ed_genre_long, on="edition_id", how="left")
        .dropna(subset=["genre_id"])
        .groupby("edition_id")["genre_id"]
        .apply(set)
        .to_dict()
    )

    # Вычисляем фичи для каждой пары
    uids = df["user_id"].to_numpy()
    eids = df["edition_id"].to_numpy()
    n = len(df)
    nov = np.zeros(n, dtype=np.float32)
    ovl = np.zeros(n, dtype=np.float32)
    wgm = np.zeros(n, dtype=np.float32)

    for idx in range(n):
        ug = user_genres_all.get(uids[idx], set())
        cg = cand_genres.get(eids[idx], set())
        if not cg:
            continue
        nov[idx] = len(cg - ug) / len(cg)
        ovl[idx] = (len(cg & ug) / len(cg)) if ug else 0.0
        ugw = ugw_dict.get(uids[idx], {})
        if ugw:
            wgm[idx] = sum(ugw.get(int(g), 0.0) for g in cg)

    df["genre_novelty"] = nov
    df["genre_overlap"] = ovl
    df["weighted_genre_match"] = wgm.astype(np.float32)

    del ugw_raw, ugw_total, past_wg, ugw_dict
    gc.collect()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ЯЗЫКОВЫЕ ПРИЗНАКИ
# ══════════════════════════════════════════════════════════════════════════════

def add_language_features(df, past, editions_item):
    """
    Добавляет признаки языковых предпочтений пользователя.

    Признаки:
      - u_lang_cnt: сколько раз юзер взаимодействовал с данным языком
      - u_lang_share: доля данного языка в истории юзера
      - lang_is_top: совпадает ли язык с самым частым у юзера
      - u_lang_entropy: энтропия по языкам (низкая = читает на одном языке)
      - u_lang_unique: число уникальных языков
      - same_lang_last_read/wish: совпадает ли язык с последним прочитанным/wish
    """
    past_lang = (
        past[["user_id", "edition_id", "event_type", "event_ts"]]
        .merge(editions_item[["edition_id", "language_id"]], on="edition_id", how="left")
    )
    past_lang["language_id"] = past_lang["language_id"].fillna(-1).astype(np.int32)

    # Общее число взаимодействий юзера
    ut = past_lang.groupby("user_id").size().rename("u_total_lang_cnt")

    # Число взаимодействий по (юзер, язык)
    ul = past_lang.groupby(["user_id", "language_id"]).size().rename("ul_cnt").reset_index()

    df = df.merge(ul, on=["user_id", "language_id"], how="left")
    df["ul_cnt"] = df["ul_cnt"].fillna(0).astype(np.int32)
    df["u_total_lang_cnt"] = df["user_id"].map(ut).fillna(0).astype(np.int32)
    df["u_lang_cnt"] = df["ul_cnt"].astype(np.int32)
    df["u_lang_share"] = (
        df["u_lang_cnt"] / df["u_total_lang_cnt"].replace(0, np.nan)
    ).fillna(0).astype(np.float32)

    # Самый частый язык юзера
    ul_sorted = (ul.sort_values(["user_id", "ul_cnt"], ascending=[True, False])
                 .drop_duplicates("user_id"))
    ul_sorted = ul_sorted.rename(columns={"language_id": "u_top_lang"})
    df = df.merge(ul_sorted[["user_id", "u_top_lang"]], on="user_id", how="left")
    df["u_top_lang"] = df["u_top_lang"].fillna(-1).astype(np.int32)
    df["lang_is_top"] = (df["language_id"] == df["u_top_lang"]).astype(np.int8)

    # Энтропия по языкам
    tmp = ul.copy()
    tmp["u_total"] = tmp["user_id"].map(ut).fillna(0).astype(np.int32)
    tmp["p"] = (tmp["ul_cnt"] / tmp["u_total"].replace(0, np.nan)).fillna(0).astype(np.float32)
    ent = (tmp.groupby("user_id")["p"]
           .apply(lambda x: float(-np.sum(x.to_numpy() * np.log(x.to_numpy() + 1e-12))))
           .rename("u_lang_entropy")
           .reset_index())
    uniq = tmp.groupby("user_id").size().rename("u_lang_unique").reset_index()

    df = df.merge(ent, on="user_id", how="left").merge(uniq, on="user_id", how="left")
    df["u_lang_entropy"] = df["u_lang_entropy"].fillna(0).astype(np.float32)
    df["u_lang_unique"] = df["u_lang_unique"].fillna(0).astype(np.int16)

    # Совпадение языка с последним прочитанным / wish
    def _last_lang(event_type):
        pp = past_lang[past_lang["event_type"] == event_type]
        if len(pp) == 0:
            return pd.DataFrame({
                "user_id": df["user_id"].unique(),
                f"last_lang_{event_type}": -1
            })
        last = (pp.sort_values(["user_id", "event_ts"])
                .drop_duplicates("user_id", keep="last"))
        return last[["user_id", "language_id"]].rename(
            columns={"language_id": f"last_lang_{event_type}"}
        )

    lr = _last_lang(2)  # последний read
    lw = _last_lang(1)  # последний wish
    df = df.merge(lr, on="user_id", how="left").merge(lw, on="user_id", how="left")
    df["last_lang_2"] = df["last_lang_2"].fillna(-1).astype(np.int32)
    df["last_lang_1"] = df["last_lang_1"].fillna(-1).astype(np.int32)
    df["same_lang_last_read"] = (df["language_id"] == df["last_lang_2"]).astype(np.int8)
    df["same_lang_last_wish"] = (df["language_id"] == df["last_lang_1"]).astype(np.int8)

    # Очистка промежуточных колонок
    df.drop(columns=["ul_cnt", "u_total_lang_cnt", "u_top_lang",
                      "last_lang_2", "last_lang_1"], inplace=True)
    del past_lang, ut, ul, ul_sorted, tmp, ent, uniq, lr, lw
    gc.collect()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ПОСЛЕДОВАТЕЛЬНЫЕ (SEQUENTIAL) ПРИЗНАКИ
# ══════════════════════════════════════════════════════════════════════════════

def add_sequential_features(df, past, editions_item):
    """
    Признаки тренда по авторам за последние 7 и 30 дней.

    - ua_cnt_30: число взаимодействий с данным автором за 30 дней
    - ua_cnt_7: число взаимодействий с данным автором за 7 дней
    - ua_trend: нормализованный тренд (рост интереса к автору)
    """
    past_auth = (
        past[["user_id", "edition_id", "event_ts", "event_type"]]
        .merge(editions_item[["edition_id", "author_id"]], on="edition_id", how="left")
    )
    past_end = past["event_ts"].max()

    # 30-дневное окно
    p30_auth = past_auth[past_auth["event_ts"] >= (past_end - pd.Timedelta(days=30))]
    ua30 = p30_auth.groupby(["user_id", "author_id"]).size().rename("ua_cnt_30").reset_index()
    df = df.merge(ua30, on=["user_id", "author_id"], how="left")
    df["ua_cnt_30"] = df["ua_cnt_30"].fillna(0).astype(np.int16)

    # 7-дневное окно
    p7_auth = past_auth[past_auth["event_ts"] >= (past_end - pd.Timedelta(days=7))]
    ua7 = p7_auth.groupby(["user_id", "author_id"]).size().rename("ua_cnt_7").reset_index()
    df = df.merge(ua7, on=["user_id", "author_id"], how="left")
    df["ua_cnt_7"] = df["ua_cnt_7"].fillna(0).astype(np.int16)

    # Тренд: 7-дневная активность × 4.28 (нормализация к месяцу) vs 30-дневная
    df["ua_trend"] = ((df["ua_cnt_7"] * 4.28 + 1) / (df["ua_cnt_30"] + 1)).astype(np.float32)

    del past_auth, p30_auth, ua30, p7_auth, ua7
    gc.collect()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ЖАНРОВЫЙ TF-IDF
# ══════════════════════════════════════════════════════════════════════════════

def add_genre_tfidf_features(df, past, ed_genre_long):
    """
    Жанровый TF-IDF: насколько жанры кандидата соответствуют
    «уникальному вкусу» пользователя.

    TF = частота жанра у юзера (read=3x, wish=1x)
    IDF = log(N_users / (1 + df_genre)) — редкие жанры ценнее

    Признаки: genre_tfidf_sum, genre_tfidf_max, genre_tfidf_mean
    по жанрам кандидата.
    """
    user_genre = (
        past[["user_id", "edition_id", "event_type"]]
        .merge(ed_genre_long, on="edition_id", how="left")
        .dropna(subset=["genre_id"])
    )
    if len(user_genre) == 0:
        df["genre_tfidf_sum"] = 0.0
        df["genre_tfidf_max"] = 0.0
        df["genre_tfidf_mean"] = 0.0
        return df

    user_genre["w"] = np.where(user_genre["event_type"].to_numpy() == 2, 3.0, 1.0).astype(np.float32)
    ug_agg = user_genre.groupby(["user_id", "genre_id"])["w"].sum().reset_index()

    # IDF: обратная документная частота по жанрам
    n_users = past["user_id"].nunique()
    genre_df = user_genre.groupby("genre_id")["user_id"].nunique().rename("df").reset_index()
    genre_df["idf"] = np.log(n_users / (1.0 + genre_df["df"])).astype(np.float32)

    ug_agg = ug_agg.merge(genre_df[["genre_id", "idf"]], on="genre_id", how="left")
    ug_agg["tfidf"] = (ug_agg["w"] * ug_agg["idf"]).astype(np.float32)

    # Для каждой пары (юзер, кандидат) суммируем TF-IDF по совпадающим жанрам
    cand_genres_df = (
        df[["user_id", "edition_id"]].drop_duplicates()
        .merge(ed_genre_long, on="edition_id", how="left")
        .dropna(subset=["genre_id"])
    )
    merged = cand_genres_df.merge(
        ug_agg[["user_id", "genre_id", "tfidf"]],
        on=["user_id", "genre_id"], how="left"
    )
    merged["tfidf"] = merged["tfidf"].fillna(0).astype(np.float32)

    gt = merged.groupby(["user_id", "edition_id"]).agg(
        genre_tfidf_sum=("tfidf", "sum"),
        genre_tfidf_max=("tfidf", "max"),
        genre_tfidf_mean=("tfidf", "mean")
    ).reset_index()

    df = df.merge(gt, on=["user_id", "edition_id"], how="left")
    for c in ["genre_tfidf_sum", "genre_tfidf_max", "genre_tfidf_mean"]:
        df[c] = df[c].fillna(0).astype(np.float32)

    del user_genre, ug_agg, genre_df, cand_genres_df, merged, gt
    gc.collect()
    return df


def add_recency_genre_features(df, past, ed_genre_long, ref_ts):
    """
    Рецентность жанров: насколько недавно пользователь взаимодействовал
    с жанрами кандидата.

    Вес = exp(-days_ago / 30) × event_weight (read=3, wish=1).
    Суммируется по совпадающим жанрам → genre_recency_sum, genre_recency_max.
    """
    past_end = past["event_ts"].max()
    if pd.isna(past_end):
        past_end = pd.Timestamp(ref_ts)

    pg = (past[["user_id", "edition_id", "event_type", "event_ts"]]
          .merge(ed_genre_long, on="edition_id", how="left")
          .dropna(subset=["genre_id"]))

    if len(pg) == 0:
        df["genre_recency_sum"] = 0.0
        df["genre_recency_max"] = 0.0
        return df

    # Экспоненциальный decay: half-life ≈ 20.8 дней
    days_ago = (past_end - pg["event_ts"]).dt.total_seconds() / 86400.0
    decay = np.exp(-days_ago.to_numpy(np.float32) / 30.0).astype(np.float32)
    evt_w = np.where(pg["event_type"].to_numpy() == 2, 3.0, 1.0).astype(np.float32)
    pg["rw"] = (decay * evt_w).astype(np.float32)

    ug_rw = pg.groupby(["user_id", "genre_id"])["rw"].sum().reset_index()

    cand_g = (df[["user_id", "edition_id"]].drop_duplicates()
              .merge(ed_genre_long, on="edition_id", how="left")
              .dropna(subset=["genre_id"]))
    merged = cand_g.merge(ug_rw, on=["user_id", "genre_id"], how="left")
    merged["rw"] = merged["rw"].fillna(0).astype(np.float32)

    gt = merged.groupby(["user_id", "edition_id"]).agg(
        genre_recency_sum=("rw", "sum"),
        genre_recency_max=("rw", "max")
    ).reset_index()

    df = df.merge(gt, on=["user_id", "edition_id"], how="left")
    df["genre_recency_sum"] = df["genre_recency_sum"].fillna(0).astype(np.float32)
    df["genre_recency_max"] = df["genre_recency_max"].fillna(0).astype(np.float32)

    del pg, ug_rw, cand_g, merged, gt
    gc.collect()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ПРИЗНАКИ ИСТОРИИ ЮЗЕР–ЭЛЕМЕНТ
# ══════════════════════════════════════════════════════════════════════════════

def add_user_item_history_features(df, past, editions_item):
    """
    Признаки взаимодействия пользователя с конкретной книгой/автором.

    - ub_cnt: сколько раз юзер взаимодействовал с этой книгой (book_id)
    - ub_n_editions: сколько изданий этой книги юзер видел
    - ua_loyalty: доля взаимодействий с данным автором от всех
    - ua_user_read_rate: доля прочитанных среди всех взаимодействий с автором
    """
    # Взаимодействия с книгой (book_id может повторяться для разных edition_id)
    past_books = (past[["user_id", "edition_id"]]
                  .merge(editions_item[["edition_id", "book_id"]], on="edition_id", how="left"))
    ub = past_books.groupby(["user_id", "book_id"]).agg(
        ub_cnt=("edition_id", "size"),
        ub_n_editions=("edition_id", "nunique")
    ).reset_index()

    df = df.merge(ub, on=["user_id", "book_id"], how="left")
    df["ub_cnt"] = df["ub_cnt"].fillna(0).astype(np.int16)
    df["ub_n_editions"] = df["ub_n_editions"].fillna(0).astype(np.int16)

    # Лояльность к автору
    past_auth = (past[["user_id", "edition_id", "event_type"]]
                 .merge(editions_item[["edition_id", "author_id"]], on="edition_id", how="left"))
    u_total = past.groupby("user_id").size().rename("u_total_int")
    ua_total = past_auth.groupby(["user_id", "author_id"]).size().rename("ua_total_int").reset_index()
    ua_total["u_total_int"] = ua_total["user_id"].map(u_total).fillna(1).astype(np.int32)
    ua_total["ua_loyalty"] = (
        ua_total["ua_total_int"] / ua_total["u_total_int"].replace(0, np.nan)
    ).fillna(0).astype(np.float32)

    df = df.merge(ua_total[["user_id", "author_id", "ua_loyalty"]],
                  on=["user_id", "author_id"], how="left")
    df["ua_loyalty"] = df["ua_loyalty"].fillna(0).astype(np.float32)

    # Read rate по автору
    ua_reads = (past_auth[past_auth["event_type"] == 2]
                .groupby(["user_id", "author_id"]).size()
                .rename("ua_user_reads").reset_index())
    ua_all = past_auth.groupby(["user_id", "author_id"]).size().rename("ua_user_all").reset_index()
    ua_rr = ua_all.merge(ua_reads, on=["user_id", "author_id"], how="left")
    ua_rr["ua_user_reads"] = ua_rr["ua_user_reads"].fillna(0)
    ua_rr["ua_user_read_rate"] = (
        ua_rr["ua_user_reads"] / ua_rr["ua_user_all"].replace(0, np.nan)
    ).fillna(0).astype(np.float32)

    df = df.merge(ua_rr[["user_id", "author_id", "ua_user_read_rate"]],
                  on=["user_id", "author_id"], how="left")
    df["ua_user_read_rate"] = df["ua_user_read_rate"].fillna(0).astype(np.float32)

    del past_books, ub, past_auth, u_total, ua_total, ua_reads, ua_all, ua_rr
    gc.collect()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ПОЗИЦИОННЫЕ И РАНГОВЫЕ ПРИЗНАКИ КАНДИДАТОВ
# ══════════════════════════════════════════════════════════════════════════════

def add_candidate_position_features(df):
    """
    Ранг кандидата внутри списка пользователя по популярности и рейтингу.

    - cand_pop_rank: ранг по i_pop (чем популярнее, тем ниже ранг)
    - cand_pop_rank_pct: процентильный ранг по популярности
    - cand_rating_rank: ранг по среднему рейтингу
    """
    df["cand_pop_rank"] = (df.groupby("user_id")["i_pop"]
                           .rank(method="min", ascending=False)
                           .astype(np.float32))
    df["cand_pop_rank_pct"] = (
        df["cand_pop_rank"] / df.groupby("user_id")["i_pop"].transform("size")
    ).astype(np.float32)
    df["cand_rating_rank"] = (df.groupby("user_id")["i_mean_rating"]
                              .rank(method="min", ascending=False)
                              .astype(np.float32))
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ВРЕМЕННЫЕ ПАТТЕРНЫ
# ══════════════════════════════════════════════════════════════════════════════

def add_time_pattern_features(df, past, ref_ts):
    """
    Временные паттерны поведения пользователя.

    - u_top_dow: самый частый день недели активности
    - u_weekend_share: доля выходных среди всех дней активности
    - u_days_since_last_read: дней с последнего чтения
    - u_days_since_last_wish: дней с последнего добавления в wish
    """
    past_ts = past[["user_id", "event_ts", "event_type"]].copy()
    past_ts["dow"] = past_ts["event_ts"].dt.dayofweek.astype(np.int8)

    # Самый частый день недели
    u_dow = past_ts.groupby(["user_id", "dow"]).size().reset_index(name="cnt")
    u_dow_top = (u_dow.sort_values(["user_id", "cnt"], ascending=[True, False])
                 .drop_duplicates("user_id"))
    u_dow_top = u_dow_top.rename(columns={"dow": "u_top_dow"})
    df = df.merge(u_dow_top[["user_id", "u_top_dow"]], on="user_id", how="left")
    df["u_top_dow"] = df["u_top_dow"].fillna(0).astype(np.int8)

    # Доля выходных
    u_weekend = (past_ts.groupby("user_id")
                 .apply(lambda g: float((g["dow"] >= 5).sum()) / max(len(g), 1))
                 .rename("u_weekend_share")
                 .reset_index())
    df = df.merge(u_weekend, on="user_id", how="left")
    df["u_weekend_share"] = df["u_weekend_share"].fillna(0.286).astype(np.float32)

    # Дней с последнего чтения
    reads = past_ts[past_ts["event_type"] == 2]
    if len(reads) > 0:
        last_read = reads.groupby("user_id")["event_ts"].max().rename("u_last_read_ts").reset_index()
        df = df.merge(last_read, on="user_id", how="left")
        df["u_days_since_last_read"] = (
            (pd.Timestamp(ref_ts) - df["u_last_read_ts"]).dt.total_seconds() / 86400.0
        ).fillna(9999).astype(np.float32)
        df.drop(columns=["u_last_read_ts"], inplace=True)
    else:
        df["u_days_since_last_read"] = 9999.0

    # Дней с последнего wish
    wishes = past_ts[past_ts["event_type"] == 1]
    if len(wishes) > 0:
        last_wish = wishes.groupby("user_id")["event_ts"].max().rename("u_last_wish_ts").reset_index()
        df = df.merge(last_wish, on="user_id", how="left")
        df["u_days_since_last_wish"] = (
            (pd.Timestamp(ref_ts) - df["u_last_wish_ts"]).dt.total_seconds() / 86400.0
        ).fillna(9999).astype(np.float32)
        df.drop(columns=["u_last_wish_ts"], inplace=True)
    else:
        df["u_days_since_last_wish"] = 9999.0

    del past_ts, u_dow, u_dow_top, u_weekend
    gc.collect()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  КО-ИНТЕРАКЦИОННЫЕ ПРИЗНАКИ
# ══════════════════════════════════════════════════════════════════════════════

def add_cointeraction_features(df, past):
    """
    Признаки на уровне элемента (item-level):
      - i_n_users: число уникальных пользователей, взаимодействовавших с item
      - i_n_readers: число уникальных читателей
      - i_conversion: конверсия (n_readers / n_users)
    """
    i_users = past.groupby("edition_id")["user_id"].nunique().rename("i_n_users").reset_index()
    df = df.merge(i_users, on="edition_id", how="left")
    df["i_n_users"] = df["i_n_users"].fillna(0).astype(np.int32)

    i_readers = (past[past["event_type"] == 2]
                 .groupby("edition_id")["user_id"]
                 .nunique()
                 .rename("i_n_readers")
                 .reset_index())
    df = df.merge(i_readers, on="edition_id", how="left")
    df["i_n_readers"] = df["i_n_readers"].fillna(0).astype(np.int32)

    df["i_conversion"] = (
        df["i_n_readers"] / df["i_n_users"].replace(0, np.nan)
    ).fillna(0).astype(np.float32)

    del i_users, i_readers
    gc.collect()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  TARGET ENCODING ДЛЯ ИЗДАТЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════

def add_publisher_target_encoding(df, past, editions_item):
    """
    Target encoding для издателя: сглаженная доля прочитанных книг.

    te_pub_read_rate = (n_reads + SMOOTH × global_rate) / (n_total + SMOOTH)

    SMOOTH=50 — для защиты от переобучения на редких издателях.
    """
    past_pub = (past[["user_id", "edition_id", "event_type"]]
                .merge(editions_item[["edition_id", "publisher_id"]], on="edition_id", how="left"))

    global_read_rate = float((past["event_type"] == 2).mean())
    SMOOTH = 50

    pub_te = past_pub.groupby("publisher_id").agg(
        _n=("event_type", "size"),
        _nr=("event_type", lambda x: int((x == 2).sum()))
    ).reset_index()
    pub_te["te_pub_read_rate"] = (
        (pub_te["_nr"] + SMOOTH * global_read_rate) / (pub_te["_n"] + SMOOTH)
    ).astype(np.float32)

    df = df.merge(pub_te[["publisher_id", "te_pub_read_rate"]], on="publisher_id", how="left")
    df["te_pub_read_rate"] = df["te_pub_read_rate"].fillna(global_read_rate).astype(np.float32)

    del past_pub, pub_te
    gc.collect()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ЖАНРОВЫЕ ПОПАРНЫЕ ПРИЗНАКИ (Jaccard)
# ══════════════════════════════════════════════════════════════════════════════

def add_genre_pair_features(df, past, ed_genre_long):
    """
    Попарное сравнение жанров пользователя и кандидата через Jaccard.

    - genre_jaccard: |user_genres ∩ item_genres| / |user_genres ∪ item_genres|
    - genre_intersect_cnt: число общих жанров
    - genre_union_size: размер объединения
    """
    user_genre_set = (
        past[["user_id", "edition_id"]]
        .merge(ed_genre_long, on="edition_id", how="left")
        .dropna(subset=["genre_id"])
        .groupby("user_id")["genre_id"]
        .apply(set).to_dict()
    )
    item_genre_set = (
        df[["edition_id"]].drop_duplicates()
        .merge(ed_genre_long, on="edition_id", how="left")
        .dropna(subset=["genre_id"])
        .groupby("edition_id")["genre_id"]
        .apply(set).to_dict()
    )

    uids = df["user_id"].to_numpy()
    eids = df["edition_id"].to_numpy()
    n = len(df)
    jaccard_sim = np.zeros(n, dtype=np.float32)
    genre_intersect = np.zeros(n, dtype=np.float32)
    genre_union_size = np.zeros(n, dtype=np.float32)

    for idx in range(n):
        ug = user_genre_set.get(uids[idx], set())
        ig = item_genre_set.get(eids[idx], set())
        if not ug or not ig:
            continue
        inter = len(ug & ig)
        union = len(ug | ig)
        jaccard_sim[idx] = inter / union if union > 0 else 0.0
        genre_intersect[idx] = inter
        genre_union_size[idx] = union

    df["genre_jaccard"] = jaccard_sim
    df["genre_intersect_cnt"] = genre_intersect.astype(np.int16)
    df["genre_union_size"] = genre_union_size.astype(np.int16)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ВРЕМЕННАЯ ЛОЯЛЬНОСТЬ (АВТОР + ЖАНР)
# ══════════════════════════════════════════════════════════════════════════════

def add_temporal_loyalty_features(df, past, editions_item, ed_genre_long, ref_ts):
    """
    Признаки долгосрочной лояльности к авторам и жанрам.

    Авторы:
      - ua_days_since_first: дней с первого взаимодействия с автором
      - author_loyalty: интенсивность (n_interactions / span_days)

    Жанры (агрегаты по жанрам кандидата):
      - genre_loyalty_mean/max: средняя/макс интенсивность по жанрам
      - genre_days_since_first_min: минимум дней с первого контакта с жанром
    """
    t0 = time.time()

    # ── Лояльность к авторам ──
    past_auth = (past[["user_id", "edition_id", "event_ts"]]
                 .merge(editions_item[["edition_id", "author_id"]], on="edition_id", how="left"))

    ua_first_last = past_auth.groupby(["user_id", "author_id"]).agg(
        ua_first_ts=("event_ts", "min"),
        ua_last_ts=("event_ts", "max"),
        ua_total=("event_ts", "size")
    ).reset_index()

    ua_first_last["ua_span_days"] = (
        (ua_first_last["ua_last_ts"] - ua_first_last["ua_first_ts"]).dt.total_seconds() / 86400.0 + 1
    ).astype(np.float32)
    ua_first_last["ua_days_since_first"] = (
        (pd.Timestamp(ref_ts) - ua_first_last["ua_first_ts"]).dt.total_seconds() / 86400.0
    ).astype(np.float32)
    ua_first_last["author_loyalty"] = (
        ua_first_last["ua_total"] / ua_first_last["ua_span_days"]
    ).astype(np.float32)

    df = df.merge(
        ua_first_last[["user_id", "author_id", "ua_days_since_first", "author_loyalty"]],
        on=["user_id", "author_id"], how="left"
    )
    df["ua_days_since_first"] = df["ua_days_since_first"].fillna(9999).astype(np.float32)
    df["author_loyalty"] = df["author_loyalty"].fillna(0).astype(np.float32)

    del ua_first_last, past_auth
    gc.collect()

    # ── Лояльность к жанрам ──
    past_genre = (past[["user_id", "edition_id", "event_ts"]]
                  .merge(ed_genre_long, on="edition_id", how="left")
                  .dropna(subset=["genre_id"]))

    if len(past_genre) > 0:
        ug_first = (past_genre.groupby(["user_id", "genre_id"])["event_ts"]
                    .min().rename("ug_first_ts").reset_index())
        ug_cnt = (past_genre.groupby(["user_id", "genre_id"])
                  .size().rename("ug_cnt").reset_index())
        ug_first = ug_first.merge(ug_cnt, on=["user_id", "genre_id"])

        ug_first["ug_days_since_first"] = (
            (pd.Timestamp(ref_ts) - ug_first["ug_first_ts"]).dt.total_seconds() / 86400.0
        ).astype(np.float32)
        ug_first["genre_loyalty"] = (
            ug_first["ug_cnt"] / (ug_first["ug_days_since_first"] + 1)
        ).astype(np.float32)

        # Для кандидата агрегируем по его жанрам
        cand_genres = (df[["user_id", "edition_id"]].drop_duplicates()
                       .merge(ed_genre_long, on="edition_id", how="left")
                       .dropna(subset=["genre_id"]))
        cand_genres = cand_genres.merge(
            ug_first[["user_id", "genre_id", "ug_days_since_first", "genre_loyalty"]],
            on=["user_id", "genre_id"], how="left"
        )
        gl_agg = cand_genres.groupby(["user_id", "edition_id"]).agg(
            genre_loyalty_mean=("genre_loyalty", "mean"),
            genre_loyalty_max=("genre_loyalty", "max"),
            genre_days_since_first_min=("ug_days_since_first", "min")
        ).reset_index()

        df = df.merge(gl_agg, on=["user_id", "edition_id"], how="left")
        del ug_first, ug_cnt, cand_genres, gl_agg
    else:
        df["genre_loyalty_mean"] = 0.0
        df["genre_loyalty_max"] = 0.0
        df["genre_days_since_first_min"] = 9999.0

    df["genre_loyalty_mean"] = df["genre_loyalty_mean"].fillna(0).astype(np.float32)
    df["genre_loyalty_max"] = df["genre_loyalty_max"].fillna(0).astype(np.float32)
    df["genre_days_since_first_min"] = df["genre_days_since_first_min"].fillna(9999).astype(np.float32)

    del past_genre
    gc.collect()
    print(f"[{now()}][LOYALTY] done in {time.time() - t0:.1f}s")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  КОНВЕРСИЯ WISH → READ
# ══════════════════════════════════════════════════════════════════════════════

def add_item_conversion_rate(df, past):
    """
    Конверсия wish → read на уровне элемента.

    item_conversion_rate = |wish_users ∩ read_users| / |wish_users|

    Если много пользователей добавляли книгу в wish и потом прочитали —
    значит, книга «затягивает».
    """
    wish_users = (past[past["event_type"] == 1]
                  .groupby("edition_id")["user_id"]
                  .apply(set).to_dict())
    read_users = (past[past["event_type"] == 2]
                  .groupby("edition_id")["user_id"]
                  .apply(set).to_dict())

    eids_arr = df["edition_id"].to_numpy()
    n = len(df)
    conv = np.zeros(n, dtype=np.float32)

    for idx in range(n):
        eid = eids_arr[idx]
        wu = wish_users.get(eid, set())
        ru = read_users.get(eid, set())
        if len(wu) > 0:
            conv[idx] = len(wu & ru) / len(wu)

    df["item_conversion_rate"] = conv
    del wish_users, read_users
    gc.collect()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ПОПУЛЯРНОСТЬ С ВРЕМЕННЫМ ЗАТУХАНИЕМ
# ══════════════════════════════════════════════════════════════════════════════

def add_time_weighted_popularity(df, past, ref_ts):
    """
    Популярность с экспоненциальным затуханием.

    i_pop_tw7: half-life = 7 дней (быстро реагирует на тренды)
    i_pop_tw30: half-life = 30 дней (устойчивая популярность)

    Формула: sum(exp(-ln(2) * days_ago / half_life)) по всем взаимодействиям.
    """
    days_ago = (pd.Timestamp(ref_ts) - past["event_ts"]).dt.total_seconds().to_numpy() / 86400.0

    for hl, sfx in [(7, "tw7"), (30, "tw30")]:
        decay = np.exp(-np.log(2) * days_ago / hl).astype(np.float32)
        pt = past[["edition_id"]].copy()
        pt["w"] = decay
        ip = pt.groupby("edition_id")["w"].sum().rename(f"i_pop_{sfx}").reset_index()

        df = df.merge(ip, on="edition_id", how="left")
        df[f"i_pop_{sfx}"] = df[f"i_pop_{sfx}"].fillna(0).astype(np.float32)
        df[f"i_pop_{sfx}_log"] = np.log1p(df[f"i_pop_{sfx}"]).astype(np.float32)
        del pt, ip

    gc.collect()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def print_feature_importance(models, feat_cols, title="MODEL"):
    """
    Выводит топ-50 признаков по средней важности (PredictionValuesChange)
    по всем моделям ансамбля.
    """
    importances = np.zeros(len(feat_cols), dtype=np.float64)
    for model in models:
        fi = model.get_feature_importance(type="PredictionValuesChange")
        if fi.ndim > 1:
            fi = fi.sum(axis=1)
        importances += fi
    importances /= len(models)

    fi_df = (pd.DataFrame({"feature": feat_cols, "importance": importances})
             .sort_values("importance", ascending=False))

    print(f"\n{'=' * 80}")
    print(f"{title} FEATURE IMPORTANCE (avg over {len(models)} models)")
    print(f"{'=' * 80}")
    print(f"{'Rank':<6}{'Feature':<55}{'Importance':>12}")
    print(f"{'-' * 80}")
    for i, (_, row) in enumerate(fi_df.iterrows(), 1):
        if i <= 50:
            print(f"{i:<6}{row['feature']:<55}{row['importance']:>12.4f}")
    print(f"{'=' * 80}\n")

    return fi_df


# ══════════════════════════════════════════════════════════════════════════════
#  КАЛИБРОВКА ВЕРОЯТНОСТЕЙ
# ══════════════════════════════════════════════════════════════════════════════

def per_user_topk_mass_calibration(infer_df, topK=50, desired_psum_topK=3.5,
                                   T=0.9, max_scale=50.0, min_scale=0.2):
    """
    Калибровка вероятностей p_read и p_wish для reranking'а.

    Проблема: raw-вероятности классификатора могут быть слишком
    самоуверенными или наоборот. Для reranking'а нужны калиброванные p_rel.

    Алгоритм:
      1. p_rel_raw = p_read_raw + p_wish_raw
      2. Temperature scaling: p_rel_ts = σ(logit(p_rel_raw) / T)
      3. Per-user scaling: для каждого юзера масштабируем так, чтобы
         сумма p_rel по top-K кандидатам была ≈ desired_psum_topK
      4. p_read = p_rel × ratio (смесь item-level и global ratio)
      5. p_wish = p_rel - p_read
    """
    eps = 1e-9

    p_read_raw = infer_df["p_read_raw"].to_numpy(np.float32)
    p_wish_raw = infer_df["p_wish_raw"].to_numpy(np.float32)
    p_rel_raw = (p_read_raw + p_wish_raw).astype(np.float32)

    # Temperature scaling
    p_rel_ts = sigmoid(
        logit(np.clip(p_rel_raw, 1e-6, 1 - 1e-6)) / float(T)
    ).astype(np.float32)

    # Соотношение read / relevance на уровне элемента
    ratio_item = (p_read_raw / (p_rel_raw + eps)).astype(np.float32)
    global_ratio = float(np.clip(np.nanmean(ratio_item), 0.10, 0.90))

    infer_df["_p_rel_ts"] = p_rel_ts
    infer_df["_ratio_item"] = ratio_item

    # Per-user масштабирование
    scales = np.zeros(len(infer_df), dtype=np.float32)
    for uid, g in infer_df.groupby("user_id", sort=False):
        gg = g.sort_values("base_score", ascending=False).head(int(topK))
        s = float(gg["_p_rel_ts"].sum())
        sc = float(np.clip(float(desired_psum_topK) / (s + eps), min_scale, max_scale))
        scales[g.index.values] = sc

    p_rel = np.clip(p_rel_ts * scales, 0.0, 1.0).astype(np.float32)

    # Финальное разделение на p_read и p_wish
    ratio = np.clip(0.75 * ratio_item + 0.25 * global_ratio, 0.0, 1.0).astype(np.float32)
    infer_df["p_rel"] = p_rel
    infer_df["p_read"] = (p_rel * ratio).astype(np.float32)
    infer_df["p_wish"] = np.clip(
        p_rel - infer_df["p_read"].to_numpy(), 0.0, 1.0
    ).astype(np.float32)

    infer_df.drop(columns=["_p_rel_ts", "_ratio_item"], inplace=True)
    return infer_df


# ══════════════════════════════════════════════════════════════════════════════
#  DIVERSITY-AWARE RERANKING
# ══════════════════════════════════════════════════════════════════════════════

def expected_rerank_top20(u_df, genres_by_edition, author_by_edition=None,
                          forbid_same_book=True, topM=200, lock_n=8,
                          lambda_base=0.025, beta=0.5, alpha_ndcg=0.7,
                          tail_div_boost_start=1.5, tail_div_boost_end=3.0,
                          max_per_author_head=3, max_per_author_tail=1,
                          eps=1e-9):
    """
    Жадный MMR-подобный reranking с адаптивным переключением.

    Идея: первые lock_n позиций заполняем «жадно» по base_score (relevance),
    затем постепенно увеличиваем вес разнообразия.

    Целевая функция для хвостовых позиций:
      score = α × w_rel × Δ_nDCG
            + (1-α) × w_div × (β × Δ_coverage + (1-β) × Δ_ILD)
            + λ × base_z

    Где:
      - Δ_nDCG: вклад в нормализованную DCG (gain = 3×p_read + 1×p_wish)
      - Δ_coverage: вклад в жанровое покрытие (Coverage-based diversity)
      - Δ_ILD: вклад в внутрисписковую разнообразность (Intra-List Diversity)
      - base_z: z-нормализованный base_score (стабилизирует порядок)

    Ограничения:
      - Не более max_per_author_head одной автора в голове (первые lock_n)
      - Не более max_per_author_tail в хвосте
      - Запрет дублирования книг (одна book_id → одно место в списке)

    Adaptive lock: если разрыв в gain'ах между позициями lock_n и lock_n+1
    слишком мал (<0.05), уменьшаем lock_n чтобы раньше начать диверсификацию.

    Returns:
        list из 20 edition_id в порядке рекомендации
    """
    if len(u_df) == 0:
        return []

    # Берём topM лучших кандидатов
    u_df = u_df.sort_values("base_score", ascending=False).head(int(topM)).copy()
    eids = u_df["edition_id"].to_list()

    # Извлекаем словари для быстрого доступа
    base_z = dict(zip(u_df["edition_id"], u_df["base_z"]))
    p_read = dict(zip(u_df["edition_id"], u_df["p_read"].astype(np.float32)))
    p_wish = dict(zip(u_df["edition_id"], u_df["p_wish"].astype(np.float32)))
    p_rel = dict(zip(u_df["edition_id"], u_df["p_rel"].astype(np.float32)))
    book = dict(zip(u_df["edition_id"], u_df["book_id"]))

    # Gain для DCG: 3×p_read + 1×p_wish
    gain = {i: float(3.0 * p_read[i] + 1.0 * p_wish[i]) for i in eids}

    # DCG-веса по позициям (1/log2(k+1))
    w = [0.0] + [1.0 / np.log2(k + 1) for k in range(1, 21)]
    sum_w = float(np.sum(w[1:]))

    # Ideal DCG для нормализации
    gains_sorted = sorted([gain[i] for i in eids], reverse=True)[:20]
    ideal_dcg = float(np.sum([w[k] * gains_sorted[k - 1]
                               for k in range(1, 1 + len(gains_sorted))]))
    if ideal_dcg < 1e-8:
        ideal_dcg = 1.0

    # Адаптивный lock: если gap маленький, раньше начинаем диверсификацию
    sorted_gains_all = sorted([gain[i] for i in eids], reverse=True)
    effective_lock = lock_n
    if len(sorted_gains_all) > lock_n and lock_n > 3:
        gap = sorted_gains_all[lock_n - 1] - sorted_gains_all[lock_n]
        if gap < 0.05:
            effective_lock = max(5, lock_n - 3)

    # Состояние жадного выбора
    selected, selected_set = [], set()
    used_books = set()
    author_cnt = {}
    q_cov = {}          # жанровое покрытие: genre_id → вероятность покрытия
    sum_p = 0.0
    ild_num = 0.0       # числитель ILD
    ild_den = 0.0       # знаменатель ILD
    ild_cur = 0.0       # текущее значение ILD

    # --- Вспомогательные функции ---

    def author_ok(eid, k):
        """Проверяет, не превышен ли лимит автора для позиции k."""
        if author_by_edition is None:
            return True
        a = author_by_edition.get(eid, None)
        if a is None or a == -1:
            return True
        lim = max_per_author_head if k <= effective_lock else max_per_author_tail
        return author_cnt.get(a, 0) < lim

    def ok_constraints(eid, k):
        """Проверяет все ограничения для добавления eid на позицию k."""
        if eid in selected_set:
            return False
        if forbid_same_book:
            b = book.get(eid, None)
            if b is not None and b != -1 and b in used_books:
                return False
        return author_ok(eid, k)

    def add_item(eid, k):
        """Добавляет элемент в выборку и обновляет состояние."""
        nonlocal sum_p, ild_num, ild_den, ild_cur, q_cov

        pi = float(p_rel[eid])

        # Обновляем ILD
        if pi > 0.0 and len(selected) > 0:
            Gi = genres_by_edition.get(eid, set())
            s = 0.0
            for j in selected:
                pj = float(p_rel[j])
                if pj <= 0.0:
                    continue
                Gj = genres_by_edition.get(j, set())
                s += pj * jaccard_distance(Gi, Gj)
            ild_num += pi * s
            ild_den += pi * sum_p
            ild_cur = (ild_num / (ild_den + eps)) if ild_den > 0 else 0.0

        sum_p += pi

        # Обновляем жанровое покрытие
        Gi = genres_by_edition.get(eid, set())
        if Gi:
            omp = 1.0 - pi
            for g in Gi:
                qg = float(q_cov.get(g, 0.0))
                q_cov[g] = 1.0 - (1.0 - qg) * omp

        selected.append(eid)
        selected_set.add(eid)

        if forbid_same_book:
            b = book.get(eid, None)
            if b is not None and b != -1:
                used_books.add(b)

        if author_by_edition is not None:
            a = author_by_edition.get(eid, None)
            if a is not None and a != -1:
                author_cnt[a] = author_cnt.get(a, 0) + 1

    # ── Фаза 1: Жадный выбор (head) ──
    for k in range(1, effective_lock + 1):
        pick = None
        for eid in eids:
            if ok_constraints(eid, k):
                pick = eid
                break
        if pick is None:
            break
        add_item(pick, k)

    # ── Фаза 2: Diversity-aware выбор (tail) ──
    for k in range(len(selected) + 1, 21):
        # Линейная интерполяция весов: от head к tail
        progress = min(1.0, max(0.0,
            (k - effective_lock) / max(1, 20 - effective_lock)))
        w_div = tail_div_boost_start + progress * (tail_div_boost_end - tail_div_boost_start)
        w_rel = 1.0 - 0.08 * progress
        ild_before = float(ild_cur)

        best_eid, best_val = None, -1e18

        for eid in eids:
            if not ok_constraints(eid, k):
                continue

            # Δ nDCG: вклад в качество ранжирования
            d_ndcg = (w[k] * gain[eid]) / ideal_dcg

            # Δ Coverage: вклад в жанровое покрытие
            Gi = genres_by_edition.get(eid, set())
            if Gi:
                miss = sum(1.0 - float(q_cov.get(g, 0.0)) for g in Gi)
                d_cov = (w[k] / sum_w) * float(p_rel[eid]) * miss / float(len(Gi))
            else:
                d_cov = 0.0

            # Δ ILD: вклад в внутрисписковую разнообразность
            pi = float(p_rel[eid])
            if pi > 0.0 and len(selected) > 0:
                s = 0.0
                for j in selected:
                    pj = float(p_rel[j])
                    if pj <= 0.0:
                        continue
                    Gj = genres_by_edition.get(j, set())
                    s += pj * jaccard_distance(Gi, Gj)
                new_num = ild_num + pi * s
                new_den = ild_den + pi * sum_p
                d_ild = ((new_num / (new_den + eps)) if new_den > 0 else 0.0) - ild_before
            else:
                d_ild = 0.0

            # Итоговый скор: баланс relevance, diversity, stability
            val = (alpha_ndcg * w_rel * d_ndcg
                   + (1.0 - alpha_ndcg) * w_div * (beta * d_cov + (1.0 - beta) * d_ild)
                   + float(lambda_base) * float(base_z.get(eid, 0.0)))

            if val > best_val:
                best_val, best_eid = val, eid

        # Fallback: если ничего не подошло по ограничениям
        if best_eid is None:
            for eid in eids:
                if eid not in selected_set:
                    best_eid = eid
                    break
        if best_eid is None:
            break

        add_item(best_eid, k)

    # Дозаполняем до 20, если не хватило
    if len(selected) < 20:
        for eid in eids:
            if eid not in selected_set:
                selected.append(eid)
            if len(selected) >= 20:
                break

    return selected[:20]


# ══════════════════════════════════════════════════════════════════════════════
#  СБОРКА ВСЕХ ПРИЗНАКОВ
# ══════════════════════════════════════════════════════════════════════════════

def build_features(pairs, past, editions_item, u_demo, ref_ts,
                   ed_genre_long, pairs_ug, text_emb, text_dim=24,
                   add_als=False, als_params=None,
                   add_als_recent=False, als_recent_days=90,
                   add_bpr=False, bpr_params=None,
                   add_als2=False, add_als_td=False, add_svd_cf=False,
                   i2v_model=None, verbose=True):

    """Главная функция генерации признаков."""


    t0 = time.time()
    df = pairs.copy()
    emb_cols = [f"text_svd_{i}" for i in range(text_dim)]

    # ── Item metadata ──
    meta_cols = ["edition_id", "book_id", "author_id", "publisher_id",
                 "publication_year", "age_restriction", "language_id", "genre_cnt"]
    df = df.merge(editions_item[meta_cols], on="edition_id", how="left")

    # ── Text embeddings ──
    text_meta_cols = ["title_len", "desc_len", "desc_word_cnt", "title_has_digit"]
    df = df.merge(text_emb[["edition_id"] + emb_cols + text_meta_cols],
                  on="edition_id", how="left")
    for c in emb_cols:
        df[c] = df[c].fillna(0).astype(np.float32)
    for c in text_meta_cols:
        df[c] = df[c].fillna(0).astype(np.int32)

    # ── User demographics ──
    df = df.merge(u_demo, on="user_id", how="left")
    df["gender"] = df["gender"].fillna(0).astype(np.int8)
    df["age_bucket"] = df["age_bucket"].fillna(0).astype(np.int8)
    df["age_num"] = df["age_num"].fillna(-1).astype(np.float32)
    df["age_restriction"] = df["age_restriction"].fillna(-1).astype(np.int32)
    df["age_violation"] = (
        (df["age_num"] >= 0) & (df["age_restriction"] > 0) &
        (df["age_num"] < df["age_restriction"].astype(np.float32))
    ).astype(np.int8)

    # ── user_has_book: читал ли юзер уже эту книгу ──
    df_uids = df["user_id"].to_numpy()
    pub = (past[["user_id", "edition_id"]]
           .merge(editions_item[["edition_id", "book_id"]], on="edition_id")
           .groupby("user_id")["book_id"].apply(set).to_dict())
    db = df["book_id"].to_numpy()
    df["user_has_book"] = np.array(
        [1 if (pub.get(df_uids[i]) and db[i] in pub[df_uids[i]]) else 0
         for i in range(len(df))],
        dtype=np.int8
    )
    del pub
    gc.collect()

    # ── User aggregates ──
    ua = precompute_user_aggs(past, editions_item, ed_genre_long, ref_ts)
    df = df.merge(ua, on="user_id", how="left")
    for c in ua.columns:
        if c == "user_id":
            continue
        if df[c].dtype in [np.float32, np.float64]:
            df[c] = df[c].fillna(0).astype(np.float32)
        else:
            df[c] = df[c].fillna(0).astype(np.int32)
    del ua
    gc.collect()

    # ── Publication year diff (кандидат vs предпочтение юзера) ──
    df["pub_year_diff"] = (
        df["publication_year"].fillna(2020).astype(np.float32) - df["u_avg_pub_year"]
    ).astype(np.float32)

    # ── Author target encoding ──
    past_with_author = past.merge(
        editions_item[["edition_id", "author_id"]], on="edition_id", how="left"
    )
    global_read_rate = float((past["event_type"] == 2).mean())
    SMOOTH_ALPHA = 100

    auth_te = past_with_author.groupby("author_id").agg(
        _n=("event_type", "size"),
        _nr=("event_type", lambda x: int((x == 2).sum()))
    ).reset_index()
    auth_te["te_auth_read_rate"] = (
        (auth_te["_nr"] + SMOOTH_ALPHA * global_read_rate) / (auth_te["_n"] + SMOOTH_ALPHA)
    ).astype(np.float32)

    df = df.merge(auth_te[["author_id", "te_auth_read_rate"]], on="author_id", how="left")
    df["te_auth_read_rate"] = df["te_auth_read_rate"].fillna(global_read_rate).astype(np.float32)
    del past_with_author, auth_te
    gc.collect()

    # ── Все остальные feature-блоки ──
    df = build_genre_novelty_features(df, past, ed_genre_long)
    df["language_id"] = df["language_id"].fillna(-1).astype(np.int32)
    df = add_language_features(df, past, editions_item)
    df = add_sequential_features(df, past, editions_item)
    df = add_genre_tfidf_features(df, past, ed_genre_long)
    df = add_recency_genre_features(df, past, ed_genre_long, ref_ts)
    df = add_user_item_history_features(df, past, editions_item)
    df = add_time_pattern_features(df, past, ref_ts)
    df = add_cointeraction_features(df, past)
    df = add_publisher_target_encoding(df, past, editions_item)
    df = add_genre_pair_features(df, past, ed_genre_long)
    df = add_temporal_loyalty_features(df, past, editions_item, ed_genre_long, ref_ts)
    df = add_item_conversion_rate(df, past)
    df = add_time_weighted_popularity(df, past, ref_ts)

    # ── Item-level rating stats ──
    reads = past[past["event_type"] == 2][["edition_id", "rating"]].copy()
    i_rating = reads.groupby("edition_id").agg(
        i_mean_rating=("rating", "mean"),
        i_rating_cnt=("rating", "size")
    ).reset_index()
    df = df.merge(i_rating, on="edition_id", how="left")
    df["i_mean_rating"] = df["i_mean_rating"].fillna(0).astype(np.float32)
    df["i_rating_cnt"] = df["i_rating_cnt"].fillna(0).astype(np.int32)
    df["rating_diff"] = (df["u_mean_rating"] - df["i_mean_rating"]).astype(np.float32)
    df["rating_median_diff"] = (df["u_median_rating"] - df["i_mean_rating"]).astype(np.float32)
    del reads, i_rating
    gc.collect()

    # ── Item popularity (overall + window) ──
    past_end = past["event_ts"].max()
    if pd.isna(past_end):
        past_end = pd.Timestamp(ref_ts)

    ip = past.groupby("edition_id").agg(
        i_pop=("event_type", "size"),
        i_read=("event_type", lambda x: int((x == 2).sum())),
        i_wish=("event_type", lambda x: int((x == 1).sum())),
        i_last=("event_ts", "max")
    ).reset_index()
    df = df.merge(ip, on="edition_id", how="left")
    for c in ["i_pop", "i_read", "i_wish"]:
        df[c] = df[c].fillna(0).astype(np.int32)
    df["i_pop_log"] = np.log1p(df["i_pop"]).astype(np.float32)
    df["i_read_share"] = (df["i_read"] / df["i_pop"].replace(0, np.nan)).fillna(0).astype(np.float32)
    df["i_days_since_last"] = (
        (pd.Timestamp(ref_ts) - df["i_last"]).dt.total_seconds() / 86400.0
    ).fillna(9999).astype(np.float32)
    df.drop(columns=["i_last"], inplace=True)
    del ip
    gc.collect()

    # Оконная популярность (30 и 7 дней)
    past30 = past[past["event_ts"] >= (past_end - pd.Timedelta(days=30))]
    past7 = past[past["event_ts"] >= (past_end - pd.Timedelta(days=7))]

    for pf, nm in [(past30, "30"), (past7, "7")]:
        ipx = pf.groupby("edition_id").size().rename(f"i_pop_{nm}").reset_index()
        df = df.merge(ipx, on="edition_id", how="left")
        df[f"i_pop_{nm}"] = df[f"i_pop_{nm}"].fillna(0).astype(np.int32)
        df[f"i_pop_{nm}_log"] = np.log1p(df[f"i_pop_{nm}"]).astype(np.float32)
        del ipx

    df["i_pop_trend"] = (
        (df["i_pop_7"] * 4.28 + 1) / (df["i_pop_30"] + 1)
    ).astype(np.float32)

    # ── User-Author/Publisher interaction stats ──
    past_ba = past.merge(
        editions_item[["edition_id", "author_id", "book_id", "publisher_id"]],
        on="edition_id", how="left"
    )

    ua_agg = past_ba.groupby(["user_id", "author_id"]).agg(
        ua_cnt=("edition_id", "size"),
        ua_last=("event_ts", "max")
    ).reset_index()
    df = df.merge(ua_agg, on=["user_id", "author_id"], how="left")
    df["ua_cnt"] = df["ua_cnt"].fillna(0).astype(np.int16)
    df["ua_days_since_last"] = (
        (pd.Timestamp(ref_ts) - df["ua_last"]).dt.total_seconds() / 86400.0
    ).fillna(9999).astype(np.float32)
    df.drop(columns=["ua_last"], inplace=True)
    del ua_agg
    gc.collect()

    reads_ba = past_ba[past_ba["event_type"] == 2].copy()
    ua_ra = reads_ba.groupby(["user_id", "author_id"]).size().rename("ua_read_cnt").reset_index()
    df = df.merge(ua_ra, on=["user_id", "author_id"], how="left")
    df["ua_read_cnt"] = df["ua_read_cnt"].fillna(0).astype(np.int16)
    del ua_ra
    gc.collect()

    # Рейтинг юзера для данного автора
    rbr = reads_ba.dropna(subset=["rating"])
    if len(rbr) > 0:
        ura = rbr.groupby(["user_id", "author_id"])["rating"].agg(
            ua_mean_rat="mean", ua_max_rat="max"
        ).reset_index()
        df = df.merge(ura, on=["user_id", "author_id"], how="left")
        del ura
    else:
        df["ua_mean_rat"] = 0.0
        df["ua_max_rat"] = 0.0
    df["ua_mean_rat"] = df["ua_mean_rat"].fillna(0).astype(np.float32)
    df["ua_max_rat"] = df["ua_max_rat"].fillna(0).astype(np.float32)

    # Publisher stats
    up_agg = past_ba.groupby(["user_id", "publisher_id"]).size().rename("up_cnt").reset_index()
    df = df.merge(up_agg, on=["user_id", "publisher_id"], how="left")
    df["up_cnt"] = df["up_cnt"].fillna(0).astype(np.int16)
    del up_agg

    upr = reads_ba.groupby(["user_id", "publisher_id"]).size().rename("up_read_cnt").reset_index()
    df = df.merge(upr, on=["user_id", "publisher_id"], how="left")
    df["up_read_cnt"] = df["up_read_cnt"].fillna(0).astype(np.int16)
    del upr, reads_ba
    gc.collect()

    # Publisher share
    utm = past.groupby("user_id").size()
    upt = past_ba.groupby(["user_id", "publisher_id"]).size().rename("_upt").reset_index()
    upt["_ut"] = upt["user_id"].map(utm).fillna(1)
    upt["up_share"] = (upt["_upt"] / upt["_ut"]).astype(np.float32)
    df = df.merge(upt[["user_id", "publisher_id", "up_share"]],
                  on=["user_id", "publisher_id"], how="left")
    df["up_share"] = df["up_share"].fillna(0).astype(np.float32)
    del upt
    gc.collect()

    # ── Entity popularity (book, author) ──
    for entity, col in [("book", "book_id"), ("author", "author_id")]:
        ep = past_ba.groupby(col).size().rename(f"{entity}_pop").reset_index()
        df = df.merge(ep, on=col, how="left")
        df[f"{entity}_pop"] = df[f"{entity}_pop"].fillna(0).astype(np.int32)
        df[f"{entity}_pop_log"] = np.log1p(df[f"{entity}_pop"]).astype(np.float32)
        del ep

    # 30-дневная популярность
    p30ba = past30.merge(editions_item[["edition_id", "book_id", "author_id"]],
                         on="edition_id", how="left")
    for entity, col in [("book", "book_id"), ("author", "author_id")]:
        p30x = p30ba.groupby(col).size().rename(f"{entity}_pop_30").reset_index()
        df = df.merge(p30x, on=col, how="left")
        df[f"{entity}_pop_30"] = df[f"{entity}_pop_30"].fillna(0).astype(np.int32)
        df[f"{entity}_pop_30_log"] = np.log1p(df[f"{entity}_pop_30"]).astype(np.float32)
        del p30x
    del p30ba
    gc.collect()

    # 7-дневная популярность + тренд
    p7ba = past7.merge(editions_item[["edition_id", "book_id", "author_id"]],
                       on="edition_id", how="left")
    for entity, col in [("book", "book_id"), ("author", "author_id")]:
        p7x = p7ba.groupby(col).size().rename(f"{entity}_pop_7").reset_index()
        df = df.merge(p7x, on=col, how="left")
        df[f"{entity}_pop_7"] = df[f"{entity}_pop_7"].fillna(0).astype(np.int32)
        df[f"{entity}_pop_trend"] = (
            (df[f"{entity}_pop_7"] * 4.28 + 1) / (df[f"{entity}_pop_30"] + 1)
        ).astype(np.float32)
        del p7x
    del p7ba, past_ba
    gc.collect()

    # ── Candidate position ranks ──
    df = add_candidate_position_features(df)

    # ── Top-10 genre match (взвешенное совпадение с топ жанрами юзера) ──
    tmp = (past[["user_id", "edition_id", "event_type"]]
           .merge(ed_genre_long, on="edition_id", how="left")
           .dropna(subset=["genre_id"]))
    tmp["w"] = np.where(tmp["event_type"].to_numpy() == 2, 2.0, 1.0).astype(np.float32)
    ugw = tmp.groupby(["user_id", "genre_id"])["w"].sum().reset_index()
    ugw["rnk"] = ugw.groupby("user_id")["w"].rank(method="first", ascending=False)
    ug_top = ugw[ugw["rnk"] <= 10].drop(columns=["rnk"])
    del tmp, ugw
    gc.collect()

    match = pairs_ug.merge(ug_top, on=["user_id", "genre_id"], how="inner")
    ma = match.groupby(["user_id", "edition_id"]).agg(
        topg_match_cnt=("genre_id", "nunique"),
        topg_match_wsum=("w", "sum")
    ).reset_index()
    df = df.merge(ma, on=["user_id", "edition_id"], how="left")
    df["topg_match_cnt"] = df["topg_match_cnt"].fillna(0).astype(np.int16)
    df["topg_match_wsum"] = df["topg_match_wsum"].fillna(0).astype(np.float32)
    df["topg_match_frac"] = (
        df["topg_match_cnt"] / df["genre_cnt"].replace(0, np.nan)
    ).fillna(0).astype(np.float32)
    del match, ma, ug_top
    gc.collect()

    # ── Text user profile (time-weighted average of embeddings) ──
    ptxt = (past[["user_id", "edition_id", "event_type", "event_ts"]]
            .merge(text_emb[["edition_id"] + emb_cols], on="edition_id", how="left"))
    for c in emb_cols:
        ptxt[c] = ptxt[c].fillna(0).astype(np.float32)

    # Веса: time decay (60 дней) × event weight (read=2x)
    dd = (pd.Timestamp(ref_ts) - ptxt["event_ts"]).dt.total_seconds() / 86400.0
    wt = np.exp(-dd.to_numpy(np.float32) / 60.0).astype(np.float32)
    we = np.where(ptxt["event_type"].to_numpy() == 2, 2.0, 1.0).astype(np.float32)
    ptxt["w"] = (wt * we).astype(np.float32)

    for c in emb_cols:
        ptxt[c] = ptxt[c] * ptxt["w"]
    us = ptxt.groupby("user_id")[emb_cols].sum()
    ws = ptxt.groupby("user_id")["w"].sum().replace(0, np.nan)
    uv = us.div(ws, axis=0).fillna(0).reset_index()
    uv.columns = ["user_id"] + [f"user_{c}" for c in emb_cols]
    del ptxt, us, ws, wt, we, dd
    gc.collect()

    # Косинусная близость: text profile vs item
    df = df.merge(uv, on="user_id", how="left")
    uec = [f"user_{c}" for c in emb_cols]
    for c in uec:
        df[c] = df[c].fillna(0).astype(np.float32)

    im = df[emb_cols].to_numpy(np.float32)
    um = df[uec].to_numpy(np.float32)
    dot = np.sum(im * um, axis=1).astype(np.float32)
    inrm = np.sqrt(np.sum(im ** 2, axis=1)).astype(np.float32)
    unrm = np.sqrt(np.sum(um ** 2, axis=1)).astype(np.float32)

    df["text_dot"] = dot
    df["text_item_norm"] = inrm
    df["text_user_norm"] = unrm
    df["text_cos"] = (dot / (inrm * unrm + 1e-6)).astype(np.float32)
    df.drop(columns=uec, inplace=True)
    del uv, um, dot, unrm
    gc.collect()

    # ── Cosine similarity с последним прочитанным / wish ──
    for en, ec in [("lastread", 2), ("lastwish", 1)]:
        p = past[past["event_type"] == ec][["user_id", "edition_id", "event_ts"]].copy()
        if len(p) > 0:
            p = (p.sort_values(["user_id", "event_ts"])
                 .drop_duplicates("user_id", keep="last")[["user_id", "edition_id"]])
            p = (p.merge(text_emb[["edition_id"] + emb_cols], on="edition_id", how="left")
                 .drop(columns=["edition_id"]))
            p.columns = ["user_id"] + [f"{en}_{c}" for c in emb_cols]
        else:
            p = pd.DataFrame({"user_id": df["user_id"].unique()})
            for c in emb_cols:
                p[f"{en}_{c}"] = 0.0

        df = df.merge(p, on="user_id", how="left")
        cols = [f"{en}_{c}" for c in emb_cols]
        for c in cols:
            df[c] = df[c].fillna(0).astype(np.float32)

        mat = df[cols].to_numpy(np.float32)
        d = np.sum(im * mat, axis=1).astype(np.float32)
        n = np.sqrt(np.sum(mat ** 2, axis=1)).astype(np.float32)
        df[f"text_cos_{en}"] = (d / (inrm * n + 1e-6)).astype(np.float32)
        df.drop(columns=cols, inplace=True)
        del p, mat, d, n
        gc.collect()

    df.drop(columns=emb_cols, inplace=True)
    del im, inrm
    gc.collect()

    # ══════════════════════════════════════════════════════════════════
    #  CF-СКОРЫ (6 моделей коллаборативной фильтрации)
    # ══════════════════════════════════════════════════════════════════

    pair_key = df[["user_id", "edition_id"]]

    # ALS (основная)
    if add_als:
        df["als_score"] = als_score_pairs(past, pair_key, **(als_params or {})).astype(np.float32)
        df["als_user_z"] = (df.groupby("user_id")["als_score"]
                            .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
                            .astype(np.float32))
        df["als_rank"] = (df.groupby("user_id")["als_score"]
                          .rank(method="average", ascending=False)
                          .astype(np.float32))
    else:
        df["als_score"] = 0.0
        df["als_user_z"] = 0.0
        df["als_rank"] = 9999.0

    # ALS на свежих данных (последние N дней)
    if add_als_recent:
        cr = pd.Timestamp(ref_ts) - pd.Timedelta(days=int(als_recent_days))
        pr = past[past["event_ts"] >= cr].copy()
        if len(pr) < 5000:
            pr = past  # fallback на все данные если мало свежих
        df["als_r_score"] = als_score_pairs(pr, pair_key, **(als_params or {})).astype(np.float32)
        df["als_r_user_z"] = (df.groupby("user_id")["als_r_score"]
                              .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
                              .astype(np.float32))
        df["als_r_rank"] = (df.groupby("user_id")["als_r_score"]
                            .rank(method="average", ascending=False)
                            .astype(np.float32))
        del pr
    else:
        df["als_r_score"] = 0.0
        df["als_r_user_z"] = 0.0
        df["als_r_rank"] = 9999.0

    # BPR
    if add_bpr:
        df["bpr_score"] = bpr_score_pairs(past, pair_key, **(bpr_params or {})).astype(np.float32)
        df["bpr_user_z"] = (df.groupby("user_id")["bpr_score"]
                            .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
                            .astype(np.float32))
    else:
        df["bpr_score"] = 0.0
        df["bpr_user_z"] = 0.0

    # ALS-2 (облегчённая)
    if add_als2:
        df["als2_score"] = als2_score_pairs(past, pair_key).astype(np.float32)
        df["als2_user_z"] = (df.groupby("user_id")["als2_score"]
                             .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
                             .astype(np.float32))
    else:
        df["als2_score"] = 0.0
        df["als2_user_z"] = 0.0

    # ALS с time-decay
    if add_als_td:
        df["als_td_score"] = als_timedecay_score_pairs(past, pair_key, ref_ts).astype(np.float32)
        df["als_td_user_z"] = (df.groupby("user_id")["als_td_score"]
                               .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
                               .astype(np.float32))
    else:
        df["als_td_score"] = 0.0
        df["als_td_user_z"] = 0.0

    # SVD-CF
    if add_svd_cf:
        df["svd_cf_score"] = svd_collab_score_pairs(
            past, pair_key, n_components=32, verbose=verbose
        ).astype(np.float32)
        df["svd_cf_user_z"] = (df.groupby("user_id")["svd_cf_score"]
                               .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
                               .astype(np.float32))
    else:
        df["svd_cf_score"] = 0.0
        df["svd_cf_user_z"] = 0.0

    # ── Item2Vec scores ──
    if i2v_model is not None:
        cm, cx = item2vec_user_scores(i2v_model, past, pair_key, top_k=20)
        df["i2v_cos_mean"] = cm
        df["i2v_cos_max"] = cx
        df["i2v_cos_last_read"] = item2vec_last_item_cos(
            i2v_model, past, pair_key, event_type_filter=2)
        df["i2v_cos_last_wish"] = item2vec_last_item_cos(
            i2v_model, past, pair_key, event_type_filter=1)
    else:
        df["i2v_cos_mean"] = 0.0
        df["i2v_cos_max"] = 0.0
        df["i2v_cos_last_read"] = 0.0
        df["i2v_cos_last_wish"] = 0.0

    # Финальная типизация
    for c in ["book_id", "author_id", "publisher_id", "publication_year", "language_id"]:
        df[c] = df[c].fillna(-1).astype(np.int32)

    # ══════════════════════════════════════════════════════════════════
    #  CF AGREEMENT (согласованность между CF-моделями)
    # ══════════════════════════════════════════════════════════════════

    cf_z_cols = [c for c in ["als_user_z", "als_r_user_z", "bpr_user_z",
                             "als2_user_z", "als_td_user_z", "svd_cf_user_z"]
                 if c in df.columns]
    if len(cf_z_cols) >= 2:
        cf_z = df[cf_z_cols].to_numpy(np.float32)
        df["cf_mean_z"] = np.nanmean(cf_z, axis=1).astype(np.float32)
        df["cf_std_z"] = np.nanstd(cf_z, axis=1).astype(np.float32)   # высокая → модели не согласны
        df["cf_max_z"] = np.nanmax(cf_z, axis=1).astype(np.float32)
    else:
        df["cf_mean_z"] = 0.0
        df["cf_std_z"] = 0.0
        df["cf_max_z"] = 0.0

    # ══════════════════════════════════════════════════════════════════
    #  PER-USER RANKS (нормализованные ранги внутри юзера)
    # ══════════════════════════════════════════════════════════════════

    rank_src = [
        "als_score", "als_r_score", "text_cos", "topg_match_wsum",
        "i_pop_log", "i_pop_7_log", "author_pop_log", "i_mean_rating",
        "text_cos_lastread", "text_cos_lastwish", "ua_cnt",
        "i2v_cos_mean", "i2v_cos_max", "bpr_score",
        "i2v_cos_last_read", "up_cnt", "als2_score",
        "u_lang_share", "lang_is_top",
        "genre_tfidf_sum", "genre_tfidf_max",
        "genre_recency_sum", "genre_recency_max",
        "genre_jaccard", "te_pub_read_rate", "i_conversion",
        "author_loyalty", "genre_loyalty_max", "item_conversion_rate",
        "weighted_genre_match", "i_pop_tw7_log", "i_pop_tw30_log",
        "als_td_score", "svd_cf_score",
    ]

    for col in rank_src:
        if col not in df.columns:
            df[col] = 0.0
        rk = df.groupby("user_id")[col].rank(method="min", ascending=False).astype(np.float32)
        mx = df.groupby("user_id")[col].transform("size").astype(np.float32)
        df[f"{col}_rnk"] = (rk / mx).astype(np.float32)

    # Агрегированные ранги
    rnk_cols = [f"{c}_rnk" for c in rank_src]
    df["mean_rnk"] = df[rnk_cols].mean(axis=1).astype(np.float32)
    df["min_rnk"] = df[rnk_cols].min(axis=1).astype(np.float32)

    # ══════════════════════════════════════════════════════════════════
    #  КРОСС-ПРИЗНАКИ (взаимодействия между группами фичей)
    # ══════════════════════════════════════════════════════════════════

    df["als_x_textcos"] = (df["als_score"] * df["text_cos"]).astype(np.float32)
    df["als_x_genre"] = (df["als_score"] * df["topg_match_frac"]).astype(np.float32)
    df["pop_x_genre"] = (df["i_pop_log"] * df["topg_match_frac"]).astype(np.float32)
    df["i2v_x_textcos"] = (df["i2v_cos_mean"] * df["text_cos"]).astype(np.float32)
    df["bpr_x_als"] = (df["bpr_score"] * df["als_score"]).astype(np.float32)
    df["als2_x_als"] = (df["als2_score"] * df["als_score"]).astype(np.float32)
    df["als2_x_textcos"] = (df["als2_score"] * df["text_cos"]).astype(np.float32)
    df["als_x_novelty"] = (df["als_score"] * df["genre_novelty"]).astype(np.float32)
    df["textcos_x_novelty"] = (df["text_cos"] * df["genre_novelty"]).astype(np.float32)
    df["textcos_x_langshare"] = (df["text_cos"] * df["u_lang_share"]).astype(np.float32)
    df["als_x_langshare"] = (df["als_score"] * df["u_lang_share"]).astype(np.float32)
    df["gtfidf_x_als"] = (df["genre_tfidf_sum"] * df["als_score"]).astype(np.float32)
    df["gtfidf_x_textcos"] = (df["genre_tfidf_sum"] * df["text_cos"]).astype(np.float32)
    df["genre_recency_x_als"] = (df["genre_recency_sum"] * df["als_score"]).astype(np.float32)
    df["genre_recency_x_textcos"] = (df["genre_recency_sum"] * df["text_cos"]).astype(np.float32)
    df["jaccard_x_als"] = (df["genre_jaccard"] * df["als_score"]).astype(np.float32)
    df["jaccard_x_textcos"] = (df["genre_jaccard"] * df["text_cos"]).astype(np.float32)
    df["conversion_x_als"] = (df["i_conversion"] * df["als_score"]).astype(np.float32)
    df["loyalty_x_als"] = (df["ua_loyalty"] * df["als_score"]).astype(np.float32)
    df["bpr_x_i2v"] = (df["bpr_score"] * df["i2v_cos_mean"]).astype(np.float32)
    df["als_x_conversion"] = (df["als_score"] * df["i_conversion"]).astype(np.float32)
    df["conv_x_pop"] = (df["item_conversion_rate"] * df["i_pop_log"]).astype(np.float32)
    df["wgm_x_als"] = (df["weighted_genre_match"] * df["als_score"]).astype(np.float32)
    df["als_td_x_als"] = (df["als_td_score"] * df["als_score"]).astype(np.float32)
    df["svd_x_als"] = (df["svd_cf_score"] * df["als_score"]).astype(np.float32)
    df["svd_x_textcos"] = (df["svd_cf_score"] * df["text_cos"]).astype(np.float32)

    if verbose:
        print(f"[{now()}][FEATS] rows={len(df):,} cols={len(df.columns)} "
              f"built in {time.time() - t0:.1f}s")
    gc.collect()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  СБОРКА ОБУЧАЮЩЕГО ДАТАСЕТА (ОДНО ВРЕМЕННОЕ ОКНО)
# ══════════════════════════════════════════════════════════════════════════════

def build_window_dataset(cutoff, interactions, candidates, editions_item, u_demo,
                         ed_genre_long, cand_ug, text_emb, text_dim=24,
                         add_als=False, als_params=None,
                         add_als_recent=False, als_recent_days=90,
                         add_bpr=False, bpr_params=None,
                         add_als2=False, add_als_td=False, add_svd_cf=False,
                         i2v_model=None, neg_per_user=120, verbose=True):
    """
    Строит обучающий/валидационный датасет для одного временного окна.

    Алгоритм:
      1. past = interactions до cutoff
      2. future = interactions в [cutoff, cutoff + 30 дней)
      3. Создаём метки из future: label 3 (read), 1 (wish), 0 (no event)
      4. Объединяем кандидатов с метками, добавляем «extra positives»
         (позитивные пары не в candidates)
      5. Генерируем все признаки
      6. Даунсэмплируем негативы (до neg_per_user на юзера)

    Returns:
        DataFrame с признаками + label, label2, label_mc
    """
    fut_end = cutoff + pd.Timedelta(days=30)
    past = interactions[interactions["event_ts"] < cutoff].copy()
    future = interactions[
        (interactions["event_ts"] >= cutoff) &
        (interactions["event_ts"] < fut_end)
    ].copy()

    # Создаём метки и оставляем только позитивные
    fut_lab = make_label_from_future(future)
    fut_lab = fut_lab[fut_lab["label"] > 0]

    if verbose:
        print(f"\n[{now()}][WIN] cutoff={cutoff} past={len(past):,} "
              f"future={len(future):,} pos={len(fut_lab):,}")

    # Объединяем кандидатов с метками
    pairs = candidates.merge(fut_lab, on=["user_id", "edition_id"], how="left")
    pairs["label"] = pairs["label"].fillna(0).astype(np.int8)

    # Extra positives: позитивные пары, которых нет в candidates
    extra = fut_lab.merge(candidates, on=["user_id", "edition_id"],
                          how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"][["user_id", "edition_id", "label"]]
    if verbose:
        print(f"[{now()}][WIN] extra positives: {len(extra):,}")

    pairs = pd.concat([pairs, extra], ignore_index=True)

    # Жанры для extra positives
    eug = (extra[["user_id", "edition_id"]]
           .merge(ed_genre_long, on="edition_id", how="left")
           .dropna(subset=["genre_id"]))
    eug["genre_id"] = eug["genre_id"].astype(np.int32)
    pug = pd.concat(
        [cand_ug[["user_id", "edition_id", "genre_id"]],
         eug[["user_id", "edition_id", "genre_id"]]],
        ignore_index=True
    )

    # Item2Vec: обучаем на past если не передана
    wi2v = (i2v_model if i2v_model is not None
            else build_item2vec_model(past, dim=64, window=10, epochs=15, verbose=verbose))

    # Генерация признаков
    X = build_features(
        pairs[["user_id", "edition_id"]], past, editions_item, u_demo, cutoff,
        ed_genre_long, pug, text_emb, text_dim,
        add_als=add_als, als_params=als_params,
        add_als_recent=add_als_recent, als_recent_days=als_recent_days,
        add_bpr=add_bpr, bpr_params=bpr_params,
        add_als2=add_als2, add_als_td=add_als_td, add_svd_cf=add_svd_cf,
        i2v_model=wi2v, verbose=False
    )

    # Формируем метки в разных форматах
    y = pairs["label"].to_numpy(np.int8)
    X["label"] = y
    X["label2"] = np.where(y == 3, 6, np.where(y == 1, 2, 0)).astype(np.int8)  # для ранкера
    X["label_mc"] = ((y == 1).astype(np.int8) + 2 * (y == 3).astype(np.int8)).astype(np.int8)  # 0/1/2

    if verbose:
        dist = X["label"].value_counts().to_dict()
        print(f"[{now()}][WIN] label dist: {dist} "
              f"pos_rate={(X['label'] > 0).mean():.4f}")

    # Даунсэмплинг негативов
    X = downsample_negatives_per_user(X, neg_per_user=neg_per_user, verbose=verbose)

    del past, future, fut_lab, extra, eug, pug, pairs
    gc.collect()
    return X


# ══════════════════════════════════════════════════════════════════════════════
#  ГЛАВНАЯ ФУНКЦИЯ — ПОЛНЫЙ ПАЙПЛАЙН
# ══════════════════════════════════════════════════════════════════════════════

def main(data_dir, submit_dir, out_path="submission.csv", use_gpu=True,
         neg_per_user=120, windows_days_back=(7, 10, 15, 20, 25),
         seeds=(42,), topM=200, text_dim=24,
         use_als=True, use_als_recent=True, als_recent_days=90,
         use_bpr=True, use_als2=True, use_als_td=True, use_svd_cf=True,
         verbose=True):
    """
    Полный пайплайн: загрузка → обучение → инференс → submission.

    Этапы:
    ┌───────────────────────────────────────────────────────────────────┐
    │  1. Загрузка и предобработка данных                               │
    │  2. Построение текстовых эмбеддингов (TF-IDF + SVD)               │
    │  3. Обучение глобальной Item2Vec                                  │
    │  4. Подготовка служебных структур (жанры, маппинги)               │
    │  5. Построение валидационного датасета (cutoff = max_ts - 7d)     │
    │  6. Обучение на 4 временных окнах:                                │
    │     - CatBoostRanker (YetiRank, 8000 итераций)                    │
    │     - CatBoostClassifier (MultiClass, 3000 итераций)              │
    │  7. Инференс на тестовых кандидатах                               │
    │  8. Калибровка вероятностей                                       │
    │  9. Diversity-aware reranking → top-20 на юзера                   │
    │  10. Сохранение submission.csv                                    │
    └───────────────────────────────────────────────────────────────────┘
    """
    t_all = time.time()

    # ── 1. Загрузка данных ──
    users = pd.read_csv(os.path.join(data_dir, "users.csv"))
    interactions = pd.read_csv(os.path.join(data_dir, "interactions.csv"))
    editions = pd.read_csv(os.path.join(data_dir, "editions.csv"))
    authors = pd.read_csv(os.path.join(data_dir, "authors.csv"))
    book_genres = pd.read_csv(os.path.join(data_dir, "book_genres.csv"))
    candidates = pd.read_csv(os.path.join(submit_dir, "candidates.csv"))

    interactions["event_ts"] = pd.to_datetime(interactions["event_ts"])
    max_ts = interactions["event_ts"].max()
    print(f"[{now()}][LOAD] interactions={interactions.shape} "
          f"candidates={candidates.shape} max_ts={max_ts}")

    # ── 2. Предобработка ──
    editions_item = build_editions_item(editions, book_genres)
    u_demo = precompute_user_demo(users)

    text_cache = f"text_svd_dim{text_dim}_mf35000_combined_v2.parquet"
    text_emb = build_text_embeddings(
        editions, authors, text_dim=text_dim,
        cache_path=text_cache, verbose=verbose
    )
    del editions, authors
    gc.collect()

    # ── 3. Item2Vec (глобальная, для инференса) ──
    print(f"\n[{now()}][I2V] Building global Item2Vec for inference...")
    i2v_global = build_item2vec_model(
        interactions, dim=64, window=10, epochs=15, verbose=verbose
    )

    # ── 4. Служебные структуры ──
    genres_by_edition = {
        eid: set(gl) for eid, gl
        in zip(editions_item["edition_id"], editions_item["genre_list"])
    }
    author_by_edition = dict(
        zip(editions_item["edition_id"], editions_item["author_id"])
    )

    # Длинная таблица жанров (edition_id → genre_id)
    ed_genre_long = (
        editions_item[["edition_id", "book_id"]]
        .merge(book_genres, on="book_id", how="left")[["edition_id", "genre_id"]]
    )
    ed_genre_long = ed_genre_long.dropna(subset=["genre_id"])
    ed_genre_long["genre_id"] = ed_genre_long["genre_id"].astype(np.int32)

    # Жанры кандидатов
    cand_ug = (candidates
               .merge(ed_genre_long, on="edition_id", how="left")
               .dropna(subset=["genre_id"]))
    cand_ug["genre_id"] = cand_ug["genre_id"].astype(np.int32)

    # ── 5. Временные окна ──
    cutoffs = sorted(
        [max_ts - pd.Timedelta(days=d) for d in windows_days_back],
        reverse=True
    )
    valid_cutoff = cutoffs[0]    # самый свежий — валидация
    train_cutoffs = cutoffs[1:]  # остальные — обучение

    # Гиперпараметры CF-моделей
    als_params = dict(factors=96, iters=18, reg=0.05, alpha=15.0)
    bpr_params = dict(factors=64, iters=100, reg=0.01, lr=0.05)
    cf_kwargs = dict(
        add_als=use_als, als_params=als_params,
        add_als_recent=use_als_recent, als_recent_days=als_recent_days,
        add_bpr=use_bpr, bpr_params=bpr_params,
        add_als2=use_als2, add_als_td=use_als_td, add_svd_cf=use_svd_cf
    )

    # ── 6. Валидационный датасет ──
    print(f"\n[{now()}][VAL] cutoff={valid_cutoff}")
    val_df = build_window_dataset(
        valid_cutoff, interactions, candidates, editions_item, u_demo,
        ed_genre_long, cand_ug, text_emb, text_dim,
        **cf_kwargs, i2v_model=None, neg_per_user=neg_per_user, verbose=True
    )
    val_df = ensure_grouped(val_df)

    # Определяем колонки признаков
    cat_cols = ["user_id", "edition_id", "book_id", "author_id", "publisher_id",
                "age_restriction", "gender", "age_bucket", "language_id"]
    exclude = {"label", "label2", "label_mc"}
    all_cols = set(val_df.columns) - exclude
    cat_cols = [c for c in cat_cols if c in all_cols]
    feat_cols = cat_cols + sorted([c for c in all_cols if c not in cat_cols])
    print(f"[{now()}] Feature count: {len(feat_cols)} (cat={len(cat_cols)})")

    # CatBoost Pools для валидации
    va_pool = Pool(val_df[feat_cols], label=val_df["label2"],
                   group_id=val_df["user_id"], cat_features=cat_cols)
    val_clf_pool = Pool(val_df[feat_cols], label=val_df["label_mc"],
                        cat_features=cat_cols)
    del val_df
    gc.collect()

    # ── 7. Обучение моделей ──
    cb_models = []      # CatBoostRanker'ы
    cb_clf_models = []  # CatBoostClassifier'ы

    for c in train_cutoffs:
        print(f"\n[{now()}][TRAIN] cutoff={c}")
        train_df = build_window_dataset(
            c, interactions, candidates, editions_item, u_demo,
            ed_genre_long, cand_ug, text_emb, text_dim,
            **cf_kwargs, i2v_model=None, neg_per_user=neg_per_user, verbose=True
        )
        train_df = ensure_grouped(train_df)

        # Добавляем отсутствующие колонки (если вдруг)
        for fc in feat_cols:
            if fc not in train_df.columns:
                train_df[fc] = 0

        tr_pool = Pool(train_df[feat_cols], label=train_df["label2"],
                       group_id=train_df["user_id"], cat_features=cat_cols)
        tr_clf_pool = Pool(train_df[feat_cols], label=train_df["label_mc"],
                           cat_features=cat_cols)
        del train_df
        gc.collect()

        for s in seeds:
            # ── Ranker: CatBoostRanker с YetiRank ──
            ranker = CatBoostRanker(
                loss_function="YetiRank",
                eval_metric="PairLogit",
                metric_period=100,
                depth=8,
                iterations=8000,
                learning_rate=0.05,
                l2_leaf_reg=14.0,
                bootstrap_type="Bernoulli",
                subsample=0.8,
                random_strength=2.0,
                random_seed=s,
                task_type=("GPU" if use_gpu else "CPU"),
                od_type="Iter",
                od_wait=400,
                verbose=200,
                gpu_ram_part=0.80,
                allow_writing_files=False
            )
            print(f"\n[{now()}][CB-RANK] fit cutoff={c} seed={s}")
            ranker.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            cb_models.append(ranker)

            # ── Classifier: CatBoostClassifier (3 класса) ──
            clf = CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="MultiClass",
                class_weights=[1.0, 6.0, 8.0],  # upweight wish и read
                depth=8,
                iterations=3000,
                learning_rate=0.05,
                l2_leaf_reg=10.0,
                random_strength=2.0,
                bootstrap_type="Bernoulli",
                subsample=0.8,
                random_seed=s,
                task_type=("GPU" if use_gpu else "CPU"),
                od_type="Iter",
                od_wait=1000,
                verbose=200,
                allow_writing_files=False
            )
            print(f"\n[{now()}][CB-CLF] fit cutoff={c} seed={s}")
            clf.fit(tr_clf_pool, eval_set=val_clf_pool, use_best_model=True)
            cb_clf_models.append(clf)

        del tr_pool, tr_clf_pool
        gc.collect()

    del va_pool, val_clf_pool
    gc.collect()

    # ── Feature importance (для анализа) ──
    print_feature_importance(cb_models, feat_cols, title="RANKER")
    print_feature_importance(cb_clf_models, feat_cols, title="CLASSIFIER")

    # ══════════════════════════════════════════════════════════════════
    #  8. ИНФЕРЕНС НА ТЕСТОВЫХ КАНДИДАТАХ
    # ══════════════════════════════════════════════════════════════════

    print(f"\n[{now()}][INFER] building features...")
    infer_ref_ts = max_ts + pd.Timedelta(seconds=1)

    infer_X = build_features(
        candidates[["user_id", "edition_id"]],
        interactions, editions_item, u_demo, infer_ref_ts,
        ed_genre_long, cand_ug[["user_id", "edition_id", "genre_id"]],
        text_emb, text_dim,
        **cf_kwargs, i2v_model=i2v_global, verbose=True
    )
    del i2v_global
    gc.collect()

    # Добавляем отсутствующие колонки
    for fc in feat_cols:
        if fc not in infer_X.columns:
            infer_X[fc] = 0

    infer_pool_rank = Pool(infer_X[feat_cols], group_id=infer_X["user_id"],
                           cat_features=cat_cols)
    infer_pool_clf = Pool(infer_X[feat_cols], cat_features=cat_cols)

    # ── Предсказания ранкера (ансамбль) ──
    print(f"[{now()}][INFER] predict ranker ensemble ({len(cb_models)} models)...")
    pred_rank = np.mean(
        [m.predict(infer_pool_rank).astype(np.float32) for m in cb_models],
        axis=0
    )
    infer_X["pred"] = pred_rank.astype(np.float32)
    infer_X["base_score"] = infer_X["pred"].astype(np.float32)
    infer_X["base_z"] = (
        infer_X.groupby("user_id")["base_score"]
        .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
        .astype(np.float32)
    )

    # ── Предсказания классификатора (ансамбль) ──
    print(f"[{now()}][INFER] predict classifier ensemble ({len(cb_clf_models)} models)...")
    proba = np.mean(
        [cm.predict_proba(infer_pool_clf).astype(np.float32) for cm in cb_clf_models],
        axis=0
    )
    infer_X["p_wish_raw"] = proba[:, 1].astype(np.float32)  # P(class=1: wish)
    infer_X["p_read_raw"] = proba[:, 2].astype(np.float32)  # P(class=2: read)

    # ── 9. Калибровка ──
    infer_X = per_user_topk_mass_calibration(
        infer_X, topK=50, desired_psum_topK=3.5, T=0.9, max_scale=50.0
    )

    # Оставляем только нужные колонки для reranking'а
    infer_small = infer_X[[
        "user_id", "edition_id", "book_id",
        "base_score", "base_z", "p_read", "p_wish", "p_rel"
    ]].copy()
    del infer_X, infer_pool_rank, infer_pool_clf, pred_rank, proba
    gc.collect()

    # Диагностика калибровки
    ps = infer_small.groupby("user_id")["p_rel"].sum()
    print(f"[{now()}][DIAG] p_sum quantiles: "
          f"{ps.quantile([0.1, 0.5, 0.9]).to_dict()}")

    # ══════════════════════════════════════════════════════════════════
    #  10. DIVERSITY-AWARE RERANKING → TOP-20
    # ══════════════════════════════════════════════════════════════════

    print(f"[{now()}][RERANK] adaptive-lock rerank (topM={topM})")
    recs = []

    for uid, u_df in infer_small.groupby("user_id", sort=False):
        u_df = u_df.sort_values("base_score", ascending=False).head(int(topM))

        top20 = expected_rerank_top20(
            u_df,
            genres_by_edition=genres_by_edition,
            author_by_edition=author_by_edition,
            forbid_same_book=True,
            topM=topM,
            lock_n=8,
            lambda_base=0.025,
            beta=0.5,
            alpha_ndcg=0.7,
            tail_div_boost_start=1.5,
            tail_div_boost_end=3.0,
            max_per_author_head=3,
            max_per_author_tail=1
        )

        for r, eid in enumerate(top20, start=1):
            recs.append((uid, eid, r))

    # ── Формирование и валидация submission ──
    sub = pd.DataFrame(recs, columns=["user_id", "edition_id", "rank"])

    # Проверки корректности
    assert sub.groupby("user_id").size().min() == 20, "Не у всех юзеров 20 рекомендаций"
    assert sub.groupby("user_id").size().max() == 20, "У кого-то больше 20 рекомендаций"
    assert sub.duplicated(["user_id", "edition_id"]).sum() == 0, "Дубликаты edition в юзере"
    assert sub.duplicated(["user_id", "rank"]).sum() == 0, "Дубликаты рангов в юзере"

    sub.to_csv(out_path, index=False)
    print(f"[{now()}][DONE] saved {out_path} shape={sub.shape} total={(time.time()-t_all):.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--submit_dir", type=str, required=True)
    parser.add_argument("--out_path", type=str, default="submission.csv")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--neg_per_user", type=int, default=120)
    parser.add_argument("--text_dim", type=int, default=24)

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        submit_dir=args.submit_dir,
        out_path=args.out_path,
        use_gpu=args.use_gpu,
        neg_per_user=args.neg_per_user,
        windows_days_back=(7, 10, 15, 20, 25),
        seeds=(42,),
        topM=200,
        text_dim=args.text_dim,
        use_als=True,
        use_als_recent=True,
        als_recent_days=90,
        use_bpr=True,
        use_als2=True,
        use_als_td=True,
        use_svd_cf=True,
        verbose=True
    )