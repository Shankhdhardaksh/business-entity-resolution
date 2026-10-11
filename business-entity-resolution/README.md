# Business Entity Resolution

Matching noisy business records across three data sources at the scale of 26M records, built for the **Amazon ML Challenge 2026** by team Black Box.

Given a record in Source 1, find every record in Source 2 and Source 3 that refers to the same real-world business. The records span the US, India (9 Indic scripts) and France, and contain legal-suffix variants, typos, aliases ("dba") and missing addresses.

## Results

| Metric | Value |
|---|---|
| Blocking recall (train) | 73.3% micro, 73.2% macro, 40 candidates per entity |
| Validation AUC | 0.999542 |
| Validation macro F0.5 | 0.8342 (threshold 0.83) |
| Test S1 entities scored | 1,732,544 |
| Candidate pairs scored | 69,296,252 |
| Official format validator | PASS |

The metric is macro **F0.5**, which weights precision twice as much as recall.

## Pipeline

```
raw TSVs
  -> src/preprocess.py                         normalize names/addresses, extract fields (Parquet)
  -> src/tf_idf_faiss.ipynb                    candidate generation (blocking)
  -> src/feature_engineering_and_training.py   18 pairwise features, LightGBM, threshold tuning
  -> src/predict_test.py                       score test pairs, write the two submission TSVs
```

1. **Preprocessing:** lowercases and transliterates names (including Indic scripts), canonicalizes legal suffixes, extracts aliases/handles/web names, builds a phonetic consonant skeleton, maps states to codes and extracts address numbers.
2. **Blocking:** per (country, state) partition, character 3-5 gram TF-IDF on a weighted name + phonetic + address string, compressed to 128 dimensions with TruncatedSVD, then exact top-40 cosine search with FAISS.
3. **Matching:** 18 pairwise features per candidate pair feed a LightGBM classifier. A probability threshold tuned for macro F0.5 decides the final matches.

**Features:** fuzzy name similarity (Levenshtein, token sort, token set, partial ratio, phonetic ratio), fuzzy address similarity, address-number exact match and overlap, state/country match, name-kind match, Indic-script flag, name length ratio, and the blocking score.

## Design choices

- Train/validation split by S1 entity, not by row, to avoid leakage.
- Negatives undersampled to about 4:1 to handle class imbalance.
- Threshold swept from 0.30 to 0.95 on held-out entities to maximize macro F0.5.
- Batched processing (1M pairs per batch) with fuzzy features computed in parallel worker processes, and test predictions written to disk batch by batch, so 69M pairs fit in laptop memory.
- Tested merging a second blocker (IDF token overlap) for more recall. Macro F0.5 dropped from about 0.83 to about 0.79 because the extra candidates had no blocking score, so the final submission uses the single FAISS blocker. With a precision-weighted metric, more recall is not automatically better.

## Running it

The competition data is not included. The scripts use relative paths set as constants at the top of each file, so adjust them to match where you place the data.

```bash
pip install -r requirements.txt

python src/preprocess.py <raw_dir> <cleaned_dir>     # once for train, once for test
# run src/tf_idf_faiss.ipynb                          writes blocking_person_a_train.pkl / blocking_person_a_test.pkl
python src/feature_engineering_and_training.py        writes output_features/lgb_model.txt and best_threshold.txt
python src/predict_test.py                            writes output/matching_results.tsv and candidate_pairs.tsv
```

`feature_engineering_and_training.py` and `predict_test.py` expect, relative to where you run them:

```
dataset/train/train_source{1,2,3}.parquet, train_ground_truth.tsv
dataset/test/test_source{1,2,3}.parquet
blocking_person_a_train.pkl, blocking_person_a_test.pkl
```

The notebook reads the cleaned Parquet files from `DATA_DIR` at the top of its first cell.

## Tech stack

Python, Polars, scikit-learn, FAISS, LightGBM, RapidFuzz, NumPy, pandas, anyascii.

## Team

| Member | Contribution |
|---|---|
| Yash Patil | Text preprocessing and field normalization (`preprocess.py`) |
| Yashvi Garg | TF-IDF + SVD + FAISS blocking (`tf_idf_faiss.ipynb`) |
| Saloni Sinha | IDF token-overlap blocking (tested, not used in the final submission) |
| Daksh Shankhdhar | Feature engineering, model training, test inference, validation (`feature_engineering_and_training.py`, `predict_test.py`) |
