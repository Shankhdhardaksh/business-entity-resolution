import os
import gc
import math
import pickle
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from concurrent.futures import ProcessPoolExecutor
import lightgbm as lgb

DATA_DIR = "dataset"

TRAIN_S1 = os.path.join(DATA_DIR, "train", "train_source1.parquet")
TRAIN_S2 = os.path.join(DATA_DIR, "train", "train_source2.parquet")
TRAIN_S3 = os.path.join(DATA_DIR, "train", "train_source3.parquet")
TRAIN_GT = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv")

CANDIDATE_PKLS_TRAIN = ["blocking_person_a_train.pkl"]

OUT_DIR = "output_features"

NEG_PER_POS = 4
VAL_FRACTION = 0.15
N_WORKERS = max(1, os.cpu_count() - 1)
FUZZY_CHUNK_SIZE = 200_000
BATCH_SIZE = 1_000_000
RANDOM_SEED = 42

FIELDS = ["entity_id", "name_core", "name_phon", "addr_clean", "addr_nums",
          "state", "country", "name_kind", "is_indic", "addr_empty"]

FUZZY_COLS = [
    "name_ratio", "name_token_sort", "name_token_set", "name_partial_ratio",
    "name_phon_ratio", "addr_token_sort", "addr_token_set",
]

FEATURE_COLS = [
    "blocking_score", "is_source2",
    "state_match", "country_match", "both_addr_empty", "one_addr_empty",
    "name_kind_match", "either_is_indic", "name_length_ratio",
    "addr_num_exact_match", "addr_num_overlap",
] + FUZZY_COLS

KEEP_COLS = ["source1_entity_id", "candidate_entity_id", "label"] + FEATURE_COLS


def dict_to_polars(cand_dict, batch_size=2_000_000):
    frames = []
    ids_buf, cids_buf, scores_buf = [], [], []
    count = 0
    for s1_id, cands in cand_dict.items():
        for cid, score in cands:
            ids_buf.append(s1_id)
            cids_buf.append(cid)
            scores_buf.append(score)
            count += 1
            if count >= batch_size:
                frames.append(pl.DataFrame({
                    "source1_entity_id": ids_buf, "candidate_entity_id": cids_buf, "blocking_score": scores_buf,
                }))
                ids_buf, cids_buf, scores_buf = [], [], []
                count = 0
    if ids_buf:
        frames.append(pl.DataFrame({
            "source1_entity_id": ids_buf, "candidate_entity_id": cids_buf, "blocking_score": scores_buf,
        }))
    if not frames:
        return pl.DataFrame(schema={"source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8, "blocking_score": pl.Float64})
    return pl.concat(frames, how="vertical_relaxed")


def load_and_merge_candidates(pkl_paths):
    frames = []
    for p in pkl_paths:
        if not os.path.exists(p):
            print(f"  ⚠️  candidate file not found, skipping: {p}")
            continue
        print(f"  loading {p} ...")
        with open(p, "rb") as f:
            d = pickle.load(f)
        print(f"    converting {len(d):,} entities to table (batched)...")
        frame = dict_to_polars(d)
        del d
        gc.collect()
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"None of these candidate files exist: {pkl_paths}")
    combined = frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical_relaxed")
    del frames
    gc.collect()
    if len(pkl_paths) > 1:
        combined = combined.group_by(["source1_entity_id", "candidate_entity_id"]).agg(pl.col("blocking_score").max())
    return combined


def add_vectorized_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("s1_state").fill_null("") == pl.col("s2_state").fill_null("")).cast(pl.Int8).alias("state_match"),
        (pl.col("s1_country").fill_null("") == pl.col("s2_country").fill_null("")).cast(pl.Int8).alias("country_match"),
        (pl.col("s1_addr_empty").fill_null(False) & pl.col("s2_addr_empty").fill_null(False)).cast(pl.Int8).alias("both_addr_empty"),
        (pl.col("s1_addr_empty").fill_null(False) != pl.col("s2_addr_empty").fill_null(False)).cast(pl.Int8).alias("one_addr_empty"),
        (pl.col("s1_name_kind").fill_null("") == pl.col("s2_name_kind").fill_null("")).cast(pl.Int8).alias("name_kind_match"),
        (pl.col("s1_is_indic").fill_null(False) | pl.col("s2_is_indic").fill_null(False)).cast(pl.Int8).alias("either_is_indic"),
        pl.col("s1_name_core").fill_null("").str.len_chars().alias("_len1"),
        pl.col("s2_name_core").fill_null("").str.len_chars().alias("_len2"),
        pl.col("candidate_entity_id").str.starts_with("S2-").cast(pl.Int8).alias("is_source2"),
    )
    df = df.with_columns(
        (pl.min_horizontal("_len1", "_len2") / pl.max_horizontal("_len1", "_len2").clip(lower_bound=1))
        .fill_nan(0.0).alias("name_length_ratio")
    )
    df = df.with_columns(
        pl.col("s1_addr_nums").fill_null("").alias("_an1"),
        pl.col("s2_addr_nums").fill_null("").alias("_an2"),
    )
    df = df.with_columns(
        ((pl.col("_an1") == pl.col("_an2")) & (pl.col("_an1") != "") & (pl.col("_an2") != ""))
        .cast(pl.Int8).alias("addr_num_exact_match")
    )
    s1_sets = df["_an1"].str.split(" ").to_list()
    s2_sets = df["_an2"].str.split(" ").to_list()

    def jaccard_list(a, b):
        sa, sb = set(a) - {""}, set(b) - {""}
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    overlap = [jaccard_list(a, b) for a, b in zip(s1_sets, s2_sets)]
    df = df.with_columns(pl.Series("addr_num_overlap", overlap, dtype=pl.Float32))
    return df.drop(["_len1", "_len2", "_an1", "_an2"])


def _fuzzy_chunk(args):
    names1, names2, phons1, phons2, addrs1, addrs2 = args
    n = len(names1)
    out = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):
        n1, n2 = names1[i] or "", names2[i] or ""
        p1, p2 = phons1[i] or "", phons2[i] or ""
        a1, a2 = addrs1[i] or "", addrs2[i] or ""
        out[i, 0] = fuzz.ratio(n1, n2) / 100.0
        out[i, 1] = fuzz.token_sort_ratio(n1, n2) / 100.0
        out[i, 2] = fuzz.token_set_ratio(n1, n2) / 100.0
        out[i, 3] = fuzz.partial_ratio(n1, n2) / 100.0
        out[i, 4] = fuzz.ratio(p1, p2) / 100.0
        out[i, 5] = fuzz.token_sort_ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0
        out[i, 6] = fuzz.token_set_ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0
    return out


def add_fuzzy_features(df: pl.DataFrame, executor: ProcessPoolExecutor) -> pl.DataFrame:
    n = df.height
    if n == 0:
        for col in FUZZY_COLS:
            df = df.with_columns(pl.Series(col, [], dtype=pl.Float32))
        return df

    names1 = df["s1_name_core"].fill_null("").to_list()
    names2 = df["s2_name_core"].fill_null("").to_list()
    phons1 = df["s1_name_phon"].fill_null("").to_list()
    phons2 = df["s2_name_phon"].fill_null("").to_list()
    addrs1 = df["s1_addr_clean"].fill_null("").to_list()
    addrs2 = df["s2_addr_clean"].fill_null("").to_list()

    chunks = []
    for start in range(0, n, FUZZY_CHUNK_SIZE):
        end = min(start + FUZZY_CHUNK_SIZE, n)
        chunks.append((names1[start:end], names2[start:end], phons1[start:end], phons2[start:end], addrs1[start:end], addrs2[start:end]))

    results = list(executor.map(_fuzzy_chunk, chunks))
    feat_arr = np.concatenate(results, axis=0)
    for j, col in enumerate(FUZZY_COLS):
        df = df.with_columns(pl.Series(col, feat_arr[:, j]))
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading preprocessed source data...")
    s1_train = pl.read_parquet(TRAIN_S1)
    s2_train = pl.read_parquet(TRAIN_S2)
    s3_train = pl.read_parquet(TRAIN_S3)
    s2s3_train = pl.concat([s2_train, s3_train], how="vertical_relaxed")
    del s2_train, s3_train
    gc.collect()

    gt = pl.read_csv(TRAIN_GT, separator="\t")

    print("Loading + merging candidate pairs...")
    train_pairs_df = load_and_merge_candidates(CANDIDATE_PKLS_TRAIN)
    print(f"  train pairs: {train_pairs_df.height:,}")

    s1_ren = s1_train.select([c for c in FIELDS if c in s1_train.columns]).rename(
        {c: f"s1_{c}" for c in FIELDS if c != "entity_id"}
    ).rename({"entity_id": "source1_entity_id"})

    s2s3_ren = s2s3_train.select([c for c in FIELDS if c in s2s3_train.columns]).rename(
        {c: f"s2_{c}" for c in FIELDS if c != "entity_id"}
    ).rename({"entity_id": "candidate_entity_id"})

    gt_long = (
        gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
        .with_columns(pl.col("matched_entity_ids").str.split(",").alias("matched_list"))
        .explode("matched_list")
        .select(pl.col("source1_entity_id"), pl.col("matched_list").alias("candidate_entity_id"), pl.lit(1).cast(pl.Int8).alias("label"))
    )

    true_map = {}
    for row in gt.to_dicts():
        raw = row.get("matched_entity_ids")
        true_map[row["source1_entity_id"]] = set(raw.split(",")) if raw else set()

    all_s1_ids = s1_train["entity_id"].to_list()
    rng = np.random.default_rng(RANDOM_SEED)
    val_ids = set(rng.choice(all_s1_ids, size=int(len(all_s1_ids) * VAL_FRACTION), replace=False).tolist())
    val_ids_series = pl.Series("val_ids", list(val_ids))
    print(f"  {len(val_ids):,} S1 entities held out for validation")

    n_rows = train_pairs_df.height
    n_batches = math.ceil(n_rows / BATCH_SIZE)
    print(f"\nProcessing {n_rows:,} train pairs in {n_batches} batches of ~{BATCH_SIZE:,}...")

    fit_frames = []
    val_frames = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        for i in range(n_batches):
            batch = train_pairs_df.slice(i * BATCH_SIZE, BATCH_SIZE)
            batch = batch.join(s1_ren, on="source1_entity_id", how="left")
            batch = batch.join(s2s3_ren, on="candidate_entity_id", how="left")
            batch = add_vectorized_features(batch)
            batch = add_fuzzy_features(batch, executor)
            batch = batch.join(gt_long, on=["source1_entity_id", "candidate_entity_id"], how="left")
            batch = batch.with_columns(pl.col("label").fill_null(0))

            is_val = batch["source1_entity_id"].is_in(val_ids_series)
            val_part = batch.filter(is_val).select(KEEP_COLS)
            fit_part_all = batch.filter(~is_val)

            pos = fit_part_all.filter(pl.col("label") == 1)
            neg = fit_part_all.filter(pl.col("label") == 0)
            n_keep_neg = min(neg.height, pos.height * NEG_PER_POS)
            neg_sampled = neg.sample(n=n_keep_neg, seed=RANDOM_SEED + i) if n_keep_neg > 0 else neg.head(0)
            fit_part = pl.concat([pos, neg_sampled], how="vertical_relaxed").select(KEEP_COLS)

            fit_frames.append(fit_part)
            val_frames.append(val_part)

            del batch, is_val, val_part, fit_part_all, pos, neg, neg_sampled, fit_part
            gc.collect()
            print(f"  batch {i+1}/{n_batches} done "
                  f"(fit so far={sum(f.height for f in fit_frames):,}, val so far={sum(f.height for f in val_frames):,})")

    del train_pairs_df, s1_ren, s2s3_ren, gt_long
    gc.collect()

    fit_set = pl.concat(fit_frames, how="vertical_relaxed")
    val_set = pl.concat(val_frames, how="vertical_relaxed")
    del fit_frames, val_frames
    gc.collect()

    print(f"\nFinal: fit={fit_set.height:,} rows  val={val_set.height:,} rows")
    fit_set.write_parquet(os.path.join(OUT_DIR, "train_fit_features.parquet"))
    val_set.write_parquet(os.path.join(OUT_DIR, "train_val_features.parquet"))
    print(f"Wrote {OUT_DIR}/train_fit_features.parquet and train_val_features.parquet")

    print("\nTraining LightGBM...")
    X_fit = fit_set.select(FEATURE_COLS).to_pandas()
    y_fit = fit_set["label"].to_pandas()
    X_val = val_set.select(FEATURE_COLS).to_pandas()
    y_val = val_set["label"].to_pandas()

    train_data = lgb.Dataset(X_fit, label=y_fit)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.03,
        "max_depth": 8, "num_leaves": 63, "min_child_samples": 100,
        "subsample": 0.8, "colsample_bytree": 0.8, "verbosity": -1,
    }

    model = lgb.train(
        params, train_data, num_boost_round=2000, valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    model.save_model(os.path.join(OUT_DIR, "lgb_model.txt"))
    print(f"Saved model to {OUT_DIR}/lgb_model.txt")

    importance = sorted(zip(FEATURE_COLS, model.feature_importance(importance_type="gain")), key=lambda x: -x[1])
    print("\nFeature importance (gain):")
    for name, imp in importance:
        print(f"  {name}: {imp:.1f}")

    print("\nTuning threshold for macro F0.5 on validation set...")
    val_set = val_set.with_columns(pl.Series("pred_prob", model.predict(X_val)))

    val_grouped = val_set.group_by("source1_entity_id").agg(pl.col("candidate_entity_id"), pl.col("pred_prob"))
    val_records = val_grouped.to_dicts()

    best_threshold, best_f05 = 0.5, -1.0
    for threshold in np.arange(0.30, 0.96, 0.01):
        f05_scores = []
        for rec in val_records:
            s1_id = rec["source1_entity_id"]
            true_set = true_map.get(s1_id, set())
            predicted = {cid for cid, p in zip(rec["candidate_entity_id"], rec["pred_prob"]) if p >= threshold}
            if not true_set:
                f05_scores.append(1.0 if not predicted else 0.0)
                continue
            tp = len(true_set & predicted)
            precision = tp / len(predicted) if predicted else 0.0
            recall = tp / len(true_set)
            denom = 0.25 * precision + recall
            f05_scores.append((1.25 * precision * recall) / denom if denom > 0 else 0.0)
        macro_f05 = float(np.mean(f05_scores))
        if macro_f05 > best_f05:
            best_f05, best_threshold = macro_f05, threshold

    print(f"\nBest threshold: {best_threshold:.2f}  Macro F0.5: {best_f05:.4f}")
    with open(os.path.join(OUT_DIR, "best_threshold.txt"), "w") as f:
        f.write(str(best_threshold))
    print(f"Saved to {OUT_DIR}/best_threshold.txt")


if __name__ == "__main__":
    main()
