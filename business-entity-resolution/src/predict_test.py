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

TEST_S1 = os.path.join(DATA_DIR, "test", "test_source1.parquet")
TEST_S2 = os.path.join(DATA_DIR, "test", "test_source2.parquet")
TEST_S3 = os.path.join(DATA_DIR, "test", "test_source3.parquet")

CANDIDATE_PKLS_TEST = ["blocking_person_a_test.pkl"]

MODEL_PATH = os.path.join("output_features", "lgb_model.txt")
THRESHOLD_PATH = os.path.join("output_features", "best_threshold.txt")

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

N_WORKERS = max(1, os.cpu_count() - 1)
FUZZY_CHUNK_SIZE = 200_000
BATCH_SIZE = 1_000_000

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


def dict_to_polars(cand_dict, batch_size=2_000_000):
    frames = []
    ids_buf, cids_buf, scores_buf = [], [], []
    count = 0
    for s1_id, cands in cand_dict.items():
        for c in cands:
            if isinstance(c, (tuple, list)):
                cid, score = c[0], c[1]
            else:
                cid, score = c, None
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
    print("Loading model + threshold...")
    model = lgb.Booster(model_file=MODEL_PATH)
    with open(THRESHOLD_PATH) as f:
        threshold = float(f.read().strip())
    print(f"  threshold = {threshold}")

    print("Loading preprocessed test source data...")
    s1_test = pl.read_parquet(TEST_S1)
    s2_test = pl.read_parquet(TEST_S2)
    s3_test = pl.read_parquet(TEST_S3)
    s2s3_test = pl.concat([s2_test, s3_test], how="vertical_relaxed")
    del s2_test, s3_test
    gc.collect()

    print("Loading + merging candidate pairs...")
    test_pairs_df = load_and_merge_candidates(CANDIDATE_PKLS_TEST)
    test_pairs_df = test_pairs_df.with_columns(pl.col("blocking_score").cast(pl.Float64))
    print(f"  test pairs: {test_pairs_df.height:,}")

    s1_ren = s1_test.select([c for c in FIELDS if c in s1_test.columns]).rename(
        {c: f"s1_{c}" for c in FIELDS if c != "entity_id"}
    ).rename({"entity_id": "source1_entity_id"})

    s2s3_ren = s2s3_test.select([c for c in FIELDS if c in s2s3_test.columns]).rename(
        {c: f"s2_{c}" for c in FIELDS if c != "entity_id"}
    ).rename({"entity_id": "candidate_entity_id"})

    all_test_s1_ids = s1_test["entity_id"].to_list()

    n_rows = test_pairs_df.height
    n_batches = math.ceil(n_rows / BATCH_SIZE)
    print(f"\nScoring {n_rows:,} test pairs in {n_batches} batches of ~{BATCH_SIZE:,}...")

    cand_batch_dir = os.path.join(OUT_DIR, "_cand_batches")
    match_batch_dir = os.path.join(OUT_DIR, "_match_batches")
    prob_batch_dir = os.path.join(OUT_DIR, "_prob_batches")
    os.makedirs(cand_batch_dir, exist_ok=True)
    os.makedirs(match_batch_dir, exist_ok=True)
    os.makedirs(prob_batch_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        for i in range(n_batches):
            cand_out_path = os.path.join(cand_batch_dir, f"batch_{i:04d}.parquet")
            match_out_path = os.path.join(match_batch_dir, f"batch_{i:04d}.parquet")
            prob_out_path = os.path.join(prob_batch_dir, f"batch_{i:04d}.parquet")
            if os.path.exists(cand_out_path) and os.path.exists(match_out_path) and os.path.exists(prob_out_path):
                print(f"  batch {i+1}/{n_batches} already done, skipping")
                continue

            batch = test_pairs_df.slice(i * BATCH_SIZE, BATCH_SIZE)

            batch.select(["source1_entity_id", "candidate_entity_id"]).write_parquet(cand_out_path)

            batch = batch.join(s1_ren, on="source1_entity_id", how="left")
            batch = batch.join(s2s3_ren, on="candidate_entity_id", how="left")
            batch = add_vectorized_features(batch)
            batch = add_fuzzy_features(batch, executor)

            X = batch.select(FEATURE_COLS).to_pandas()
            probs = model.predict(X)

            batch.select(["source1_entity_id", "candidate_entity_id"]).with_columns(
                pl.Series("prob", probs)
            ).write_parquet(prob_out_path)

            kept = batch.select(["source1_entity_id", "candidate_entity_id"]).filter(
                pl.Series("keep", probs >= threshold)
            )
            kept.write_parquet(match_out_path)

            del batch, X, probs, kept
            gc.collect()
            print(f"  batch {i+1}/{n_batches} scored")

    del test_pairs_df, s1_ren, s2s3_ren
    gc.collect()

    print("\nBuilding candidate_pairs.tsv (reading batches back from disk)...")
    cand_df = pl.read_parquet(os.path.join(cand_batch_dir, "*.parquet"))
    cand_grouped = cand_df.group_by("source1_entity_id").agg(pl.col("candidate_entity_id"))
    cand_map = {row["source1_entity_id"]: row["candidate_entity_id"] for row in cand_grouped.to_dicts()}

    with open(os.path.join(OUT_DIR, "candidate_pairs.tsv"), "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_test_s1_ids:
            ids = cand_map.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(ids)}\n")
    print(f"  wrote {OUT_DIR}/candidate_pairs.tsv")

    print("Building matching_results.tsv (reading batches back from disk)...")
    match_df = pl.read_parquet(os.path.join(match_batch_dir, "*.parquet"))
    match_grouped = match_df.group_by("source1_entity_id").agg(pl.col("candidate_entity_id"))
    match_map = {row["source1_entity_id"]: row["candidate_entity_id"] for row in match_grouped.to_dicts()}

    n_singleton_pred = 0
    with open(os.path.join(OUT_DIR, "matching_results.tsv"), "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_test_s1_ids:
            ids = match_map.get(s1_id, [])
            if not ids:
                n_singleton_pred += 1
            f.write(f"{s1_id}\t{','.join(ids)}\n")
    print(f"  wrote {OUT_DIR}/matching_results.tsv")
    print(f"  {len(all_test_s1_ids):,} total S1 entities, {n_singleton_pred:,} predicted as singletons ({100*n_singleton_pred/len(all_test_s1_ids):.1f}%)")
    print("\nDone. Run utils/validate_submission.py next to confirm format before uploading.")


if __name__ == "__main__":
    main()
