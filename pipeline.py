"""
Auditing Geographic Domain Shift in Genomic AI:
A Memory-Efficient Pipeline for Mycobacterium tuberculosis Drug Resistance Prediction.

Author: Mohammad Kawsar
Affiliation: Department of Computer Science, University of North Carolina at Pembroke
"""

import argparse
import gc
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run TB Genomic Domain Shift Pipeline with SHAP Explainability."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to folder containing raw CRyPTIC CSV.GZ files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./figures",
        help="Path where generated figures and clean data will be saved",
    )
    parser.add_argument(
        "--drug",
        type=str,
        default="INH",
        help="Antibiotic target phenotype (default: INH)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5000,
        help="Number of lines per chunk for streaming mutation data",
    )
    return parser.parse_args()


def extract_metadata(data_dir: str, drug: str):
    """Loads DST measurements, extracts laboratory domains, and filters target phenotypes."""
    dst_path = os.path.join(data_dir, "DST_MEASUREMENTS.csv.gz")
    if not os.path.exists(dst_path):
        raise FileNotFoundError(f"Missing required file: {dst_path}")

    print("Step 1/5: Loading and filtering DST metadata...")
    dst_df = pd.read_csv(dst_path, low_memory=False)
    dst_df.columns = dst_df.columns.str.strip().str.upper()

    # Normalize drug column names across CRyPTIC releases
    if "DRUG" in dst_df.columns:
        meta_df = dst_df[dst_df["DRUG"] == drug].copy()
        phenotype_col = "PHENOTYPE"
    elif f"{drug}_PHENOTYPE" in dst_df.columns:
        meta_df = dst_df.copy()
        phenotype_col = f"{drug}_PHENOTYPE"
    else:
        raise KeyError(f"Could not locate drug column for {drug} in DST_MEASUREMENTS.")

    # Extract site domain from UNIQUEID (e.g., 'site.04.subj...' -> '04')
    id_col = "UNIQUEID" if "UNIQUEID" in meta_df.columns else meta_df.columns[0]
    meta_df["DOMAIN"] = meta_df[id_col].astype(str).str.split(".").str[1]

    # Filter binary phenotypes: Resistant (1) vs Susceptible (0)
    meta_df = meta_df[meta_df[phenotype_col].isin(["R", "S"])].copy()
    meta_df["Target"] = (meta_df[phenotype_col] == "R").astype(int)

    top_sites = meta_df["DOMAIN"].value_counts().index[:2].tolist()
    if len(top_sites) < 2:
        raise ValueError("Insufficient geographic sites available for a domain shift audit.")

    source_site, target_site = top_sites[0], top_sites[1]
    print(f" -> Source Domain (Training): Site {source_site}")
    print(f" -> Target Domain (Testing):  Site {target_site}")

    meta_subset = meta_df[meta_df["DOMAIN"].isin([source_site, target_site])][
        [id_col, "DOMAIN", "Target"]
    ].rename(columns={id_col: "UNIQUEID"})
    valid_ids = set(meta_subset["UNIQUEID"])

    del dst_df, meta_df
    gc.collect()

    return meta_subset, valid_ids, source_site, target_site


def stream_and_merge_mutations(
    data_dir: str, meta_subset: pd.DataFrame, valid_ids: set, chunk_size: int, output_csv: str
):
    """Streams massive vertical mutation records in chunks and writes merged data to disk."""
    mut_path = os.path.join(data_dir, "MUTATIONS.csv.gz")
    if not os.path.exists(mut_path):
        raise FileNotFoundError(f"Missing required file: {mut_path}")

    if os.path.exists(output_csv):
        os.remove(output_csv)

    print(f"Step 2/5: Streaming mutations in chunks of {chunk_size} and saving to {output_csv}...")
    first_chunk = True

    for i, chunk in enumerate(pd.read_csv(mut_path, chunksize=chunk_size, low_memory=False)):
        chunk.columns = chunk.columns.str.strip().str.upper()
        id_col = "UNIQUEID" if "UNIQUEID" in chunk.columns else chunk.columns[0]
        chunk = chunk.rename(columns={id_col: "UNIQUEID"})

        filtered_chunk = chunk[chunk["UNIQUEID"].isin(valid_ids)]
        if not filtered_chunk.empty:
            merged_chunk = pd.merge(meta_subset, filtered_chunk, on="UNIQUEID", how="inner")
            merged_chunk.to_csv(output_csv, mode="a", header=first_chunk, index=False)
            first_chunk = False

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1} chunks...")

    print(" -> Streaming and vertical merge complete.")


def build_genomic_matrix(clean_csv: str, source_site: str, target_site: str):
    """Pivots vertical mutation records into a wide, memory-efficient patient matrix."""
    print("Step 3/5: Loading merged data and building wide genomic matrix...")
    df = pd.read_csv(clean_csv, low_memory=False)

    pos_col = None
    for candidate in ["POSITION", "GENOME_INDEX", "POS"]:
        if candidate in df.columns:
            pos_col = candidate
            break

    if pos_col is None:
        raise KeyError("Could not find genomic coordinate column (POSITION/GENOME_INDEX).")

    df["MUT_FEATURE"] = "Pos_" + df[pos_col].astype(str)
    df["Presence"] = 1

    wide_df = df.pivot_table(
        index=["UNIQUEID", "DOMAIN", "Target"],
        columns="MUT_FEATURE",
        values="Presence",
        fill_value=0,
    ).reset_index()

    feature_cols = [c for c in wide_df.columns if c not in ["UNIQUEID", "DOMAIN", "Target"]]
    wide_df[feature_cols] = wide_df[feature_cols].astype("int8")

    # Drop zero-variance columns across both domains
    non_zero = wide_df[feature_cols].columns[wide_df[feature_cols].sum() > 0].tolist()
    wide_df = wide_df[["UNIQUEID", "DOMAIN", "Target"] + non_zero]

    del df
    gc.collect()

    print(
        f" -> Final matrix: {wide_df.shape[0]} patients, {len(non_zero)} unique mutation features."
    )

    train_data = wide_df[wide_df["DOMAIN"] == source_site]
    test_data = wide_df[wide_df["DOMAIN"] == target_site]

    X_train = train_data.drop(columns=["UNIQUEID", "DOMAIN", "Target"])
    y_train = train_data["Target"]

    X_test = test_data.drop(columns=["UNIQUEID", "DOMAIN", "Target"])
    y_test = test_data["Target"]

    return X_train, y_train, X_test, y_test


def train_and_evaluate(X_train, y_train, X_test, y_test, output_dir: str):
    """Trains XGBoost on source domain, audits target domain, and outputs SHAP visualizations."""
    print("Step 4/5: Training XGBoost on Source Domain...")
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)

    print("\n" + "=" * 45)
    print("       GEOGRAPHIC DOMAIN SHIFT AUDIT RESULTS")
    print("=" * 45)
    print(f"Source Domain Accuracy : {train_acc:.2%}")
    print(f"Target Domain Accuracy : {test_acc:.2%}")
    print(f"Shift Delta (Target-Src): {(test_acc - train_acc):+.2%}")
    print(f"Target Domain AUROC    : {test_auc:.4f}")
    print("=" * 45)
    print("\nTarget Domain Classification Report:")
    print(classification_report(y_test, test_preds, target_names=["Susceptible", "Resistant"]))

    print("Step 5/5: Running SHAP Explainability Audit...")
    os.makedirs(output_dir, exist_ok=True)
    sample_size = min(1000, len(X_test))
    X_test_sample = X_test.sample(n=sample_size, random_state=42)
    y_test_sample = y_test.loc[X_test_sample.index]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)

    # 1. Global Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    summary_path = os.path.join(output_dir, "SHAP_Geographic_Audit_Summary.png")
    plt.savefig(summary_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f" -> Saved: {summary_path}")

    # 2. Local Failure Case Waterfall Plot
    preds_sample = model.predict(X_test_sample)
    misclassified = preds_sample != y_test_sample

    if misclassified.any():
        fail_idx = misclassified[misclassified].index[0]
        iloc_idx = X_test_sample.index.get_loc(fail_idx)

        plt.figure()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[iloc_idx],
                base_values=explainer.expected_value,
                data=X_test_sample.iloc[iloc_idx],
                feature_names=X_test_sample.columns,
            ),
            show=False,
        )
        fail_path = os.path.join(output_dir, "SHAP_Failure_Case.png")
        plt.savefig(fail_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f" -> Saved: {fail_path}")
    else:
        print(" -> No misclassified samples found in sample to generate waterfall plot.")


def main():
    args = parse_arguments()
    clean_csv_path = os.path.join(args.output_dir, "intermediate_merged.csv")

    meta_subset, valid_ids, src, tgt = extract_metadata(args.data_dir, args.drug)
    stream_and_merge_mutations(args.data_dir, meta_subset, valid_ids, args.chunk_size, clean_csv_path)
    X_train, y_train, X_test, y_test = build_genomic_matrix(clean_csv_path, src, tgt)
    train_and_evaluate(X_train, y_train, X_test, y_test, args.output_dir)

    if os.path.exists(clean_csv_path):
        os.remove(clean_csv_path)
    print("\nPipeline run complete.")


if __name__ == "__main__":
    main()