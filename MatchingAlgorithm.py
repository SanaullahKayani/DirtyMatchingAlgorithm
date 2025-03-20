#!/usr/bin/env python3
"""
Project: CAHPP Matching Pipeline
Description: This script loads raw and standard Excel files, determines the best matching columns, preprocesses the data,
matches records using multiple identifiers and text similarity (via Sentence-BERT), and outputs a CSV of matched results.
"""

import os
import re
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist  # (optional alternative metric)

# Global file paths
CLINIQUE_FILE_PATH = "MSPB-33-Achats-Déc2018_Nov2019.xlsx"
CAHPP_FILE_PATH = "BaseCahppPH.xlsx"
EMBEDDING_FILE = "cahpp_embeddings.npy"
PARQUET_FILE = "cahpp_metadata.parquet"

# =============================================================================
# Utility Functions
# =============================================================================
def normalize_values(value):
    """
    Convert numeric values into a consistent string format.
    Removes the 'PH' prefix from IDT values if present.
    """
    try:
        value = str(value).strip()
        if value.startswith("PH"):
            value = value[2:]
        return str(int(float(value)))  # Converts scientific notation to integer then to string
    except ValueError:
        return str(value)


def find_best_matching_column(df_raw, df_standard, col_standard_1, col_standard_2=None):
    """
    Identify the column in df_raw that best matches the values from col_standard_1 (and optionally col_standard_2)
    in df_standard using normalized values.
    Returns the best matching column name and the number of matches.
    """
    values_set_1 = set(df_standard[col_standard_1].dropna().astype(str).apply(normalize_values))
    values_set_2 = (
        set(df_standard[col_standard_2].dropna().astype(str).apply(normalize_values))
        if col_standard_2 else set()
    )

    best_match_col = None
    best_match_count = 0

    for col in df_raw.columns:
        df_col_values = df_raw[col].dropna().astype(str).apply(normalize_values)
        match_count = df_col_values.isin(values_set_1).sum()
        if col_standard_2:
            match_count += df_col_values.isin(values_set_2).sum()
        if match_count > best_match_count:
            best_match_count = match_count
            best_match_col = col

    return best_match_col, best_match_count


def find_article_label_column(df_raw, df_standard, sample_size=100):
    """
    Identify the column in df_raw that contains article labels similar to DESIGNATIONFR in df_standard.
    The function looks for patterns like 'mg', 'ml', 'g', or '%' in the text.
    """
    pattern = re.compile(r'\b(\d+mg|\d+ml|\d+g|\d+%)\b', re.IGNORECASE)
    best_match_col = None
    best_match_score = 0

    for col in df_raw.columns:
        raw_values = df_raw[col].dropna().astype(str).head(sample_size)
        match_score = sum(bool(pattern.search(val)) for val in raw_values)
        if match_score > best_match_score:
            best_match_score = match_score
            best_match_col = col

    return best_match_col


def clean_text(text):
    """Clean text by converting to lowercase and removing punctuation."""
    if pd.isna(text):
        return ""
    return re.sub(r"[^\w\s]", "", str(text).lower())


def normalize_cid_ucd(value):
    """
    Convert UCD/CIP values to a clean string format without decimals.
    """
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return value.split(".")[0]

def multi_column_match(clinique_df, cahpp_df):
    """
    Matches records based on IDT, CIP, and UCD and merges the corresponding rows
    from clinique_df and cahpp_df into a new DataFrame with boolean match indicators.

    Returns:
        matched_clinique_cahpp_df: A merged DataFrame with all columns from both clinique_df and cahpp_df
        along with boolean match indicators (idt_match, cip_match, ucd_match).
    """
    # Assign unique index to both dataframes
    clinique_df = clinique_df.reset_index(drop=True)
    cahpp_df = cahpp_df.reset_index(drop=True)

    # Dictionary to store matched rows with proper updates
    match_dict = {}

    # Match IDT
    matched_idt = clinique_df.merge(cahpp_df, left_on="clinique_IDT", right_on="IDT", how="inner")
    for idx, row in matched_idt.iterrows():
        key = row["clinique_IDT"]
        if key not in match_dict:
            match_dict[key] = {**row.to_dict(), "idt_match": 1, "cip_match": 0, "ucd_match": 0}
        else:
            match_dict[key]["idt_match"] = 1  # Ensure IDT match flag is always updated

    # Match CIP
    for _, row in clinique_df.iterrows():
        cip_matches = pd.DataFrame()
        if pd.notna(row["clinique_CIP"]):
            clinique_cip = str(row["clinique_CIP"]).strip()
            if len(clinique_cip) == 7:
                cip_matches = cahpp_df[cahpp_df["CIP"] == clinique_cip].copy()
            elif len(clinique_cip) == 13:
                cip_matches = cahpp_df[cahpp_df["CIP13"] == clinique_cip].copy()

        for _, cip_row in cip_matches.iterrows():
            key = row["clinique_IDT"]
            if key not in match_dict:
                match_dict[key] = {**row.to_dict(), **cip_row.to_dict(), "idt_match": 0, "cip_match": 1, "ucd_match": 0}
            else:
                match_dict[key]["cip_match"] = 1  # Update only cip_match, keeping existing values

    # Match UCD
    for _, row in clinique_df.iterrows():
        ucd_matches = pd.DataFrame()
        if pd.notna(row["clinique_UCD"]):
            clinique_ucd = str(row["clinique_UCD"]).strip()
            ucd_matches = cahpp_df[(cahpp_df["UCD"] == clinique_ucd) | (cahpp_df["UCD13"] == clinique_ucd)].copy()

        for _, ucd_row in ucd_matches.iterrows():
            key = row["clinique_IDT"]
            if key not in match_dict:
                match_dict[key] = {**row.to_dict(), **ucd_row.to_dict(), "idt_match": 0, "cip_match": 0, "ucd_match": 1}
            else:
                match_dict[key]["ucd_match"] = 1  # Update only ucd_match, keeping existing values

    # Convert dictionary back to DataFrame
    matched_clinique_cahpp_df = pd.DataFrame(match_dict.values())

    print(f"Number of matched rows: {len(matched_clinique_cahpp_df)}")
    # print(matched_clinique_cahpp_df.head())
    # matched_clinique_cahpp_df.to_excel("matched_clinique_cahpp_df.xlsx", index=False)
    return matched_clinique_cahpp_df


def compute_confidence(matched_data, threshold=0.5):
    """
    Computes match confidence scores for each row in matched_data using text similarity
    and identifier matches. Converts scores to percentages.

    Returns:
        A list of result dictionaries containing match confidence scores.
    """

    # Compute text similarity using cosine similarity

    results = []
    for idx, row in matched_data.iterrows():
        cip_match_score = row["cip_match"]
        ucd_match_score = row["ucd_match"]
        idt_match_score = row["idt_match"]

        # Get the text similarity score (self-similarity is 1, ignore it)
        #best_match_score = similarities[idx].max()

        # Define weights for each component
        cip_weight = 0.4   # 30% for CIP match
        ucd_weight = 0.2   # 20% for UCD match
        idt_weight = 0.4   # 30% for IDT match
        #text_weight = 0.2  # 20% for text similarity

        # Calculate match confidence
        match_confidence = (
            (cip_match_score * cip_weight) +
            (ucd_match_score * ucd_weight) +
            (idt_match_score * idt_weight)
            #+ (best_match_score * text_weight)
        )

        # **Ensure it doesn’t exceed 1**
        match_confidence = min(match_confidence, 1)

        # Convert to percentage
        match_confidence_percentage = round(match_confidence * 100, 2)
        #best_match_score_percentage = round(best_match_score * 100, 2)

        if match_confidence > threshold:
            results.append({
                "Clinique_Article": row.get("libelle_article_clean"),
                "BaseCahpp_Article": row.get("DESIGNATIONFR"),
                # "Match_Confidence (%)": match_confidence_percentage,  # Convert to %
                #"Article_Similarity (%)": best_match_score_percentage,  # Convert to %
                "cip_match": cip_match_score,
                "ucd_match": ucd_match_score,
                "idt_match": idt_match_score,
                "Clinique_IDT": row.get("clinique_IDT"),
                "Clinique_CIP": row.get("clinique_CIP"),
                "Clinique_UCD": row.get("clinique_UCD"),
                "CAHPP_IDT": row.get("IDT"),
                "CAHPP_CIP": row.get("CIP"),
                "CAHPP_CIP13": row.get("CIP13"),
                "CAHPP_UCD": row.get("UCD"),
                "CAHPP_UCD13": row.get("UCD13"),
            })

    return results


def process_chunk(chunk, threshold):
    """
    Process a DataFrame chunk, matching articles for each row.
    Returns a list of matching results.
    """
    return compute_confidence(chunk, threshold)



# =============================================================================
# Main Function
# =============================================================================
def main():
    # Instantiate the SentenceTransformer model (adjust the model name as needed)
    #model = SentenceTransformer('all-MiniLM-L6-v2')

    # -------------------------------------------------------------------------
    # Step 1: Load Excel Files and Determine Matching Columns
    # -------------------------------------------------------------------------
    print("Loading Excel files...")
    df_standard = pd.read_excel(CAHPP_FILE_PATH)
    df_raw = pd.read_excel(CLINIQUE_FILE_PATH)

    clinique_columns_to_load = []
    # Identify best matching columns for identifiers and article labels
    best_idt_col, idt_matches = find_best_matching_column(df_raw, df_standard, "IDT")
    if best_idt_col:
        print(f"Best matching IDT column: {best_idt_col} ({idt_matches} matches)")
        clinique_columns_to_load.append(best_idt_col)
    else:
        print("No matching IDT column found.")

    best_cip_col, cip_matches = find_best_matching_column(df_raw, df_standard, "CIP", "CIP13")
    if best_cip_col:
        print(f"Best matching CIP column: {best_cip_col} ({cip_matches} matches)")
        clinique_columns_to_load.append(best_cip_col)
    else:
        print("No matching CIP column found.")

    best_ucd_col, ucd_matches = find_best_matching_column(df_raw, df_standard, "UCD", "UCD13")
    if best_ucd_col:
        print(f"Best matching UCD column: {best_ucd_col} ({ucd_matches} matches)")
        clinique_columns_to_load.append(best_ucd_col)
    else:
        print("No matching UCD column found.")

    best_article_label_col = find_article_label_column(df_raw, df_standard)
    if best_article_label_col:
        print(f"Best matching Article Label column: {best_article_label_col}")
        clinique_columns_to_load.append(best_article_label_col)
    else:
        print("No matching Article Label column found.")

    # -------------------------------------------------------------------------
    # Step 2: Load Only the Required Columns and Preprocess the Data
    # -------------------------------------------------------------------------
    print("Loading and preprocessing data...")
    cahpp_columns_to_load = ["IDT", "CIP", "CIP13", "UCD", "UCD13", "DESIGNATIONFR"]

    clinique_df = pd.read_excel(CLINIQUE_FILE_PATH, usecols=clinique_columns_to_load)
    cahpp_df = pd.read_excel(CAHPP_FILE_PATH, usecols=cahpp_columns_to_load)

    # Clean text columns
    if best_article_label_col:
        clinique_df["libelle_article_clean"] = clinique_df[best_article_label_col].apply(clean_text)
    cahpp_df["DESIGNATIONFR_clean"] = cahpp_df["DESIGNATIONFR"].apply(clean_text)

    # Normalize UCD/CIP columns for CAHPP file
    for col in ["UCD13", "CIP13", "UCD", "CIP"]:
        cahpp_df[col] = cahpp_df[col].apply(normalize_cid_ucd)

    # Normalize and clean Clinique identifier columns
    if best_idt_col:
        clinique_df[best_idt_col] = clinique_df[best_idt_col].astype(str).str.strip().str.replace("^PH", "", regex=True)
    if best_cip_col:
        clinique_df[best_cip_col] = clinique_df[best_cip_col].apply(normalize_cid_ucd)
    if best_ucd_col:
        clinique_df[best_ucd_col] = clinique_df[best_ucd_col].apply(normalize_cid_ucd)

    # Ensure CAHPP identifier columns are string type and stripped
    for col in ["IDT", "CIP", "CIP13", "UCD", "UCD13"]:
        cahpp_df[col] = cahpp_df[col].astype(str).str.strip()

    # Rename Clinique columns to standardized names
    rename_map = {
        best_idt_col: "clinique_IDT",
        best_cip_col: "clinique_CIP",
        best_ucd_col: "clinique_UCD",
        best_article_label_col: "clinique_article_label",
    }
    clinique_df.rename(columns=rename_map, inplace=True)

    # -------------------------------------------------------------------------
    # Step 3: Multi-Column Matching Based on Identifiers
    # -------------------------------------------------------------------------
    print("Performing multi-column matching...")
    matched_data = multi_column_match(clinique_df, cahpp_df)
    print(f"Number of matched rows: {len(matched_data)}")

    # # -------------------------------------------------------------------------
    # # Step 4: Load or Compute Embeddings for CAHPP Articles
    # # -------------------------------------------------------------------------
    # print("Processing embeddings...")
    # if os.path.exists(EMBEDDING_FILE) and os.path.exists(PARQUET_FILE):
    #     print("Loading precomputed embeddings and metadata...")
    #     cahpp_embeddings = np.load(EMBEDDING_FILE)
    #     cahpp_df = pd.read_parquet(PARQUET_FILE)
    # else:
    #     print("Computing embeddings...")
    #     cahpp_embeddings = model.encode(matched_data["DESIGNATIONFR_clean"].tolist())
    #     np.save(EMBEDDING_FILE, cahpp_embeddings)
    #     cahpp_df.to_parquet(PARQUET_FILE, index=False)

    # -------------------------------------------------------------------------
    # Step 5: Text Similarity Matching and Parallel Processing
    # -------------------------------------------------------------------------
    print("Performing text similarity & confidence matching...")
    threshold = 0.5  # Define the minimum match confidence threshold
    num_cores = cpu_count()
    chunk_size = len(matched_data) // num_cores if num_cores > 0 else len(matched_data)
    chunks = [matched_data.iloc[i:i + chunk_size] for i in range(0, len(matched_data), chunk_size)]

    # Prepare arguments for starmap
    args = [(chunk, threshold) for chunk in chunks]


    with Pool(processes=num_cores) as pool:
        results = pool.starmap(process_chunk, args)

    flattened_results = [item for sublist in results for item in sublist]

    # Save the final matched results to CSV
    result_df = pd.DataFrame(flattened_results)
    result_csv_path = "matched_results.csv"
    result_df.to_csv(result_csv_path, index=False)
    print(f"Matching complete. Results saved to '{result_csv_path}'.")
    result_df.to_excel("matched_resultsOfMSPB-33.xlsx", index=False)

# =============================================================================
# Entry Point
# =============================================================================
if __name__ == '__main__':
    main()
