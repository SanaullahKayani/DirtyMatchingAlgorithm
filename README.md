# CAHPP Matching Pipeline

## Project Overview

The **CAHPP Matching Pipeline** is a Python script designed to process and match records between two datasets: a raw dataset from **Clinique** and a standardized dataset from **CAHPP**. The script performs matching based on multiple identifiers (IDT, CIP, UCD), cleans and preprocesses the data, and calculates confidence scores based on text similarity using the **Sentence-BERT** model.

The final output is a CSV file containing the matched records with confidence scores.

## Features

- **Data Preprocessing:** 
  - Cleans and normalizes various identifiers (IDT, CIP, UCD).
  - Handles and normalizes text data for similarity matching.
  
- **Matching Logic:**
  - Matches records based on IDT, CIP, and UCD identifiers.
  - Uses text similarity to further match articles with high accuracy.

- **Parallel Processing:** 
  - Utilizes multiple CPU cores to speed up the matching process.

- **Output:**
  - A CSV and Excel file of matched results, including confidence scores for each match.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- sentence-transformers
- scipy

You can install the required libraries using:

```bash
pip install -r requirements.txt
