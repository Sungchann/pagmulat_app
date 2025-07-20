import os
import pandas as pd
from pagmulat_api.data_processing_transformation.processors.full_pipeline import run_full_preprocessing_pipeline

# Define paths relative to backend root
RAW_PATH = "data/raw/student_raw.csv"
PROCESSED_PATH = "data/processed/student_final_preprocessed.csv"

# Load raw dataset
df_raw = pd.read_csv(RAW_PATH)

# Run preprocessing
df_processed = run_full_preprocessing_pipeline(df_raw)

# Ensure output folder exists
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)

# Save the result
df_processed.to_csv(PROCESSED_PATH, index=False)

print(f"[✓] Preprocessed data saved to: {PROCESSED_PATH}")
