import pandas as pd
from .demographics import process_age, process_year_level, process_program
from .platform_usage import process_platform_familiarity, process_platform_reliance
from .motivation_productivity import process_academic_motivation
from .distraction_analysis import process_digital_distractions

def run_full_preprocessing_pipeline(df):
    age_encoded = process_age(df)
    year_encoded = process_year_level(df)
    program_encoded = process_program(df)
    platform_familiar = process_platform_familiarity(df)
    platform_rely = process_platform_reliance(df)
    motivation = process_academic_motivation(df)
    distractions = process_digital_distractions(df)

    final_df = pd.concat([
        age_encoded,
        year_encoded,
        program_encoded,
        platform_familiar,
        platform_rely,
        motivation,
        distractions
    ], axis=1)

    return final_df
