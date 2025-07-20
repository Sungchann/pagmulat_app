import pandas as pd
import numpy as np

def load_cleaned_data(path="data/interim/student_cleaned.csv"):
    return pd.read_csv(path)

def identify_rare_patterns(df, behavior_columns, threshold=2):
    pattern_counts = df[behavior_columns].value_counts().reset_index()
    pattern_counts.columns = behavior_columns + ['Count']
    return pattern_counts[pattern_counts['Count'] <= threshold]

def generate_synthetic_from_rare_patterns(rare_patterns, behavior_columns, augment_by=3):
    synthetic_data = pd.DataFrame()
    for _, row in rare_patterns.iterrows():
        pattern = row[:-1]
        for _ in range(augment_by):
            synthetic_data = pd.concat([synthetic_data, pd.DataFrame([pattern])], ignore_index=True)
    synthetic_data[behavior_columns] = synthetic_data[behavior_columns].astype(int)
    return synthetic_data

def fill_proportionally(synthetic_df, real_df, column_groups):
    for col_set in column_groups:
        proportions = real_df[col_set].sum() / len(real_df)
        proportions = proportions / proportions.sum()
        choices = col_set
        for idx in synthetic_df.index:
            selected = np.random.choice(choices, p=proportions)
            for col in col_set:
                synthetic_df.at[idx, col] = 1 if col == selected else 0
    return synthetic_df

def save_synthetic_data(final_df, output_path="data/processed/dataset_with_synthetic_data.csv"):
    final_df.to_csv(output_path, index=False)

def run_synthetic_pipeline():
    df = load_cleaned_data()
    
    behavior_columns = [
        'ChatGPT_AI_Usage_Always', 'Social_Media_Daily_Visit_Always',
        'uses_music', 'uses_pomodoro', 'reduce_social_media',
        'reduce_ai_dependence', 'uses_official_docs', 'time_awareness'
    ]
    
    year_cols = ['Year_1', 'Year_2', 'Year_3', 'Year_4']
    program_cols = ['Program_CS', 'Program_EMC', 'Program_IT']
    age_cols = ['Age_Under18', 'Age_18_20', 'Age_21_23', 'Age_24_26', 'Age_27_29', 'Age_30_Plus']
    
    rare_patterns = identify_rare_patterns(df, behavior_columns)
    synthetic = generate_synthetic_from_rare_patterns(rare_patterns, behavior_columns)
    synthetic = fill_proportionally(synthetic, df, [year_cols, program_cols, age_cols])
    
    final_df = pd.concat([df, synthetic], ignore_index=True)
    save_synthetic_data(final_df)

if __name__ == "__main__":
    run_synthetic_pipeline()
