def process_digital_distractions(df):
    distraction_cols = [col for col in df.columns if 'Q20' in col or 'Distraction' in col]
    return df[distraction_cols]
