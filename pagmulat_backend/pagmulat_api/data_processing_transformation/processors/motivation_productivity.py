def process_academic_motivation(df):
    motivation_cols = [col for col in df.columns if 'Q17' in col or 'Motivation' in col]
    return df[motivation_cols]
