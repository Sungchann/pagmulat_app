import pandas as pd
from ..mappings.roman_to_int_map import ROMAN_TO_INT
from ..mappings.program_mapping import PROGRAM_MAP

def process_age(df):
    bins = [0, 18, 20, 22, 24, 100]
    labels = ['<18', '18-20', '21-22', '23-24', '25+']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    return pd.get_dummies(df['Age_Group'], prefix='Age')

def process_year_level(df):
    df['Year_Level_Num'] = df['Year_Level'].map(ROMAN_TO_INT)
    return pd.get_dummies(df['Year_Level_Num'], prefix='Year')

def process_program(df):
    df['Program_Name'] = df['Program'].map(PROGRAM_MAP)
    return pd.get_dummies(df['Program_Name'], prefix='Program')
