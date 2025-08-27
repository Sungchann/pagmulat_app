import pandas as pd
from .arm_columns import all_columns

def process_response(survey_response):
    processed_data = {column: 0 for column in all_columns}

    for key, value in survey_response.items():
        if isinstance(value, str):
            if key in ['gender', 'digital_habits', 'digital_change', 'digital_reflection']:
                pass 
            else:
                if value in all_columns:
                    processed_data[value] = 1

    age = survey_response.get('age')
    if age is not None:
        if age < 18:
            processed_data['Age_Under18'] = 1
        elif 18 <= age <= 20:
            processed_data['Age_18_20'] = 1
        elif 21 <= age <= 23:
            processed_data['Age_21_23'] = 1
        elif 24 <= age <= 26:
            processed_data['Age_24_26'] = 1
        elif 27 <= age <= 29:
            processed_data['Age_27_29'] = 1
        elif age >= 30:
            processed_data['Age_30_Plus'] = 1

    for checkbox_key in ['motivation_triggers', 'productivity_tools', 'academic_tools']:
        selected_values = survey_response.get(checkbox_key, [])
        for value in selected_values:
            if value in all_columns:
                processed_data[value] = 1

    return processed_data
