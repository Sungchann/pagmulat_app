import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import re

def preprocess_student_data(file_path):
    students = pd.read_csv(file_path)
    column_renamed_map = {
        'Timestamp': 'Timestamp',
        'Year Level': 'Year_Level',
        'Program': 'Program',
        'Age (Numeric Form Only)': 'Age',
        'Gender': 'Gender',
        '5. How often do you use the Learning Management System (e.g., Canvas, MS Teams, Google Classroom)?': 'LMS_Usage',
        '6. How often do you use online code compiling platforms (e.g., GitHub, Replit)?': 'Code_Compiler_Usage',
        '7. How often do you use ChatGPT or other AI tools for studying?': 'ChatGPT_AI_Usage',
        '8. How often do you visit social media on a daily basis (e.g., Facebook, Tiktok, Instagram)?': 'Social_Media_Daily_Visit',
        '9. Do you follow a fixed daily study schedule?': 'Fixed_Study_Schedule',
        '10. On average, how many hours do you study outside class per day?': 'Study_Hours_Outside_Class',
        '11. What time do you usually start studying?': 'Study_Start_Time',
        '12. How often do you submit assignments on time? ': 'Submit_Assignments_On_Time',
        '13. When working on group projects, how often do you use online collaboration tools (e.g., Google Docs, MS Teams, Discord)?': 'Online_Collaboration_Tools',
        '14. What time do you usually sleep on weekdays?': 'Sleep_Time_Weekdays',
        '15. Do you feel burnt out or mentally exhausted due to school-related digital activities?': 'Burnout_Exhaustion',
        '16. Do you take regular breaks when using devices for studying?': 'Regular_Breaks_Studying',
        'Optional: Any digital habits you want to share that help you study better?': 'Optional_Study_Habits',
        ' 17. How motivated are you to study on a daily basis?': 'Motivation_Level',
        '18. What usually triggers your motivation to study? (Select all that apply)': 'Motivation_Triggers',
        '19. Do you consider yourself productive when studying using digital tools? ': 'Productive_Digital_Tools',
        '20. Which of the following tools do you use for productivity? (Select all that apply)': 'Productivity_Tools',
        '21. How often do you get distracted by social media during study time?': 'Social_Media_Distraction',
        '22. What platform distracts you the most during school hours?': 'Most_Distracting_Platform',
        '23. Do you use apps or methods to block distractions while studying?': 'Distraction_Blocking_Apps',
        '24. Which of the following platforms do you use regularly for academic purposes? (Select all that apply)': 'Academic_Platforms',
        '25. Do you think you rely too much on digital tools(e.g., AI, Google, Youtube) for completing academic tasks?': 'Over_Reliance_Digital_Tools',
        '26. What digital behavior do you think you need to maintain and somehow improve?': 'Behavior_To_Maintain_Improve',
        '27. What digital behavior do you think you need to change or reduce?': 'Behavior_To_Change_Reduce',
        '28. Do you think your current digital habits affect your grades or performance? Why or why not?': 'Affect_Grades_Performance',
    }
    students.rename(columns=column_renamed_map, inplace=True)
    students.drop(columns=['Timestamp'], inplace=True, errors='ignore')
    # Age cleaning
    age_column_as_str = students['Age'].astype(str)
    non_numeric_mask = age_column_as_str.str.contains(r'[^0-9]', regex=True)
    students.loc[non_numeric_mask, 'Age'] = 20  # Example fix for 'XX'
    students['Age'] = pd.to_numeric(students['Age'], errors='coerce')
    age_bins = [0, 18, 21, 24, 27, 30, np.inf]
    age_labels = ['<18', '18-20', '21-23', '24-26', '27-29', '30+']
    students['Age_Group_Label'] = pd.cut(
        students['Age'], bins=age_bins, labels=age_labels, right=False, include_lowest=True)
    # One-hot encoding for basic info
    processed_student_basic_info = pd.concat([
        pd.get_dummies(students['Year_Level'], prefix='Year_Level'),
        pd.get_dummies(students['Program'], prefix='Program'),
        pd.get_dummies(students['Age_Group_Label'], prefix='Age')
    ], axis=1)
    # ...additional preprocessing as in your notebook...
    return processed_student_basic_info

# --- ARM Mining from Preprocessed XLSX ---
from mlxtend.frequent_patterns import apriori, association_rules

def mine_arm_from_xlsx(xlsx_path, min_support=0.15, min_confidence=0.65, min_lift=1.2):
    """
    Loads a preprocessed XLSX file and runs Apriori/association rule mining.
    Returns a DataFrame of rules.
    """
    df = pd.read_excel(xlsx_path)
    # Ensure all columns are boolean (0/1 or True/False)
    df_bool = df.astype(bool)
    frequent_itemsets = apriori(df_bool, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['lift'] > min_lift]
    # Return selected columns for clarity
    return rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
