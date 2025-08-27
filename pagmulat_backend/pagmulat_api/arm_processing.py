import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import re

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
