import pandas as pd
import os
from ..config import config
from rest_framework.decorators import api_view
from rest_framework.response import Response

COLUMNS_TO_DROP = [
    'representativity', 'leverage', 'conviction',
    'zhangs_metric', 'jaccard', 'certainty', 'kulczynski'
]

def clean_df(df: pd.DataFrame) -> list:
    df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns], errors='ignore')


    df = df.rename(columns={
        'antecedents': 'antecedent',
        'consequents': 'consequent'
    })

    df = df.where(pd.notnull(df), None)

    return df.to_dict(orient='records')


@api_view(['GET'])
def get_algorithm_rules(request):
    try:
        apriori_file = os.path.join('data/rules', 'apriori_association_rules.csv')
        fp_growth_file = os.path.join('data/rules', 'fpgrowth_association_rules.csv')

        # Read CSV files
        apriori_df = pd.read_csv(apriori_file)
        fp_growth_df = pd.read_csv(fp_growth_file)

        # Clean dataframes
        apriori_rules = clean_df(apriori_df)
        fp_growth_rules = clean_df(fp_growth_df)

        return Response({
            "minSupport": config.MIN_SUPPORT,
            "minConfidence": config.MIN_CONFIDENCE,
            "minLift": config.MIN_LIFT,
            "aprioriRules": apriori_rules,
            "fpGrowthRules": fp_growth_rules
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)
