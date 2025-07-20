from ..utils.list_splitters import make_list_from_commas
from ..utils.encoding_helpers import multilabel_binarize

def process_platform_familiarity(df):
    df['Platforms_Familiar_List'] = df['Platforms_Familiar'].apply(make_list_from_commas)
    return multilabel_binarize(df, 'Platforms_Familiar_List', 'Familiar')

def process_platform_reliance(df):
    df['Platforms_Rely_List'] = df['Platforms_Rely_On'].apply(make_list_from_commas)
    return multilabel_binarize(df, 'Platforms_Rely_List', 'RelyOn')
