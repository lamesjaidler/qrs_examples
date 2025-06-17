import pandas as pd
from typing import List
from styler_formatters import _BaseStylerFormatter


def apply_styler_formatters(df: pd.DataFrame,
                            styler_formatters: List[_BaseStylerFormatter]):
    df_style = df.style
    for styler in styler_formatters:
        df_style = styler.transform(df_style)
    return df_style
