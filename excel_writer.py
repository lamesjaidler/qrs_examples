import pandas as pd


def write_dfs_to_excel(config: dict,
                       fpath: str):
    """
    Write dataframes to separate sheets in an Excel file with hyperlinks.

    Parameters
    ----------
    config : dict
        Config dict where the keys are sheet names and the values are Styler 
        objects
    fpath : str
        Path to write the Excel file to

    Example
    -------
    ```python
    config = {
            '<sheet_name>': Styler(),
            ...
        }
    ```
    """
    with pd.ExcelWriter(fpath, engine='xlsxwriter') as writer:
        for sheet_name, styler in config.items():
            styler.to_excel(
                writer,
                index=False,
                sheet_name=sheet_name,
            )  # Export data
