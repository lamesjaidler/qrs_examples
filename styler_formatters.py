import pandas as pd
from abc import ABC, abstractmethod
from pandas.io.formats.style import Styler
from typing import Literal


class _BaseStylerFormatter(ABC):

    @abstractmethod
    def transform(self,
                  styler: Styler):
        pass


class HighlightValues(_BaseStylerFormatter):

    def __init__(self,
                 value_colour_map: dict,
                 **applymap_kwargs):
        self.value_colour_map = value_colour_map
        self.applymap_kwargs = applymap_kwargs

    def transform(self,
                  styler: Styler):
        return styler.applymap(self._highlight_value, **self.applymap_kwargs)

    def _highlight_value(self,
                         x):
        for value, (bground_colour, text_colour) in self.value_colour_map.items():
            if x == value:
                return f'background-color: {bground_colour}; color: {text_colour}'
        return ''


class BackgroundGradient(_BaseStylerFormatter):

    def __init__(self,
                 background_gradient_kwargs):
        self.background_gradient_kwargs = background_gradient_kwargs

    def transform(self,
                  styler: Styler):
        return styler.background_gradient(**self.background_gradient_kwargs)


class FormatDuration(_BaseStylerFormatter):
    def __init__(self,
                 col_name,
                 from_type: Literal['minutes', 'timedelta64[ns]']):
        self.col_name = col_name
        self.from_type = from_type

    def transform(self,
                  styler: Styler):
        if self.from_type == 'timedelta64[ns]':
            return styler.applymap(self._format_duration, subset=[self.col_name])
        elif self.from_type == 'minutes':
            return styler.applymap(self._format_duration_minutes, subset=[self.col_name])

    def _format_duration(self,
                         td):
        """
        Format a pandas Timedelta (e.g. 0 days 00:25:27) to HH:MM:SS string.
        """
        # Convert Timedelta to 'HH:MM:SS'
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _format_duration_minutes(self,
                                 minutes):
        """
        Format a number of minutes (float, e.g. 5.5) to MM:SS string.
        """
        # Convert minutes to 'MM:SS'
        minutes_int = int(minutes)
        seconds = int((minutes - minutes_int) * 60)
        return f"{minutes_int:02d}:{seconds:02d}"


class FormatPercentage(_BaseStylerFormatter):
    def __init__(self,
                 round_to=2,
                 **applymap_kwargs):
        self.round_to = round_to
        self.applymap_kwargs = applymap_kwargs

    def transform(self,
                  styler: Styler):
        if self.round_to == 0:
            num_format = '0%'
        else:
            zeroes = ''.join(self.round_to * ['0'])
            num_format = f'0.{zeroes}%'
        return styler.applymap(lambda s: f'number-format: {num_format}', **self.applymap_kwargs)


class FormatNumber:
    """
    A Styler formatter that applies Excel-like custom number formats,
    including suffixes like "k", "M", "B", "T" for thousands, millions, etc.

    Parameters
    ----------
    round_to: int
        The number of decimal places to display.
    rounding: {'normal', 'thousands', 'millions', 'billions', 'trillions'}
        How to scale and label the numbers.
    applymap_kwargs: dict
        Optional additional arguments to pass to Styler.applymap.
    """

    def __init__(
        self,
        round_to: int = 2,
        rounding: Literal['normal', 'thousands',
                          'millions', 'billions', 'trillions'] = 'normal',
        **applymap_kwargs
    ):
        self.round_to = round_to
        self.rounding = rounding
        self.applymap_kwargs = applymap_kwargs

    def transform(self, styler: Styler) -> Styler:
        # Build the decimal part, e.g. if round_to=2 => .00
        decimal_part = ''
        if self.round_to > 0:
            decimal_part = '.' + ('0' * self.round_to)

        # Map each rounding option to:
        #   - how many commas to insert (which divides by 1,000 each time)
        #   - what suffix to show
        suffix_map = {
            'thousands': (1, 'k'),
            'millions':  (2, 'M'),
            'billions':  (3, 'B'),
            'trillions': (4, 'T')
        }

        if self.rounding == 'normal':
            # No extra commas or suffix; just a standard format
            # Example: #,##0.00 for 2 decimals
            num_format = f"#,##0{decimal_part}"
        else:
            commas, suffix = suffix_map[self.rounding]
            # For thousands => #,##0.00,"k"
            # For millions => #,##0.00,,"M"
            # etc.
            num_format = f"#,##0{decimal_part}{',' * commas}\"{suffix}\""

        # Apply the format to each cell via a lambda that returns the CSS property
        return styler.applymap(lambda _: f'number-format: {num_format}', **self.applymap_kwargs)


class FormatHyperlink:
    """
    A Styler formatter that inserts Excel-formula hyperlinks based on two columns:
      1) A "text" column (the text that appears in Excel).
      2) A "url" column (the target URL).

    This class modifies the given text column to contain an `=HYPERLINK(...)` formula
    referencing the URL column.

    Parameters
    ----------
    text_col : str
        Name of the column containing the display text.
    url_col : str
        Name of the column containing the full URL.
    target_col : Optional[str]
        If provided, a new column with this name will be created in place of overwriting
        the text_col. Otherwise, text_col is replaced with a hyperlink formula in-place.
    """

    def __init__(self,
                 text_col: str,
                 url_col: str,
                 target_col: str = None):
        self.text_col = text_col
        self.url_col = url_col
        self.target_col = target_col

    def transform(self, styler: Styler) -> Styler:
        """
        Replaces values in `text_col` (or a new `target_col`) with
        =HYPERLINK("url", "text") formulas so that Excel will render
        them as clickable links.
        """
        df = styler.data  # The DataFrame behind the Styler
        # df = df.reindex(
        #     df.columns.drop(
        #         self.url_col, errors='ignore').tolist() + [self.url_col],
        #     axis=1
        # )

        # If user asked to write into a separate column, create one if it doesn't exist:
        if self.target_col:
            df[self.target_col] = df[self.text_col]

        target_column_name = self.target_col or self.text_col

        # Build a hyperlink formula in each row
        for idx, row in df.iterrows():
            text_val = row[self.text_col]
            url_val = row[self.url_col]

            # Skip if either is missing
            if pd.isna(text_val) or pd.isna(url_val):
                continue

            # If the text or URL might contain double quotes, you should
            # escape them with double-double quotes (" -> "") to keep
            # the formula valid.
            text_escaped = str(text_val).replace('"', '""')
            url_escaped = str(url_val).replace('"', '""')

            # Insert Excel HYPERLINK formula
            # e.g. =HYPERLINK("https://example.com", "Click me")
            df.at[
                idx, target_column_name
            ] = f'=HYPERLINK("{url_escaped}", "{text_escaped}")'

        # df.drop([self.url_col], axis=1, inplace=True)
        # Return the original Styler, now with formulas in the relevant column
        return styler


class FormatHeaders:
    """
    A Styler transformer that sets a background color and optional font styling
    for column headers in Excel (and HTML) output.

    This uses pandas ≥ 1.4's .applymap_index(..., axis="columns") method,
    which applies CSS properties to the header cells.

    Parameters
    ----------
    bg_color : str, optional
        A valid CSS color string for the header background,
        e.g. "yellow", "#FFFF00", "rgb(255,255,0)", etc.
    font_family : str, optional
        The font family for the header text, e.g. "Arial" or "Calibri".
    font_size : str or int, optional
        The font size for the header text, e.g. "10pt".
        If you pass an int (like 10), we’ll convert it to "10pt".
    font_color : str, optional
        A valid CSS color string for the header text color,
        e.g. "black", "#000000", etc.
    font_weight : str, optional
        A CSS font-weight, e.g. "bold".
    """

    def __init__(self,
                 bg_color: str = "#FFFF00",
                 font_family: str = None,
                 font_size: str = None,
                 font_color: str = None,
                 font_weight: str = None,
                 align: str = "center",
                 valign: str = "middle"):
        self.bg_color = bg_color
        self.font_family = font_family
        self.font_size = font_size
        self.font_color = font_color
        self.font_weight = font_weight
        self.align = align
        self.valign = valign

    def _style_func(self, _):
        """
        Returns the CSS declarations for each header cell.
        We ignore the header text/content (_), since we're
        styling all headers uniformly.
        """
        props = []

        # Background color
        if self.bg_color:
            props.append(f"background-color: {self.bg_color}")

        # Font color
        if self.font_color:
            props.append(f"color: {self.font_color}")

        # Font family
        if self.font_family:
            props.append(f"font-family: {self.font_family}")

        # Font size
        if self.font_size:
            # If user passed an int, convert to e.g. "10pt"
            size_str = f"{self.font_size}pt" if isinstance(
                self.font_size, int) else self.font_size
            props.append(f"font-size: {size_str}")

        # Font weight
        if self.font_weight:
            props.append(f"font-weight: {self.font_weight}")

        # Horizontal text alignment
        if self.align in ["left", "center", "right"]:
            props.append(f"text-align: {self.align}")

        # Vertical text alignment
        if self.valign in ["top", "middle", "bottom"]:
            props.append(f"vertical-align: {self.valign}")

        # Join them into one CSS declaration
        return "; ".join(props)

    def transform(self, styler: Styler) -> Styler:
        """
        Applies the header formatting to column headers.
        """
        # axis="columns" means apply to the top header row for each column
        return styler.applymap_index(self._style_func, axis="columns")


class SetProperties:
    def __init__(self, properties):
        self.properties = properties

    def transform(self, styler: Styler) -> Styler:
        return styler.set_properties(**self.properties)
