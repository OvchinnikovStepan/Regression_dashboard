import pandas as pd
from typing import Tuple, Optional, List


def get_date_formats() -> List[str]:
    """Возвращает список поддерживаемых форматов дат"""
    return [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%d.%m.%Y %H:%M:%S',
        '%d.%m.%Y %H:%M',
        '%Y/%m/%d %H:%M:%S',
        '%Y/%m/%d %H:%M',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y %H:%M',
        '%Y-%m-%d',
        '%d.%m.%Y',
        '%Y/%m/%d',
        '%m/%d/%Y'
    ]


def is_date_column(column_data: pd.Series, sample_size: int = 10) -> Tuple[bool, Optional[str]]:
    """
    Проверяет, является ли столбец датой
    
    Args:
        column_data: Данные столбца
        sample_size: Размер выборки для проверки
        
    Returns:
        Tuple[bool, Optional[str]]: (является_датой, найденный_формат)
    """
    date_formats = get_date_formats()
    sample = column_data.astype(str).head(sample_size).dropna()
    
    for date_format in date_formats:
        try:
            pd.to_datetime(sample, format=date_format, errors='raise')
            return True, date_format
        except:
            continue
    return False, None
