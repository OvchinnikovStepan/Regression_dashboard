import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from typing import List, Optional
from dashboard.utils.validate_data import validate_data


def build_scatter_plot(df: pd.DataFrame, columns: List[str]) -> Optional[plt.Figure]:
    """
    Строит scatter plot только между первыми двумя колонками из списка.
    """
    if not validate_data(df):
        return None

    if len(columns) < 2:
        st.warning("Для построения графика нужно выбрать минимум 2 колонки.")
        return None

    x_col, y_col = columns[0], columns[1]

    with st.spinner("Строим график..."):
        fig, ax = plt.subplots(figsize=(23, 4))

        # Строим график рассеяния
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, alpha=0.7)

        # Заголовок и подписи осей
        ax.set_title(f'Зависимость {y_col} от {x_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)

        # Размер шрифта тиков
        ax.tick_params(axis='both', labelsize=10)

    return fig


def show_scatter(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Основная функция для отображения одного парного графика в Streamlit.
    """
    if not validate_data(df):
        return

    fig = build_scatter_plot(df, columns)

    if fig is not None:
        st.pyplot(fig)