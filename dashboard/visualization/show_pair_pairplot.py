import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from typing import List, Optional
from dashboard.utils.validate_data import validate_data


def build_pair_pairplot(df: pd.DataFrame, columns: List[str]) -> Optional[sns.axisgrid.PairGrid]:
    """
    Строит pairplot по выбранным признакам.
    """
    
    with st.spinner("Строим pairplot..."):
        # Pairplot с настройками размеров
        pairplot = sns.pairplot(
            data=df[columns],
            height=1.5,     # высота каждого подграфика
            aspect=1.5      # ширина = height * aspect => делает графики шире
        )

        # Настройка общего размера фигуры
        pairplot.fig.set_size_inches(20 , 4)  # ширина, высота всей фигуры
        pairplot.fig.suptitle('Матрица взаимосвязей параметров', fontsize=14, fontweight='bold', y =1.02)
        for ax in pairplot.axes.flat:
            if ax is not None:
                # Подписи осей (xlabel, ylabel)
                ax.set_xlabel(ax.get_xlabel(), fontsize=12)
                ax.set_ylabel(ax.get_ylabel(), fontsize=12)
                
                # Метки тиков (числа на осях)
                for tick in ax.get_xticklabels():
                    tick.set_fontsize(7)
                for tick in ax.get_yticklabels():
                    tick.set_fontsize(7)

        return pairplot


def show_pair_pairplot(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Основная функция для отображения pairplot в Streamlit.
    """
    if not validate_data(df):
        return

    pairplot = build_pair_pairplot(df, columns)

    if pairplot is not None:
        # Выводим график в центральной части страницы
        st.pyplot(pairplot.fig)