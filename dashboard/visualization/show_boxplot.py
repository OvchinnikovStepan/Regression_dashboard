import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from typing import List
from dashboard.utils.validate_data import validate_data


def select_boxplot_features(df: pd.DataFrame) -> List[str]:
    """
    UI-компонент для выбора признаков для boxplot
    """
    if df.shape[1] > 10:
        st.warning("Слишком много признаков. График может быть трудночитаем. Рекомендуется выбрать не более 5 признаков.")
    
    return st.multiselect(
        "Выберите признаки для построения boxplot:",
        options=df.select_dtypes(include='number').columns.tolist(),  # Только числовые
        default=df.select_dtypes(include='number').columns[:5].tolist()  # Авто-выбор первых 5
    )


def build_boxplot(df: pd.DataFrame, columns: List[str]):
    """
    Строит boxplot по выбранным признакам.
    """
    if not columns:
        st.info("Не выбрано ни одного числового признака для отображения boxplot.")
        return None

    with st.spinner("Строим boxplot..."):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Фильтруем только нужные столбцы
        data = df[columns]

        # Строим boxplot
        sns.boxplot(data=data, ax=ax, palette="Set2")

        # Заголовок
        ax.set_title("Распределение параметров через boxplot", fontsize=14, fontweight='bold')

        # Подписи осей
        ax.set_xlabel("Параметры", fontsize=12)
        ax.set_ylabel("Значения", fontsize=12)

        # Размер меток на осях
        ax.tick_params(axis='both', which='major', labelsize=7)

        # Сетка
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')

        # Убираем рамку
        sns.despine()

        return fig


def show_boxplot(df: pd.DataFrame) -> None:
    """
    Основная функция для отображения boxplot в Streamlit.
    Использует UI-селектор признаков и строит график.
    """
    if not validate_data(df):
        return

    # Шаг 1: Выбор признаков
    selected_features = select_boxplot_features(df)

    # Шаг 2: Построение графика
    if selected_features:
        fig = build_boxplot(df, selected_features)
        if fig is not None:
            st.pyplot(fig)
    else:
        st.info("Выберите хотя бы один числовой признак для построения boxplot.")