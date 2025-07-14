import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def build_hist_figure(df: pd.DataFrame, feature: str, bins: int = 50):
    """
    Строит matplotlib Figure с гистограммой по указанному признаку.
    Добавлено стилистическое оформление для лучшего визуального восприятия.
    """
    if feature not in df.columns:
        return None

    data = df[feature]

    fig, ax = plt.subplots(figsize=(20, 4))
    
    # Гистограмма с KDE линией
    ax.hist(data, bins=bins, edgecolor='black', color='skyblue', alpha=0.8)

    # Настройки заголовков и меток
    ax.set_title(f"Гистограмма: {feature}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Значение", fontsize=12)
    ax.set_ylabel("Частота", fontsize=12)

  
    # Сетка
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')

    # Оптимизация отступов
    plt.tight_layout()

    return fig


def show_hist(df: pd.DataFrame, selected_feature: str, bins: int = 50) -> None:
    """
    Отображает гистограмму выбранного признака через Streamlit.
    Если признак не найден — выводит предупреждение.
    """
    fig = build_hist_figure(df, selected_feature, bins)

    if fig is not None:
        st.pyplot(fig)
    else:
        st.warning(f"Признак '{selected_feature}' не найден в DataFrame.")