import streamlit as st
import pandas as pd
import asyncio
from dashboard.data_processing.info_about_dataframe import info_about_dataframe
from dashboard.data_processing.render_main_panel import render_main_panel
from dashboard.processing.linearRegressionModel import LinearRegressionModel
from dashboard.utils.data_limiting import limit_data_to_last_points
from typing import Optional, List

def render_data_overview(df: pd.DataFrame, outlier_percentage: float) -> None:
    """
    Отображает верхнюю панель с общей информацией о данных и предпросмотром DataFrame
    """
    features_size, tuples_size, = info_about_dataframe(df)
    top_cols = st.columns([4, 8])
    with top_cols[0]:
        st.markdown("#### Информация о файле:")
        st.markdown("")
        st.markdown(f"##### Кол-во записей: {tuples_size if tuples_size is not None else 'Нет информации'}")
        st.markdown(f"##### Количество признаков: {features_size if features_size is not None else 'Нет информации'}")
        st.markdown(f"##### Количество выбросов: {f'{outlier_percentage}% от всех значений' if outlier_percentage is not None else 'Нет информации'}")
    with top_cols[1]:
        st.markdown("#### Предпросмотр:")
        filtered_df = st.session_state.get('filtered_df', df)
        if filtered_df is not None:
            st.dataframe(filtered_df)
        else:
            st.markdown("Нет информации", unsafe_allow_html=True)



def render_prediction_func(df: pd.DataFrame):
    st.markdown("<h3 style='text-align: center;'>Вывод формулы</h3>", unsafe_allow_html=True)

    selected_sensors = st.session_state.get('selected_sensors', df.columns.tolist())
    equation_cols = st.columns([4,4])

    with equation_cols[0]:
        # Выбор целевого параметра
        selected_aim_option = st.selectbox(
            "Выберите целевой параметр",
            options=selected_sensors,
            key="selected_aim_option"
        )

    # Получаем доступные колонки (без целевого)
    available_features = [col for col in selected_sensors if col != selected_aim_option]

    with equation_cols[1]:
        # Multiselect с фильтром
        second_options = st.multiselect(
            "Выберите признаки для построения уравнения:",
            options=available_features,
            default=st.session_state.get("selected_features", []),
            key="selected_features"  # Это ключевой момент — Streamlit сам управляет session_state
        )

    # Обучение модели и вывод формулы
    if second_options:
        with equation_cols[0]:
            model = LinearRegressionModel()
            model.fit(df[second_options], df[selected_aim_option],target_name=selected_aim_option)
            equation = model.get_equation(latex_output=True)
            st.markdown(f"<div style='font-size:40px;'>{equation}</div>", unsafe_allow_html=True)
        with equation_cols[1]:
            st.markdown("<h3 style='text-align: center;'>Точечное предсказание</h3>", unsafe_allow_html=True)
    else:
        st.info("Выберите хотя бы один признак для построения регрессии.")

    
    
    

def render_forecasting_page(df: pd.DataFrame, outlier_percentage: float) -> None:
    """
    Рендерит страницу "Прогнозирование"
    """
    st.title("Прогнозирование")
    render_data_overview(df, outlier_percentage)
    if df is not None:
        current_df_hash = hash(pd.util.hash_pandas_object(df, index=True).sum())
        if st.session_state.get('last_df_hash') != current_df_hash:
            st.session_state.clear()
            # При загрузке нового файла ограничиваем данные последними 500 точками
            limited_df = limit_data_to_last_points(df, 500)
            st.session_state['filtered_df'] = limited_df
            st.session_state['selected_sensors'] = df.columns.tolist()
            st.session_state['sensor_editor_temp'] = df.columns.tolist()
            st.session_state['target_sensor'] = df.columns[0]
            st.session_state['last_df_hash'] = current_df_hash
            st.session_state['original_df'] = df  # Сохраняем оригинальный DataFrame
            st.session_state['is_limited_view'] = True  # Флаг, что отображается ограниченный вид
        render_main_panel(df)
        render_prediction_func(df)