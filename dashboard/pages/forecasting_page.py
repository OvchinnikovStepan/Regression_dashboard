import streamlit as st
import pandas as pd
import asyncio
from dashboard.data_processing.info_about_dataframe import info_about_dataframe
from dashboard.data_processing.render_main_panel import render_main_panel
from dashboard.processing.linearRegressionModel import LinearRegressionModel
from dashboard.data_processing.select_time_interval import start_date, end_date, filter_dataframe
from dashboard.visualization.plot_interactive_with_selection import plot_interactive_with_selection
from API.app.request_functions.create_model_payload_func import create_model_payload
from API.app.request_functions.model_request_func import get_prediction
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



def render_prediction_func(df: pd.DataFrame,):
    st.markdown("<h3 style='text-align: center;'>Вывод формулы</h3>", unsafe_allow_html=True)

    selected_sensors = st.session_state.get('selected_sensors', df.columns.tolist())
    aim_cols = st.columns([4, 8])
    with aim_cols[0]:
        selected_aim_option = st.selectbox("Выберите целевой параметр", selected_sensors)
    with aim_cols[1]:
        second_options = st.multiselect(
        "Выберите признаки для построения уравнения:",
        options=df.columns.tolist(),
        )
    model = LinearRegressionModel()
    model.fit(df[second_options], df[selected_aim_option])
    st.markdown(model.get_equation(latex_output=True))

def render_forecasting_page(df: pd.DataFrame, outlier_percentage: float) -> None:
    """
    Рендерит страницу "Прогнозирование"
    """
    st.set_page_config(page_title="Прогнозирование", layout="wide")
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