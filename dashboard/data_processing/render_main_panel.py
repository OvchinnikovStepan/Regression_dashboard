import streamlit as st
import pandas as pd
import asyncio
from dashboard.data_processing.info_about_feature import info_about_feature
from dashboard.visualization.show_hist import show_hist
from dashboard.visualization.show_scatter import show_scatter

def render_main_panel(df: pd.DataFrame):
    """
    Основная панель: График распределения целевого параметра и график зависимости целевого параметра от другого.
    Возвращает training_df для передачи в панель управления прогнозом
    """
    training_df = None

    if df is not None and not df.empty:
        aim_cols = st.columns([4, 8])
        filtered_df = st.session_state.get('filtered_df', df)
        selected_sensors = st.session_state.get('selected_sensors', df.columns.tolist())

        with aim_cols[0]:
            selected_aim_option = st.selectbox(
                "Выберите целевой параметр",
                options=selected_sensors,
                key="aim_sensor_select"
            )

        with aim_cols[1]:
            mean, median, std, minimal, maximum = info_about_feature(filtered_df, selected_aim_option)
            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
            with col1:
                st.subheader("Среднее")
                st.write(f"{mean:.3f}")
            with col2:
                st.subheader("СКО")
                st.write(f"{std:.3f}")
            with col3:
                st.subheader("Медиана")
                st.write(f"{median:.3f}")
            with col4:
                st.subheader("Мин. знач.")
                st.write(f"{minimal:.3f}")
            with col5:
                st.subheader("Макс. знач.")
                st.write(f"{maximum:.3f}")

        show_hist(df, selected_aim_option, bins=100)

        # Формируем доступные колонки для второго selectbox (исключая выбранный в первом)
        available_second_options = [col for col in selected_sensors if col != selected_aim_option]

        second_cols = st.columns([4, 8])
        with second_cols[0]:
            selected_second_option = st.selectbox(
                "Выберите вторичный параметр",
                options=available_second_options,
                key="second_sensor_select"
            )

        if selected_aim_option == selected_second_option:
            with second_cols[1]:
                st.warning("Параметры не должны совпадать")
        else:
            show_scatter(df, [selected_aim_option, selected_second_option])

    else:
        st.markdown("""<div class=\"block\" style=\"height: 200px;\"></div>""", unsafe_allow_html=True)

    return training_df