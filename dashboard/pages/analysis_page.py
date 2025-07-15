import streamlit as st
import pandas as pd
from dashboard.data_processing.info_about_dataframe import info_about_dataframe
from dashboard.data_processing.render_main_panel import render_main_panel
from dashboard.visualization.show_heatmap import show_heatmap
from dashboard.visualization.show_boxplot import show_boxplot
from dashboard.data_processing.info_about_feature import info_about_feature
from dashboard.visualization.show_pairplot import show_pairplot
from dashboard.data_processing.forecasting import forecasting
from dashboard.visualization.show_hist import show_hist
from dashboard.visualization.show_autocorrelation import show_autocorrelation
from dashboard.utils.data_limiting import limit_data_to_last_points
from typing import Optional, List, Tuple

def render_data_overview(df: pd.DataFrame, outlier_percentage: float) -> None:
    """
    Отображает верхнюю панель с общей информацией о данных
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

def handle_filter_buttons(df: pd.DataFrame) -> None:
    """
    Обрабатывает кнопки фильтрации и сброса фильтра
    """
    button_cols = st.columns(2)
    with button_cols[0]:
        if st.button("Применить фильтр", key="apply_filter_analysis"):
            if 'sensor_editor_temp' in st.session_state:
                if st.session_state['sensor_editor_temp']:
                    st.session_state['selected_sensors'] = st.session_state['sensor_editor_temp']
                    st.session_state['filtered_df'] = df[st.session_state['selected_sensors']]
                else:
                    st.error("Ошибка: Выберите хотя бы один параметр для отображения графика.")
                    st.session_state['selected_sensors'] = []
            else:
                st.session_state['selected_sensors'] = df.columns.tolist()
                st.session_state['filtered_df'] = df
            st.rerun()
    with button_cols[1]:
        if st.button("Сбросить фильтр", key="reset_filter_analysis"):
            # Возвращаемся к ограниченному виду (последние 500 точек)
            if st.session_state.get('is_limited_view_analysis', False) and st.session_state.get('original_df_analysis') is not None:
                limited_df = limit_data_to_last_points(st.session_state['original_df_analysis'], 500)
                st.session_state['filtered_df'] = limited_df
            else:
                st.session_state['filtered_df'] = df
            st.session_state['selected_sensors'] = df.columns.tolist()
            st.session_state['sensor_editor_temp'] = df.columns.tolist()
            st.rerun()

def render_parameter_and_preview_panel(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """
    Отображает панель параметров и предпросмотра
    """
    lower_cols = st.columns([6, 6])
    with lower_cols[1]:
        st.markdown("#### Параметры:")
        if df is not None and not df.empty:
            sensor_df = pd.DataFrame({
                'Датчики': df.columns,
                'Отображать': [col in st.session_state.get('selected_sensors', df.columns.tolist()) for col in df.columns]
            })
            edited_sensor_df = st.data_editor(sensor_df, key="sensor_selector_analysis")
            st.session_state['sensor_editor_temp'] = edited_sensor_df[edited_sensor_df['Отображать']]['Датчики'].tolist()
        else:
            st.markdown("Нет информации", unsafe_allow_html=True)
    with lower_cols[0]:
        st.markdown("#### Предпросмотр:")
        if df is not None:
            st.dataframe(filtered_df)
        else:
            st.markdown("Нет информации", unsafe_allow_html=True)

def render_heatmap_pairplot_panel(df: pd.DataFrame) -> None:
    """
    Отображает панель heatmap и pairplot
    """
   
    col_heat, col_pair = st.columns(2)
    with col_heat:
        st.markdown("### Heatmap")
        show_heatmap(df)
    with col_pair:
        st.markdown("### Pairplot")
        show_pairplot(df)
    col_box = st.columns([3,6,3])
    with col_box[1]:
        st.markdown("### Boxplot")
        show_boxplot(df)

def render_sensor_statistics_panel(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """
    Отображает панель статистики датчиков, прогнозирования, гистограммы и автокорреляции
    """
    st.markdown("### Статистика датчиков")
    features = filtered_df.columns.tolist() if df is not None and not df.empty else []
    if not features:
        st.info("Нет признаков для анализа.")
        return
    selected_feature = st.selectbox("Выберите признак для подробной информации о нём", features, index=0)
    if selected_feature:
        mean, median, std, minimal, maximum = info_about_feature(filtered_df, selected_feature)
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
        forecasting(df, column=selected_feature)
        col_hist, col_autocorr = st.columns(2)
        with col_hist:
            show_hist(df, selected_feature, bins=100)
        with col_autocorr:
            show_autocorrelation(df, selected_feature)


def render_analysis_page(df: pd.DataFrame, outlier_percentage: float) -> None:
    """
    Рендерит страницу "Анализ данных"
    """
    st.title("Анализ данных")
    render_data_overview(df, outlier_percentage)
    
    # Инициализация данных при загрузке нового файла
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
    render_heatmap_pairplot_panel(df)