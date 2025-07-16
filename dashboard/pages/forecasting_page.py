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
    if second_options:
        model = LinearRegressionModel()
        model.fit(df[second_options], df[selected_aim_option], target_name=selected_aim_option)

         # Сохраняем модель в session_state
        st.session_state['trained_model'] = model
        st.session_state['equation_origin'] = model.get_equation(latex_output=True)

        
        return second_options
    else:
        st.info("Выберите хотябы один призднак для построения уравнения")

def render_equation(option):
    @st.dialog("Заполните значения для признаков")
    def equation_dialog():
        values = {}
        for op in option:
            values[op] = st.number_input(
                f"Введите значение для признака {op}: ",
                value=1.0,
                format="%f"
            )
        if st.button("Подтвердить"):
            st.session_state.equation_values = values
            st.session_state.dialog_submitted = True
            st.session_state.show_dialog = False  # Сбрасываем флаг диалога для его закрытия
            st.rerun()
    
    if st.session_state.get("show_dialog", False):
        equation_dialog()
    
    # Возвращаем значения, если они есть и диалог завершен
    if 'equation_values' in st.session_state and st.session_state.get('dialog_submitted', False):
        return st.session_state.equation_values
    return None


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
            st.rerun
            st.session_state['selected_sensors'] = df.columns.tolist()
            st.session_state['sensor_editor_temp'] = df.columns.tolist()
            st.session_state['target_sensor'] = df.columns[0]
            st.session_state['last_df_hash'] = current_df_hash
            st.session_state['original_df'] = df  # Сохраняем оригинальный DataFrame
        render_main_panel(df)

    if df is not None and not df.empty:
        second_options = render_prediction_func(df)

        col1, col2 = st.columns([9, 3])
        
        values = render_equation(second_options)

        with col1:
            if st.session_state.get('equation_origin'):
                st.markdown(f"""
                    <div style="
                        display: flex;
                        align-items: center;      /* Вертикальное центрирование */
                        justify-content: center;  /* Горизонтальное центрирование */
                        font-size: 24px;
                        padding: 10px;
                        border: 3px solid #ccc;
                        width: 100%;
                        height: 100px;            /* Можно изменить под нужды */
                        box-sizing: border-box;
                        margin-bottom: 10px;
                    ">
                        {st.session_state.equation_origin}
                    </div>
                    """, unsafe_allow_html=True)
                
            sub_col1, sub_col2 = st.columns([2, 2])
            with sub_col1:
                if st.session_state.get('trained_model'):
                    st.markdown("""
                        <style>
                            .stButton button {
                                font-size: 24px !important;
                                padding: 15px 30px !important;
                                height: auto !important;
                                min-height: 50px !important;
                                width: 100% !important;
                            }
                        </style>
                        """, unsafe_allow_html=True)
                    
                    input_btn = st.button("Ввести значения для формулы", key="open_dialog_button")
                    if input_btn and second_options is not None:
                        st.session_state.show_dialog = True
                        st.session_state.dialog_submitted = False
                        st.rerun()

            with sub_col2:
                with sub_col2:
                    selected_features = st.session_state.get("selected_features", [])
                    if values is not None and selected_features:
                        try:
                            # Проверяем, что есть выбранные признаки
                            if not selected_features:
                                st.warning("Не выбрано ни одного признака для предсказания.")
                                return

                            values_list = [values[feature] for feature in selected_features]

                            # Проверяем, что все признаки были переданы
                            if len(values_list) == 0:
                                st.warning("Нет данных для указанных признаков.")
                                return

                            model = st.session_state['trained_model']
                            prediction = model.predict([values_list])[0]

                            st.markdown(f"""
                                <div style="
                                    display: flex;
                                    align-items: center;
                                    justify-content: left;
                                    font-size: 20px;
                                    padding: 10px;
                                    width: 100%;
                                    height: 100px;">
                                    Итог точечного предсказания: {prediction:.2f}
                                </div>
                            """, unsafe_allow_html=True)

                        except Exception as e:
                            st.warning("Вы не выбрали параметры")

        with col2:
            if st.session_state.get('trained_model'):
                model = st.session_state['trained_model']
                metrics = pd.DataFrame([model.evaluate()]).T
                metrics.reset_index(inplace=True)
                metrics.columns = ['Метрика', 'Значение']
                st.dataframe(metrics, use_container_width=True, hide_index=True)