# Regression_dashboard

## Описание

**Regression_dashboard** - проект по созданию информационной панели для анализа временных рядов и прогнозов


### Технологии
- **Язык:** Python  
- **Фреймворк:** Streamlit  
- **Библиотеки:** Pandas, Statsmodels, NumPy
- **База данных:** Не используется (операции выполняются в памяти)

---

## Запуск проекта

**Порт:** `8501`  
**URL:** `http://localhost:8501`

Команда запуска:
```bash
streamlit run main.py --server.port 8501
``` 

---

## Docker
Для создания образа, в корневой папке прописать:
```bash
docker build -t regression .
```
Для запуска, там же прописать:
```bash
docker run -p8501:8501 regression
````

---
