import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sympy import symbols, Eq, latex
import sympy as sp
import pandas as pd

class LinearRegressionModel:
    def __init__(self):
        """Инициализация модели линейной регрессии"""
        self.model = LinearRegression()
        self.X = None
        self.y = None
        self.coef_ = None
        self.intercept_ = None
        self.equation = None
        self.feature_names = None
        self.target_name = None
        
    def fit(self, X, y, feature_names=None, target_name=None):
        """
        Обучение модели на данных
        
        Параметры:
        X - признаки (2D массив или DataFrame)
        y - целевая переменная (1D массив или DataFrame)
        feature_names - список названий признаков (Для уравнения)
        target_name - название целевой переменной (Для уравнения)
        """
        # Преобразование входных данных
        self.X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        self.y = y.values.ravel() if isinstance(y, pd.DataFrame) else np.array(y).ravel()
        
        # Убедимся, что X двумерный даже для одномерного случая
        if len(self.X.shape) == 1:
            self.X = self.X.reshape(-1, 1)
        
        # Установка имен признаков
        if feature_names:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f'x{i}' for i in range(self.X.shape[1])]
        
        # Заменяем пробелы в именах признаков на подчеркивания
        self.feature_names = [name.replace(" ", "-") for name in self.feature_names]

        # Установка имени целевой переменной
        if target_name:
            self.target_name = target_name.replace(" ", "-")
        elif isinstance(y, pd.DataFrame):
            self.target_name = list(y.columns)[0].replace(" ", "-")
        else:
            self.target_name = 'y'

        # Обучение модели
        self.model.fit(self.X, self.y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        # Создаем символьное уравнение
        self._create_equation()
        
    def predict(self, X):
        """Предсказание значений по модели"""
        X_array = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        # Убедимся, что X двумерный
        if len(X_array.shape) == 1:
            X_array = X_array.reshape(-1, 1)
        return self.model.predict(X_array)
    
    def evaluate(self, X_test=None, y_test=None):
        """
        Расчет метрик качества модели
        
        Параметры:
        X_test - тестовые признаки (если None, используется обучающая выборка)
        y_test - тестовые значения (если None, используется обучающая выборка)
        
        Возвращает:
        Словарь с метриками
        """
        if X_test is None or y_test is None:
            X_test, y_test = self.X, self.y
        else:
            X_test = X_test.values if isinstance(X_test, pd.DataFrame) else np.array(X_test)
            y_test = y_test.values.ravel() if isinstance(y_test, pd.DataFrame) else np.array(y_test).ravel()
        
        # Убедимся, что X двумерный
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(-1, 1)
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def _create_equation(self):
        """Создание символьного уравнения регрессии"""
        # Создаем символы для переменных
        vars = symbols(' '.join(self.feature_names))
        if len(self.feature_names) == 1:
            vars = [vars]  # Для случая с одной переменной
            
        # Создаем уравнение, начиная с intercept как sympy-выражения

        # Округляем коэффициенты до 5 знаков
        equation = sp.Float(float(self.intercept_), 4)
        for coef, var in zip(self.coef_, vars):
            equation += sp.Float(float(coef), 4) * var
            
        self.equation = equation
    
    def get_equation(self, latex_output=False):
        """
        Получение уравнения регрессии в красивом формате
        
        Параметры:
        latex_output - если True, возвращает LaTeX представление
        
        Возвращает:
        Строку с уравнением или LaTeX строку
        """
        if self.equation is None:
            return "Модель еще не обучена"
        y_symbol = symbols(self.target_name)
        eq = Eq(y_symbol, self.equation)
        if latex_output:
            return latex(eq)
        return str(eq)
    
    def __str__(self):
        """Строковое представление модели"""
        if self.equation is None:
            return "LinearRegressionModel (not fitted yet)"
            
        info = "Linear Regression Model\n"
        info += f"Equation: {self.get_equation()}\n"
        
        if self.X is not None:
            metrics = self.evaluate()
            info += "Metrics on training data:\n"
            for name, value in metrics.items():
                info += f"  {name}: {value:.4f}\n"
                
        return info