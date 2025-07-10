from LinearRegressionModel import LinearRegressionModel
import pandas as pd
X_multi = pd.DataFrame([[1, 2], [2, 4], [3, 1], [4, 3], [5, 5]],columns=["sensor","block"])
y_multi = pd.DataFrame([5, 8, 7, 11, 15],columns=["target"])

model = LinearRegressionModel()
model.fit(X_multi, y_multi)


# Вывод информации о модели
print(model)

# Уравнение в LaTeX формате
print("\nLaTeX уравнение:")
print(model.get_equation(latex_output=True))

# Прогнозирование
print("\nПрогноз для x=6:", model.predict([[6,4]]))

# Оценка модели
print("\nМетрики:", model.evaluate())