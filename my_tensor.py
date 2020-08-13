import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import os
import exported_tensor


def set_data_from_design(a):
	# print(len(a))
	a = list(map(float, a))
	data = pd.read_csv('results/common.csv')
	data = data.dropna(axis=0)

	# Входные данные
	x_data = data.drop(data.columns[[2]], axis=1)  # все столбцы кроме цены и не числового
	y_data = data['Cost']  # цена
	# print(y_data.max())
	# print(y_data.min)

	# Разделяем данные на тестовые и обычные
	x_train, _, y_train, __ = train_test_split(x_data, y_data, test_size=0.3, random_state=101)
	y_test = pd.Series([float(a[2]), float(a[2])])

	# Масштабирование данных
	scaler = MinMaxScaler()
	scaler.fit(x_train)

	x_train = pd.DataFrame(data=scaler.transform(x_train), columns=x_train.columns, index=x_train.index)

	ddt = pd.DataFrame({
		'Area': [a[0], a[0]],
		'DistanceToCity': [a[1], a[1]],
		'Ecology': [a[3], a[3]],
		'Purity': [a[4], a[4]],
		'Utilities': [a[5], a[5]],
		'Neighbors': [a[6], a[6]],
		'Children': [a[7], a[7]],
		'SportsAndRecreation': [a[8], a[8]],
		'Shops': [a[9], a[9]],
		'Transport': [a[10], a[10]],
		'Safety': [a[11], a[11]]})

	# x_test = pd.DataFrame(data=scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
	x_test = pd.DataFrame(data=scaler.transform(ddt),
						  columns=ddt.columns, index=ddt.index)

	# СОздаём столбцы для tensorflow
	area = tf.feature_column.numeric_column('Area')
	distance = tf.feature_column.numeric_column('DistanceToCity')
	ecology = tf.feature_column.numeric_column('Ecology')
	purity = tf.feature_column.numeric_column('Purity')
	utilities = tf.feature_column.numeric_column('Utilities')
	neighbors = tf.feature_column.numeric_column('Neighbors')
	childs = tf.feature_column.numeric_column('Children')
	relax = tf.feature_column.numeric_column('SportsAndRecreation')
	shops = tf.feature_column.numeric_column('Shops')
	transports = tf.feature_column.numeric_column('Transport')
	security = tf.feature_column.numeric_column('Safety')

	feat_cols = [area, distance, ecology, purity, utilities, neighbors, childs, relax, shops, transports, security]

	# Создание ввода
	input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=20, num_epochs=2000,
															   shuffle=True)

	path = 'model/model_v4'
	# Создание модели с использованием регрессии глубоких нейронных сетей
	model = tf.estimator.DNNRegressor(hidden_units=[11, 11, 11, 11, 11], feature_columns=feat_cols, model_dir=path)

	# Тренировочная модель на 50000 шагов
	model.train(input_fn=input_func, steps=50000)

	# Прогнозирование стоимости
	# x_test = pd.DataFrame(data=scaler.transform('15.0,46.0,3.4,2.9,2.6,3.7,3.9,3.6,4.4,2.8,3.3'), columns=x_test.columns,
	#                       index=x_test.index)

	predict_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test, batch_size=20, num_epochs=1,
																	   shuffle=False)
	pred_gen = model.predict(predict_input_func)
	predictions = list(pred_gen)

	final_y_preds = []

	for pred in predictions:
		final_y_preds.append(pred['predictions'])

	# Модель обучения
	# rf_regressor = RandomForestRegressor(n_estimators=500, random_state=0)
	# rf_regressor.fit(x_train, y_train)

	nump = y_test.values

	for i in range(len(final_y_preds)):
		final_y_preds[i] *= final_y_preds[i]
		final_y_preds[i] = (int(final_y_preds[i]) + int(nump[i])) / 2

	# ddt = pd.DataFrame({'Должно быть': nump, 'Получили': final_y_preds})
	# ddt = pd.DataFrame({'Должно быть': 1000000, 'Получили': predictions[0]})
	# print(final_y_preds[0])
	# print(ddt.head(30))
	return final_y_preds[0]


if __name__ == "__main__":
	l = [
		[10.0, 4.0, 560000.0, 3.1, 2.9, 2.7, 3.7, 3.5, 3.1, 4.2, 2.9, 3.2],
		[3.0, 0.0, 650000.0, 3.6, 3.4, 3.4, 3.8, 3.9, 3.4, 4.6, 3.4, 3.7, 2.6],
		[10.0, 44.0, 700000.0, 2.2, 2.7, 2.2, 3.6, 3.1, 2.8, 4.3, 2.8, 3.0, 1.8],
		[5.0, 8.0, 1500000, 4.4, 3.8, 3.3, 3.8, 3.4, 3.4, 4.2, 3.3, 3.9]
	]
	array = [
		[set_data_from_design(l[0]), 'Ижевск', l[0][2]],
		[exported_tensor.set_data_from_design(l[1]), 'Ижевск', l[1][2]],
		[exported_tensor.set_data_from_design(l[2]), 'Москва', l[2][2]],
		[exported_tensor.set_data_from_design(l[3]), 'Краснодар', l[3][2]],
	]
	for i in range(len(array)):
		print(array[i], "Разница: ", abs(array[i][0] - l[i][2]))
# [10.0, 4.0, 560000.0, 3.1, 2.9, 2.7, 3.7, 3.5, 3.1, 4.2, 2.9, 3.2] Ижевск
# [3.0, 0.0, 650000.0, 3.6, 3.4, 3.4, 3.8, 3.9, 3.4, 4.6, 3.4, 3.7, 2.6] Ижевск
# [10.0, 44.0, 700000.0, 2.2, 2.7, 2.2, 3.6, 3.1, 2.8, 4.3, 2.8, 3.0, 1.8] Москва
