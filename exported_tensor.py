import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def set_data_from_design(a):
	data = pd.read_csv('results/obhiy.csv')
	data = data.dropna(axis=0)
	a = list(map(float, a))
	print(len(a), a)

	# Входные данные
	x_data = data.drop(data.columns[[2]], axis=1)  # все столбцы кроме цены и не числового
	y_data = data['Cost']  # цена

	# Разделяем данные на тестовые и обычные
	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=101)
	del x_test, y_test, y_train

	y_test = pd.Series([a[2], a[2]])

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
		'Safety': [a[11], a[11]],
		'LifeCost': [a[12], a[12]],
		'City': [a[13], a[13]]})

	# Масштабирование данных
	scaler = MinMaxScaler()
	scaler.fit(x_train)

	# print(ddt)
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
	lifecost = tf.feature_column.numeric_column('LifeCost')
	city = tf.feature_column.numeric_column('City')

	feat_cols = [area, distance, ecology, purity, utilities, neighbors, childs, relax, shops, transports, security,
				 lifecost, city]

	# new_model_v4 ведёт себя нормально
	path = 'model/new_model/new_model_v4'

	# Создание модели с использованием регрессии глубоких нейронных сетей
	model = tf.estimator.DNNRegressor(hidden_units=[13, 13, 13, 13, 13], feature_columns=feat_cols, model_dir=path)

	predict_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test, batch_size=20, num_epochs=1,
																	   shuffle=False)
	pred_gen = model.predict(predict_input_func)
	predictions = list(pred_gen)

	final_y_preds = []

	for pred in predictions:
		final_y_preds.append(pred['predictions'])

	nump_array = y_test.values

	for i in range(len(final_y_preds)):
		final_y_preds[i] *= final_y_preds[i]
		final_y_preds[i] = (int(final_y_preds[i]) + int(nump_array[i])) / 2

	return final_y_preds[0]


if __name__ == "__main__":
	city = {
		"1": "Камчатский край",
		"2": "Марий Эл",
		"3": "Чечня",
		"4": "Оренбургская область",
		"5": "Ямало-Ненецкий АО",
		"6": "Забайкальский край",
		"7": "Ярославская область",
		"8": "Владимирская область",
		"9": "Бурятия",
		"10": "Калмыкия",
		"11": "Белгородская область",
		"12": "Вологодская область",
		"13": "Волгоградская область",
		"14": "Калужская область",
		"15": "Ингушетия",
		"16": "Кабардино-Балкария",
		"17": "Иркутская область",
		"18": "Ивановская область",
		"19": "Астраханская область",
		"20": "Карачаево-Черкесия",
		"21": "Новгородская область",
		"22": "Курганская область",
		"23": "Костромская область",
		"24": "Краснодарский край",
		"25": "Магаданская область",
		"26": "Нижегородская область",
		"27": "Кировская область",
		"28": "Липецкая область",
		"29": "Мурманская область",
		"30": "Курская область",
		"31": "Мордовия",
		"32": "Хакасия",
		"33": "Карелия",
		"34": "Якутия",
		"35": "Татарстан",
		"36": "Адыгея",
		"37": "Омская область",
		"38": "Пензенская область",
		"39": "Псковская область",
		"40": "Северная Осетия",
		"41": "Башкортостан",
		"42": "Пермский край",
		"43": "Ростовская область",
		"44": "Дагестан",
		"45": "Приморский край",
		"46": "Орловская область",
		"47": "Томская область",
		"48": "Тверская область",
		"49": "Удмуртия",
		"50": "Ставропольский край",
		"51": "Ульяновская область",
		"52": "Хабаровский край",
		"53": "Смоленская область",
		"54": "Ханты-Мансийский АО",
		"55": "Челябинская область",
		"56": "Самарская область",
		"57": "Тульская область",
		"58": "Тамбовская область",
		"59": "Тюменская область",
		"60": "Свердловская область",
		"61": "Сахалинская область",
		"62": "Рязанская область",
		"63": "Республика Алтай",
		"64": "Чувашия",
		"65": "Чукотский АО",
		"66": "Брянская область",
		"67": "Еврейская АО",
		"68": "Алтайский край",
		"69": "Калининградская область",
		"70": "Архангельская область",
		"71": "Кемеровская область",
		"72": "Амурская область",
		"73": "Воронежская область",
		"74": "Красноярский край",
		"75": "Ненецкий АО",
		"76": "Тыва",
		"77": "Коми",
		"78": "Новосибирская область",
		"79": "Саратовская область",
		"80": "Ленинградская область",
		"81": "Московская область",
		"82": "Крым",
	}
	l = [
		[10.0, 4.0, 525000.0, 3.3, 3.1, 2.7, 3.7, 3.4, 3.0, 4.2, 2.9, 3.4, 81.0],
		[7.0, 10.0, 600000.0, 3.4, 3.2, 3.0, 3.7, 3.5, 3.1, 4.2, 3.2, 3.5, 2.3, 49],
		[5.0, 3.0, 500000.0, 3.1, 2.9, 2.7, 3.6, 3.3, 3.0, 4.2, 2.4, 3.1, 2.3, 81],
		[15.0, 98.0, 850000.0, 1.4, 1.6, 1.9, 3.5, 2.8, 2.4, 3.9, 2.4, 2.7, 1.6, 81]
		# [5.0, 8.0, 1500000, 4.4, 3.8, 3.3, 3.8, 3.4, 3.4, 4.2, 3.3, 3.9]
	]
	array = [
		[set_data_from_design(l[0]), city[str(l[0][-1])], l[0][2]],
		# [set_data_from_design(l[1]), city[str(l[1][-1])], l[1][2]],
		# [set_data_from_design(l[2]), city[str(l[2][-1])], l[2][2]],
		# [set_data_from_design(l[3]), 'Краснодар', l[3][2]],
	]
	for i in range(len(array)):
		print(array[i], "Разница: ", abs(array[i][0] - l[i][2]))
# [10.0, 4.0, 560000.0, 3.1, 2.9, 2.7, 3.7, 3.5, 3.1, 4.2, 2.9, 3.2] Ижевск 0
# [3.0, 0.0, 650000.0, 3.6, 3.4, 3.4, 3.8, 3.9, 3.4, 4.6, 3.4, 3.7, 2.6] Ижевск 1
# [10.0, 44.0, 700000.0, 2.2, 2.7, 2.2, 3.6, 3.1, 2.8, 4.3, 2.8, 3.0, 1.8] Москва 2
