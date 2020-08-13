import sys
from PyQt5 import uic, QtWidgets
from PyQt5 import QtCore
from programs_parser import domofond_parser
from PyQt5.QtWidgets import QMessageBox


#  https://www.domofond.ru/uchastokzemli-na-prodazhu-skoropuskovskiy-1648805439

class UI(QtWidgets.QMainWindow):
	def __init__(self):
		super(UI, self).__init__()
		uic.loadUi('ui/ver4.ui', self)
		self.link = str()
		self.data = str()
		self.city = {
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

	def get_link(self):
		try:
			self.website_cost.setText('')
			self.data = domofond_parser.get_data_by_link(self.lineEdit.text().replace(' ', '')).split(',')
			print(self.data)
			# self.data[-1] = self.city[self.data[-1]]

			self.data = [7.0, 10.0, 600000.0, 3.4, 3.2, 3.0, 3.7, 3.5, 3.1, 4.2, 3.2, 3.5, 2.3, 49]

			from exported_tensor import set_data_from_design
			result = set_data_from_design(self.data)

			self.result.setText(str(int(result)) + "₽")
			self.cost1.setText("Cost")
			self.cost2.setText("Cost on website")
			self.website_cost.setText(str(int(self.data[2])) + " ₽")
		except Exception as e:
			msg = QMessageBox.warning(self, 'Error', "Try again", QMessageBox.Ok)
			print(e)

	def keyPressEvent(self, qKeyEvent):
		if qKeyEvent.key() == QtCore.Qt.Key_Return:
			self.get_link()

	# Формат данных в data:
	# '[10сот', '4км', '570000', Эколгоия: 3.1, Чистота: 2.9, ЖКХ: 2.7, Соседи: 3.7,
	# Условия для детей: 3.5, Спорт и отдых: 3.1, Магазины: 4.2, Транспорт: 2.9, Безопасность: 3.2]
	def initUI(self):
		self.setFixedSize(800, 600)
		self.setWindowTitle("NeuroLand")
		self.lineEdit.setText(' ')
		self.lineEdit.setText('https://www.domofond.ru/uchastokzemli-na-prodazhu-skoropuskovskiy-1648805439')
		self.find_button.clicked.connect(self.get_link)

		self.show()


if __name__ == '__main__':
	app = QtWidgets.QApplication(sys.argv)
	root = UI()
	root.initUI()
	app.exec_()
