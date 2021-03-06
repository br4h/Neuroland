from bs4 import BeautifulSoup as bs
import requests
import re
from fake_useragent import UserAgent
from NeuroLand import additional_data

HEADERS = {
	'User-Agent': UserAgent(verify_ssl=False).chrome
}


def parser(start_page, end_page, page_counter=True, _headers=None):
	"""Записывает данные из domofond.ru по нескольким параметрам

	start_page(int) -- номер страницы, с которой функция начнёт парсинг
	end_page(int) -- номер страницы, на которой функция закончит парсинг
	page_counter(bool) -- показывать номер страницы, которую обрабатывает функция
	_headers -- user agent

	"""
	if _headers is None:
		_headers = HEADERS

	if start_page > end_page:
		print('Стартовая страница не может быть меньше конечной')
		exit()

	url = 'https://www.domofond.ru/prodazha-uchastkizemli-krasnodarskiy_kray-r24?Page='
	counter = 1
	for i in range(start_page, end_page + 1):
		if page_counter:
			print(f'СТРАНИЦА НОМЕР {i}:')

		try:
			soup = bs(requests.get(f'{url}{i}', headers=_headers).content, 'html.parser')  # страница
		except Exception:
			continue
		articles_links = soup.findAll('a', 'long-item-card__item___ubItG search-results__itemCardNotFirst___3fei6')[1:]
		for link in articles_links:
			try:
				response_nested = requests.get(f'https://www.domofond.ru{link["href"]}', headers=_headers)  # объявление
			except Exception:
				continue
			soup_nested = bs(response_nested.content, 'html.parser')
			detail_information = soup_nested.findAll('div', 'detail-information__row___29Fu6')

			# ОЦЕНКА РАЙОНА
			ratings = {}
			try:
				for rating in soup_nested.findAll('div', 'area-rating__row___3y4HH'):
					ratings[rating.find('div', 'area-rating__label___2Y1bh').get_text()] \
						= rating.find('div', 'area-rating__score___3ERQc').get_text()
			except AttributeError:
				print('Оценка отсутствует')
				continue

			if not ratings:
				continue

			area, price = [detail.get_text().split(':')[1] for detail in detail_information[2:4]]
			proximity = detail_information[1].get_text().split(':')[1].split(',')[0]
			price = re.sub(r'[₽ ]', '', price)
			area = re.sub(r'сот..', '', area).replace(' ', '')
			proximity = re.sub(r'[км ]', '', proximity)

			if '.' in proximity:
				proximity = proximity.split('.')[0]
			if proximity == "Вчертегорода":
				proximity = '2'

			with open('../results/old/krasnodar1.csv', 'a', encoding='utf-8') as f:
				f.write(
					f'{area},{proximity},{price},{",".join([y.replace(",", ".") for x, y in ratings.items()])}\n'.replace(
						' ', '')
				)
				print(f'записан номер {counter}')
			counter += 1
			ratings.clear()


if __name__ == "__main__":
	parser(1, 1005)  # 882 Начни с этой страницы
