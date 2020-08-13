from bs4 import BeautifulSoup as bs
import requests
import re
# from fake_useragent import UserAgent
import additional_data

HEADERS = {
	'User-Agent': ''  # UserAgent(verify_ssl=False).chrome
}


def parser(start_page=1, end_page=1, page_counter=True, _headers=None):
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

	# with open('../results/obhiy.csv', 'a', encoding='utf-8') as f:
	# 	f.write('Area,DistanceToCity,Cost,Ecology,Purity,Utilities,Neighbors,Children,SportsAndRecreation,Shops,Transport,Safety,LifeCost,City\n')
	url = 'https://www.domofond.ru/prodazha-uchastkizemli-leningradskaya_oblast-r'
	counter = 1
	# region = 81
	# page = 1658
	for region in range(82, 83):
		try:
			soup = bs(
				requests.get(f'{url}{region}?Page=1',
							 headers=_headers).content, 'html.parser')  # страница
			li = soup.findAll('li', 'pagination__page___2dfw0')
			nav = list()
			for i in range(len(li)):
				nav.append(int(li[i]['data-marker'].split('-')[1]))
			end_page = max(nav)
			print('Количество страниц: ', end_page, "Регион: ", region)
		except Exception:
			pass

		for i in range(start_page, end_page + 1):
			# print(f'{url}{region}?Page={i}')
			if page_counter:
				print('СТРАНИЦА НОМЕР: ', i)
			try:
				soup = bs(
					requests.get(f'{url}{region}?Page={i}',
								 headers=_headers).content, 'html.parser')  # страница
			except Exception as e:
				print(e)
				continue
			articles_links = soup.findAll('a', 'long-item-card__item___ubItG search-results__itemCardNotFirst___3fei6')[
							 1:]
			for link in articles_links:
				try:
					response_nested = requests.get(f'https://www.domofond.ru{link["href"]}',
												   headers=_headers)  # объявление
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
				if re.sub(r'[₽ ]', '', price) == "Неуказано":
					continue
				else:
					price = float(re.sub(r'[₽ ]', '', price))
				area = float(re.sub(r'сот..', '', area).replace(' ', ''))
				proximity = re.sub(r'[км ]', '', proximity)

				if '.' in proximity:
					proximity = proximity.split('.')[0]
				if proximity == "Вчертегорода":
					proximity = '2'
				proximity = float(proximity)
				with open('../results/obhiy.csv', 'a', encoding='utf-8') as f:
					f.write(
						f'{area},{proximity},{price},{",".join([y.replace(",", ".") for x, y in ratings.items()])},{region}\n'.replace(
							' ', '')
					)
					print(f'записан номер {counter}, регион: {region}')
				counter += 1
				ratings.clear()


def get_data_by_link(url: str):
	"""Получает и парсит полученную ссылку

	:param url: ссылка (str)
	:return: данные об участке земли (str)

	"""
	response = requests.get(url).content
	soup_nested = bs(response, 'html.parser')
	detail_information = soup_nested.findAll('div', 'detail-information__row___29Fu6')
	city = soup_nested.find('p', 'location__text___bhjoZ').get_text().split(',')[0]
	print(city)

	# Характеристики земли
	area, price = [detail.get_text().split(':')[1] for detail in detail_information[2:4]]
	proximity = detail_information[1].get_text().split(':')[1].split(',')[0]
	price = re.sub(r'[₽ ]', '', price)
	area = re.sub(r'сот..', '', area).replace(' ', '')
	proximity = re.sub(r'[км ]', '', proximity)

	if '.' in proximity:
		proximity = proximity.split('.')[0]
	if proximity == "Вчертегорода":
		proximity = '2'

	# Оценка района
	ratings = {}
	void_ratings = additional_data.get_average_from_file()
	dummy_response = f'{area},{proximity},{price},{void_ratings},{city}'
	try:
		for rating in soup_nested.findAll('div', 'area-rating__row___3y4HH'):
			ratings[rating.find('div', 'area-rating__label___2Y1bh').get_text()] \
				= rating.find('div', 'area-rating__score___3ERQc').get_text()
	except AttributeError:
		return dummy_response

	if not ratings:
		return dummy_response

	return f'{area},{proximity},{price},{",".join([{y} for x, y in ratings.items()])},{city}'


if __name__ == "__main__":
	parser()
# https://www.domofond.ru/prodazha-uchastkizemli-ryazanskaya_oblast-r62?PrivateListingType=PrivateOwner
# https://www.domofond.ru/prodazha-uchastkizemli-moskovskaya_oblast-r81?PrivateListingType=PrivateOwner <li
# class="pagination__page___2dfw0" tabindex="0" data-marker="pagination_page-1659"> <a
# href="/prodazha-uchastkizemli-moskovskaya_oblast-r81?Page=1659" tabindex="-1"
# class="search-results__pageLink___iNWzx">1659</a> </li>
