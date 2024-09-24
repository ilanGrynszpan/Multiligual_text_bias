import requests
from bs4 import BeautifulSoup

response = requests.get(
	url="https://en.wikipedia.org/wiki/democracy",
)

print(response.content)