import urllib.request
import urllib.parse
from bs4 import BeautifulSoup as bs
from sys import argv
import os

url = 'https://www.google.co.kr/search?as_st=y&hl=ko&tbs=ift%3Ajpg&tbm=isch&sa=1&ei=XB_7WYv7CseP8wXetaGQDg&q=소혜'
url = urllib.parse.quote(url, safe=':/=&?')  # url에 유니코드 적용하고, urllib.request.Request가 받아들일 수 있는 형태로 바꿈, safe는 변환하지 않을 문자들을 지정
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})  # 헤더가 없으면, bot으로 인식해서 http 403 에러 뜬다

data = urllib.request.urlopen(req).read()
soup = bs(data).prettify()
print(soup)

url = 'https://www.youtube.com/watch%3Fv%3DLRrnrAWgdeY&amp;sa=U&amp;ved=0ahUKEwjplbOZg6DXAhULjpQKHbvqBuoQwW4IFzAB&amp;usg=AOvVaw1WwLSVPP9Fw_sjws1m8QTY'
url = urllib.parse.unquote(url)  # url에 유니코드 적용하고, urllib.request.Request가 받아들일 수 있는 형태로 바꿈, safe는 변환하지 않을 문자들을 지정
url = urllib.parse.unquote(url)
print(url)