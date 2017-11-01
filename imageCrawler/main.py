import urllib.request
import urllib.parse
from bs4 import BeautifulSoup as bs
from sys import argv
import os

def make_directory(dir):
    import os
    if not os.path.isdir(dir):
        os.mkdir(dir)

def contents_load(filename):
    content_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        while True:
            content = f.readline()
            if not content:
                break
            content = content.replace('\n', '')
            content_list.append(content)
    return content_list

def print_status(tag_name, tag_number, number_of_tag, image_number, number_of_image): # 상태 확인용 $2
    os.system('cls')
    print('{0}, {1}/{2}, {3:0.1f}%'.format(tag_name, tag_number, number_of_tag, image_number / number_of_image * 100))

defalut_num_of_image = 10
input_argv = argv
if len(input_argv) == 1:
    num_of_image = defalut_num_of_image # 만약 입력이 안됐을 경우, defalut 값을 입력
elif len(input_argv) == 2:
    _, num_of_image = argv
    num_of_image = int(num_of_image) # str로 들어온다

contents_filename = 'crawling_list.txt'
result_folder_name = 'crawling_result'
google_image_onepage_limit = 20
content_counter = 0 # $2를 위함

contents_list = contents_load(contents_filename)
for content in contents_list:
    folder_name = result_folder_name + '/' + content
    image_index = 0
    content_counter += 1

    for iter in range(round(num_of_image/google_image_onepage_limit + 0.5)): # 구글은 한 페이지에 20개의 이미지를 제공하고, num_of_image만큼 이미지를 얻기 위해 20으로 나눠준 값의 올림을 해준만큼 반복
        first = True  # 첫번째 img 태그가 필요없는거라서 건너뛰기 위함($1)
        end_flag = False
        url = "http://images.google.com/images?q=" + content + '&btnG=검색&start=' + str(iter * 20) + '&sout=1'
        url = urllib.parse.quote(url, safe= ':/=&?') # url에 유니코드 적용하고, urllib.request.Request가 받아들일 수 있는 형태로 바꿈, safe는 변환하지 않을 문자들을 지정
        print(url.encode('utf8'))
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}) # 헤더가 없으면, bot으로 인식해서 http 403 에러 뜬다

        print_status(content, content_counter, len(contents_list), image_index, num_of_image)
        data = urllib.request.urlopen(req).read()
        soup = bs(data)
        image_tag_list = soup.find_all('img')
        if not image_tag_list: # 결과가 없으면 넘김
            end_flag = True
            print('더이상 ' + content + '가 없습니다.')
            break
        make_directory(result_folder_name)
        for tag in image_tag_list:
            make_directory(folder_name)
            if first: # $1
                first = False
                continue
            image_index += 1
            try:
                urllib.request.urlretrieve(tag['src'], folder_name + '/' + content + str(image_index) + '.jpg')
            except:
                image_tag_list.append(tag)
                print('for debug')
                image_index -= 1
            if image_index >= num_of_image:
                end_flag = True
                break
        if end_flag:
            break

print_status(content, len(contents_list), len(contents_list), num_of_image, num_of_image)