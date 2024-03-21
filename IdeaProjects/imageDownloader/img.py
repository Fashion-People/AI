#from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver

driver = webdriver.Chrome()

URL='http://www.google.co.kr/imghp'
driver.get(url=URL)
driver.implicitly_wait(time_to_wait=10)

from selenium.webdriver.common.keys import Keys # 키입력 라이브러리
from selenium.webdriver.common.by import By # CSS선택 라이브러리

#검색창 원소를 selector로 찾아 클릭하여 바다를 검색한다.
keyElement=driver.find_element(By.XPATH,'//*[@id="APjFqb"]')
keyElement.send_keys("바다")
keyElement.send_keys(Keys.RETURN)



import time
keyElement =driver.find_element(By.TAG_NAME, "body")
for i in range(60):
    keyElement.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.1)

try:
    #'결과 더보기' 버튼의 원소를 selector로 찾아 클릭한다
    driver.find_element(By.CSS_SELECTOR, '#islmp > div > div > div > div > div.gBPM8 > div.qvfT1 > div.YstHxe > input').click()

    for i in range(60):
        keyElement.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.1)
except:
    pass

links=[]
#이미지의 원소를 모두 찾는다
images = driver.find_elements(By.CSS_SELECTOR, "#islrg > div.islrc > div > a.wXeWr.islib.nfEiy > div.bRMDJf.islir > img")

for image in images:
    if image.get_attribute('src') is not None: #링크 주소가 없으면 이미지가 없는 것 이므로 실행하지 않는다
        links.append(image.get_attribute('src')) #이미지의 다운로드 링크 주소를 links 리스트에 추가한다.


print('찾은 이미지 개수: ', len(links)) #links 리스트의 길이 = 이미지의 개수


import urllib.request

for i,k in enumerate(links):
    url = k
    urllib.request.urlretrieve(url, "C:/Users/kangb/Desktop/바다"+str(i)+".jpg")

print('다운로드 완료')