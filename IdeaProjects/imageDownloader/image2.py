# pip install webdriver-manager selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

#from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import os

#train_namespace=["가디건","긴팔 티","니트","린넨 옷","맨투맨","민소매","반팔","블라우스","야상","얇은 셔츠","자켓","청자켓","코트","트렌치코트","패딩","후드티"]
#폴더 만드는 코드 

#for name in train_namespace:
#    os.makedirs('./Image/train2/'+str(name),exist_ok=True)

#train_namespace=["패딩","후드티"]
train_namespace=["누빔 옷"]
for name in train_namespace:
    
    chrome_options = webdriver.ChromeOptions()
    #driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=chrome_options)
    driver = webdriver.Chrome()
    URL='https://www.google.co.kr/imghp'
    driver.get(url=URL)
    driver.implicitly_wait(time_to_wait=10)
    keyElement=driver.find_element(By.XPATH,'//*[@id="APjFqb"]')
    keyElement.send_keys(name+' 중고')
    keyElement.send_keys(Keys.RETURN)


    bodyElement = driver.find_element(By.TAG_NAME,'body')
    time.sleep(1)

    for i in range(1):
        bodyElement.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)


    images = driver.find_elements(By.XPATH,'//*[@id="islrg"]/div[1]/div/a[1]')
    #print(len(images))

    imageSeq=200

    #//*[@id="Sva75c"]/div[2]/div[2]/div[2]/div[2]/c-wiz/div/div/div/div/div[3]/div[1]/a/img[1]
    #//*[@id="Sva75c"]/div[2]/div[2]/div[2]/div[2]/c-wiz/div/div/div/div/div[2]/div/a/img[1]
    #//*[@id="Sva75c"]/div[2]/div[2]/div[2]/div[2]/c-wiz/div/div/div/div/div[3]/div[1]/a/img[1]


    for image in images:
        image.click()
        time.sleep(0.5)
        highImages = driver.find_elements(By.XPATH,'//*[@id="Sva75c"]/div[2]/div[2]/div[2]/div[2]/c-wiz/div/div/div/div/div/div/a/img[1]')
        realImage=highImages[0].get_attribute('src')

        try:
            urllib.request.urlretrieve(realImage,'C:/Users/kangb/capstoneDesign/Back-end/IdeaProjects/imageDownloader/Image/train2/'+str(name)+'/'+str(imageSeq)+'.jpg')
            imageSeq +=1
        except:
            pass

    driver.quit()
