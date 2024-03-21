# pip install webdriver-manager selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

#from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import os
#train_namespace = ["dog","cat","dinosaur","rabbit","fox"]
train_namespace=["가디건","긴팔 티","누빔 옷","니트","린넨 옷","맨투맨","민소매","반팔","블라우스"
                  ,"야상","얇은 셔츠","자켓","청자켓","코트","트렌치코트","패딩","후드티"]

baseurl ="C:/Users/kangb/capstoneDesign/Back-end/IdeaProjects/Image"

for name in train_namespace:
    k=os.path.join(baseurl,"data/train",name.split(" ")[0])
    os.makedirs(k,exist_ok=True)
    
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_experimental_option('excludeSwitches',['enable-logging'])

    driver = webdriver.Chrome(options=options)

    driver.get("https://www.google.co.kr/imghp?hl=ko&authuser=0&ogbl")
    elem =driver.find_element(By.NAME,"q")
    elem.send_keys(name)
    elem.send_keys(Keys.RETURN)


    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(1)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                driver.find_element(By.CSS_SELECTOR, ".mye4qd").click()
            except:
                break
        last_height = new_height

    images = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")

    cnt = 0

    for img in images:
        if not os.path.isfile(k + "/" + str(cnt) + ".jpg"):
            try:
                start = time.time()
                img.click()
                cnt+=1
                time.sleep(2)
                src = driver.find_element(By.XPATH, "/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div/a/img").get_attribute("src")
                if start > 1*10^-5: pass
                urllib.request.urlretrieve(src, k + "/" + str(cnt) + ".jpg")
                if cnt == 500:
                    print(name," Finish!")
                    break
            except:
                print("Do Not this")
                pass    
        else:
            cnt+=1
            print("Exists already")
            pass

driver.quit()






 


