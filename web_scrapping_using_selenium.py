from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

options = Options()
options.headless = True
options.add_argument("--window-size=1920,1200")

browser = webdriver.Chrome()
browser.get(url="http://www.secfilings.com/login/")
browser.find_element_by_name("username").send_keys("bhoopendra.sharma@rho.ai")
browser.find_element_by_name("password").send_keys("Test!123#")
#browser.find_element_by_css_selector("btn btn-lg btn-primary").click()
browser.find_element_by_css_selector(".btn.btn-lg.btn-primary").click()

try:
    element = WebDriverWait(browser, 10).until(
        EC.presence_of_element_located((By.NAME, "search"))
    )
    print('Successfully logged in')
    
    link = browser.find_element_by_link_text("Search")
    link.click()

    element = WebDriverWait(browser, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".form-control.form-control-lg.form-control-borderless"))
    )
    element.send_keys("AAPL")


    table = WebDriverWait(browser, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".table.table-borderless.align-items-center"))
    )
    rows = table.find_elements(By.TAG_NAME, "tr")
        
    columnData = []
    allRowData = []
    NO_OF_COLS = 3
    for i,row in enumerate(rows):
        # Get the columns (all the column 2)        
        if i == 0 :
            columnName = row.find_elements(By.TAG_NAME, "th") #note: index start from 0, 1 is col 2
            for i,th in enumerate(columnName): 
                if  i == NO_OF_COLS :
                    break
                columnData.append(th.text) #prints text from the element
            dataframe = pd.DataFrame(columns=columnData)
        else : 
            #WebElement ele = driver.findElement(By.xpath("//span[@class='second']"));
            rowName = row.find_elements(By.TAG_NAME, "span") #note: index start from 0, 1 is col 2
            rowData = []
            for i,span in enumerate(rowName): 
                if span.get_attribute("class") == "text-blue" : 
                    NO_OF_COLS += 1
                    continue
                if  i == NO_OF_COLS : 
                    break
                rowData.append(span.text) #prints text from the element
            allRowData.append(rowData)
    
    dataframe = pd.DataFrame(allRowData, columns = columnData)  
    print(dataframe)
except Exception as e: 
    print(e)
    driver.quit()
browser.quit()
