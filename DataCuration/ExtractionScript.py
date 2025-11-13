import json
import os
import time
from datetime import date
import requests
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC

DOWNLOAD_DIR = "I220755_IHC_Pdfs"

# Ensure folder exists
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

def format_date(date_str):
    """Convert date string to DD-MM-YYYY format"""
    try:
        if "/" in date_str:
            parts = date_str.split("/")
            if len(parts) == 3:
                return f"{parts[0].zfill(2)}-{parts[1].zfill(2)}-{parts[2]}"
        return date_str
    except:
        return date_str

def safe_click(driver, element):
    """Try normal click, else JS click"""
    try:
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        WebDriverWait(driver, 5).until(EC.element_to_be_clickable(element))
        element.click()
    except Exception:
        driver.execute_script("arguments[0].click();", element)

def close_alert_if_present(driver, wait):
    try:
        # Wait shortly for the modal to appear
        modal = WebDriverWait(driver, 2).until(
            EC.visibility_of_element_located((By.ID, "msgBoxClose"))
        )
        #print("Alert modal detected. Closing...")

        # Try closing via JS (more reliable than normal click)
        close_btn = modal.find_element(By.CSS_SELECTOR, "button.close, .ButtonClass")
        driver.execute_script("arguments[0].click();", close_btn)

        # Wait until modal disappears
        wait.until(EC.invisibility_of_element_located((By.ID, "msgBoxClose")))
        #print("Alert modal closed")
        return True

    except TimeoutException:
        # Modal didn’t appear → nothing to do
        return False

def extract_table_data_only(driver):
    """Extract all case data from the table and scrape detailed popups"""
    cases_data = []
    
    try:
        # Find the results table
        wait = WebDriverWait(driver, 25)
        table = wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        rows = table.find_elements(By.TAG_NAME, "tr")[1:] 
        
        #print(f"Found {len(rows)} cases in table")
        if not rows:
            return cases_data
        
        # Print headers once for debugging
        if rows:
            header_row = table.find_elements(By.TAG_NAME, "tr")[0]
            headers = [th.text.strip() for th in header_row.find_elements(By.TAG_NAME, "th")]
            #print(f"Table headers: {headers}")
        
        for i, row in enumerate(rows):
            table = wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
            case_data = {
                "Sr": i + 1,
                "Institution_Date": "N/A",
                "Case_No": "N/A", 
                "Case_Title": "N/A",
                "Bench": [],
                "Hearing_Date": "N/A",
                "Case_Category": "N/A",
                "Status": "N/A",
                "Orders": [],
                "Comments": [],
                "CMs": [],
                "Details": {}
            }
            
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                #print(f"Case {i+1} - Found {len(cells)} cells")
                
                # Extract data based on table structure - adjust indices if needed
                if len(cells) >= 0:
                    case_data["Sr"] = cells[0].text.strip()
                if len(cells) >= 1:
                    case_data["Institution_Date"] = cells[1].text.strip()
                if len(cells) >= 2:
                    case_data["Case_No"] = cells[2].text.strip()
                if len(cells) >= 3:
                    case_data["Case_Title"] = cells[3].text.strip()
                if len(cells) >= 4:
                    case_data["Bench"] = cells[4].text.strip()
                if len(cells) >= 5:
                    hearing_text = cells[5].text.strip()
                    parts = hearing_text.split("\n")
                    case_data["Hearing_Date"] = parts[0].strip() if len(parts) > 0 else "N/A"
                    case_data["Case_Category"] = parts[1].strip() if len(parts) > 1 else "N/A"
                if len(cells) >= 6:
                    case_data["Status"] = cells[6].text.strip()
                
                # Now scrape popups (Orders, Comments, CMs, Judgement)
                case_data = extract_detailed_case_info(driver, case_data, row)
                
                # Create default detailed structure
                case_data = create_detailed_structure(case_data)
                
                cases_data.append(case_data)
                #print(f"Extracted: {case_data['Case_No']}")

            except Exception as e:
                #print(f"Error processing row {i+1}: {str(e)}")
                cases_data.append(case_data)  # Add partial case data   
    
    except Exception as e:
        print(f"Error extracting table data: {str(e)}")
    
    return cases_data

def create_detailed_structure(case_data):
    """Create the detailed structure for a case, filling only missing parts."""
    
    # Create detailed case information
    if "Details" not in case_data or not case_data["Details"]:
        case_data["Details"] = {
            "Case_No": case_data.get("Case_No", "N/A"),
            "Case_Status": case_data.get("Status", "N/A"),
            "Hearing_Date": case_data.get("Hearing_Date", "N/A"),
            "Case_Stage": "N/A",
            "Tentative_Date": "N/A", 
            "Short_Order": case_data.get("Status", "N/A"),
            "Before_Bench": case_data.get("Bench", "N/A"),
            "Case_Title": case_data.get("Case_Title", "N/A"),
            "Advocates": {
                "Petitioner": "N/A",
                "Respondent": "N/A"
            },
            "Case_Description": "Case details extracted from IHC website",
            "Disposal_Information": {
                "Disposed_Status": case_data.get("Status", "N/A"),
                "Case_Disposal_Date": (
                    case_data.get("Hearing_Date", "N/A")
                    if "disposed" in case_data.get("Status", "").lower()
                    else "N/A"
                ),
                "Disposal_Bench": case_data.get("Bench", "N/A"),
                "Consigned_Date": "N/A"
            },
            "FIR_Information": {
                "FIR_No": "N/A",
                "FIR_Date": "N/A", 
                "Police_Station": "N/A",
                "Under_Section": "N/A",
                "Incident": "N/A",
                "Accused": "N/A"
            }
        }

    # Create order information only if none extracted
    if not case_data.get("Orders"):
        case_data["Orders"] = [{
            "Sr": 1,
            "Hearing_Date": case_data.get("Hearing_Date", "N/A"),
            "Bench": case_data.get("Bench", "N/A"),
            "List_Type": "Regular List",
            "Case_Stage": case_data.get("Case_Category", "N/A"),
            "Short_Order": case_data.get("Status", "N/A"),
            "Disposal_Date": (
                case_data.get("Hearing_Date", "N/A")
                if "disposed" in case_data.get("Status", "").lower()
                else "N/A"
            ),
            "Order_File": f"orders/order_{case_data.get('Case_No', 'N_A').replace('/', '-').replace(' ', '_')}.pdf"
        }]
    
    # Create comments information only if none extracted
    if not case_data.get("Comments"):
        case_data["Comments"] = [{
            "Compliance_Date": "N/A",
            "Case_No": case_data.get("Case_No", "N/A"),
            "Case_Title": case_data.get("Case_Title", "N/A"),
            "Doc_Type": "N/A",
            "Parties": case_data.get("Case_Title", "N/A"),
            "Description": "Extracted from IHC website",
            "View_File": f"comments/comment_{case_data.get('Case_No', 'N_A').replace('/', '-').replace(' ', '_')}.pdf"
        }]
    
    # Create CMs information only if none extracted
    if not case_data.get("CMs"):
        case_data["CMs"] = [{
            "Sr": 1,
            "CM": "N/A",
            "Institution_Date": "N/A",
            "Disposal_Date": "N/A", 
            "Order_Passed": "N/A",
            "Description": "No CMs available",
            "Status": "N/A"
        }]
    
    return case_data

def extract_detailed_case_info(driver, case_data, row):
    """Extract detailed case info by opening Orders, Comments, Case CMs, Judgement popups"""
    wait = WebDriverWait(driver, 5)

    # ---------- ORDERS ----------
    try:
        #order_btn = driver.find_element(By.ID, "202150")
        close_alert_if_present(driver, wait)

        order_btn = row.find_element(By.XPATH, ".//a[contains(@class, 'lnkCseDtlfrnt')]")
        
        safe_click(driver, order_btn)
        wait.until(EC.visibility_of_element_located((By.ID, "tblCseHstry_wrapper")))
        #print("Orders modal opened")
        orders = []
        order_rows = driver.find_elements(By.CSS_SELECTOR, "tblCseHstry")
        if not order_rows:
            order_rows = driver.find_elements(By.CSS_SELECTOR, "#tblCseHstry tbody tr")
            #print(f"Found {len(order_rows)} order rows")
        
        for o_row in order_rows:
            cols = o_row.find_elements(By.TAG_NAME, "td")
            #print(f"Order row columns: {[col.text for col in cols]}")
            
            order_file = "N/A"
            if len(cols) >= 8:

                try:
                    pdf_link = cols[7].find_element(By.TAG_NAME, "a").get_attribute("href")
                    if pdf_link and pdf_link.lower().endswith(".pdf"):
                        folder = "I220755_IHC_Pdfs/I220755_IHC_Orders"
                        os.makedirs(folder, exist_ok=True)

                        # save with case no + hearing date
                        safe_case_no = case_data['Case_No'].replace("/", "-").replace(" ", "_")
                        safe_date = cols[1].text.strip().replace("/", "-").replace(" ", "_")
                        filename = os.path.join(folder, f"{safe_case_no}_{safe_date}.pdf")

                        response = requests.get(pdf_link, stream=True)
                        with open(filename, "wb") as f:
                            for chunk in response.iter_content(1024):
                                f.write(chunk)

                        order_file = filename
                except Exception as e:
                    #print(f"No order file found")
                    order_file = "N/A"
                orders.append({
                    "Sr": cols[0].text.strip(),
                    "Hearing_Date": cols[1].text.strip(),
                    "Bench": cols[2].text.strip(),
                    "List_Type": cols[3].text.strip(),
                    "Case_Stage": cols[4].text.strip(),
                    "Short_Order": cols[5].text.strip(),
                    "Disposal_Date": cols[6].text.strip(),
                    "Order_File": order_file
                })

        case_data["Orders"] = orders
        close_btn = wait.until(EC.element_to_be_clickable(
            #(By.XPATH, "//*[@id='myModalCmnts']/div[2]/div/div[1]/button")
            (By.XPATH, "/html/body/div[2]/div[11]/div[2]/div/div[1]/button")
        ))
        close_btn.click()

        # Wait until modal is invisible
        wait.until(EC.invisibility_of_element_located((By.ID, "myModalCmnts")))
        #time.sleep(2)

    except Exception as e:
        #print(f"No Orders found for case {case_data['Case_No']}")
        case_data["Orders"] = []

    # ---------- COMMENTS ----------
    try:
        #comments_btn = driver.find_element(By.ID, "202151")
        close_alert_if_present(driver, wait)
        comments_btn = row.find_element(By.XPATH, ".//a[contains(@class, 'lnkCmntsFrnt')]")
        safe_click(driver,comments_btn)

        wait.until(EC.visibility_of_element_located((By.ID, "tblCmntsHstry")))
        #print("Comments modal opened")
        comments = []
        comment_rows = driver.find_elements(By.CSS_SELECTOR, "#tblCmntsHstry tbody tr")
        for c_row in comment_rows:
            cols = c_row.find_elements(By.TAG_NAME, "td")
            #print(f"Comment row columns: {[col.text for col in cols]}")
            
            if len(cols) >= 8 and "No data" not in c_row.text:
                view_file = "N/A"
                try:
                    link_el = cols[7].find_element(By.TAG_NAME, "a")
                    pdf_url = link_el.get_attribute("href")
                    if pdf_url and pdf_url.lower().endswith(".pdf"):
                        folder = "I220755_IHC_Pdfs/I220755_IHC_Comments"
                        os.makedirs(folder, exist_ok=True)

                        # unique filename with case no + compliance date
                        safe_case_no = case_data['Case_No'].replace("/", "-").replace(" ", "_")
                        safe_date = cols[1].text.strip().replace("/", "-").replace(" ", "_")
                        filename = os.path.join(folder, f"{safe_case_no}_{safe_date}.pdf")

                        response = requests.get(pdf_url, stream=True)
                        with open(filename, "wb") as f:
                            for chunk in response.iter_content(1024):
                                f.write(chunk)

                        view_file = filename
                except Exception as e:
                    print(f"No comment file found.")
                
                comments.append({
                    "Compliance_Date": cols[1].text.strip(),
                    "Case_No": cols[2].text.strip(),
                    "Case_Title": cols[3].text.strip(),
                    "Doc_Type": cols[4].text.strip(),
                    "Parties": cols[5].text.strip(),
                    "Description": cols[6].text.strip(),
                    "View_File": view_file
                })
        case_data["Comments"] = comments
        close_btn = wait.until(EC.element_to_be_clickable(
           # (By.XPATH, "//*[@id='myModalCmnts']/div[2]/div/div[1]/button")
            (By.XPATH, "/html/body/div[2]/div[13]/div[2]/div/div[1]/button")
            
        ))
        close_btn.click()

        # Wait until modal is invisible
        wait.until(EC.invisibility_of_element_located((By.ID, "myModalCmnts")))
        #time.sleep(2)
    except Exception as e:
        print(f"No Comments found for case {case_data['Case_No']}")
        case_data["Comments"] = []

    # ---------- CASE CMs ----------
    try:
        close_alert_if_present(driver, wait)
        #cms_btn = driver.find_element(By.XPATH,"//*[@id=\"202151\"]")
        cms_btn = row.find_element(By.XPATH, ".//a[contains(@class, 'lnkCseCMsFrnt')]")
        safe_click(driver, cms_btn)
        
        wait.until(EC.visibility_of_element_located((By.ID, "tblCmsHstry_filter")))
        #print("Case CMs modal opened")
        cms = []
        cms_rows = driver.find_elements(By.CSS_SELECTOR, "#tblCmsHstry tbody tr")
        for m_row in cms_rows:
            cols = m_row.find_elements(By.TAG_NAME, "td")
            #print(f"CM row columns: {[col.text for col in cols]}")
            if len(cols) >= 7 and "No data" not in m_row.text:
                cms.append({
                    "Sr": cols[0].text.strip(),
                    "CM": cols[1].text.strip(),
                    "Institution_Date": cols[2].text.strip(),
                    "Disposal_Date": cols[3].text.strip(),
                    "Order_Passed": cols[4].text.strip(),
                    "Description": cols[5].text.strip(),
                    "Status": cols[6].text.strip()
                })
        case_data["CMs"] = cms
        
        # Close button
        close_btn = wait.until(EC.element_to_be_clickable(
            #(By.XPATH, "//*[@id='myModalCmnts']/div[2]/div/div[1]/button")
            (By.XPATH, "/html/body/div[2]/div[10]/div[2]/div/div[1]/button")
        ))
        close_btn.click()

        # Wait until modal is invisible
        wait.until(EC.invisibility_of_element_located((By.ID, "myModalCmnts")))

        #time.sleep(2)
    except Exception as e:
        #print(f"No Case CMs found for case {case_data['Case_No']}")
        case_data["CMs"] = []
    

    # ---------- JUDGEMENT (Download PDF) ----------
    try:
        close_alert_if_present(driver, wait)
        judgment_btn = row.find_element(By.XPATH, ".//a[contains(@class, 'lnkFleJgmnt')]")
        #judgment_btn = driver.find_element(By.CLASS_NAME, "lnkFleJgmnt")
        pdf_url = judgment_btn.get_attribute("href")
        if pdf_url and pdf_url.lower().endswith(".pdf"):
            # Save file in I220755_IHC_Pdfs/
            folder = "I220755_IHC_Pdfs/I220755_IHC_Judgements"
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, f"{case_data['Case_No'].replace('/', '-')}.pdf")
            
             # Save with Case Title + Case No
            safe_title = case_data['Case_Title'].replace(" ", "_").replace("/", "-")
            filename = os.path.join(folder, f"{safe_title}_{case_data['Case_No'].replace('/', '-')}.pdf")

            response = requests.get(pdf_url, stream=True)
            with open(filename, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            case_data["Judgement_File"] = filename
        else:
            case_data["Judgement_File"] = "Not Available"
    except Exception as e:
        #print(f"No Judgement found for case {case_data['Case_No']}")
        case_data["Judgement_File"] = "N/A"

    return case_data

def run_search_for_date(driver, wait, dt):
    """Run advanced search for a specific date and return extracted cases"""
    # Open search page fresh for each date
    driver.get("https://mis.ihc.gov.pk/frmCseSrch")
    adv_btn = wait.until(EC.element_to_be_clickable((By.ID, "lnkAdvncSrch")))
    adv_btn.click()
    time.sleep(1)

    date_input = wait.until(EC.presence_of_element_located((By.ID, "txtDt")))
    date_input.clear()
    date_input.send_keys(dt.strftime("%d/%m/%Y"))

    driver.find_element(By.ID, "btnAdvnSrch").click()
    try:
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "dataTables_wrapper")))
        time.sleep(2)
        return extract_table_data_only(driver)
    except Exception:
        return []
    
def append_to_json(file_path, cases, dt):
    """Append daily scraped data to the year file"""
    # If file exists, load it, otherwise start new
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"Cases": []}

    # Add today’s cases
    data["Cases"].append({
        "Date": dt.strftime("%d-%m-%Y"),
        "Cases_List": cases
    })

    # Write back immediately
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    driver = webdriver.Chrome()
    wait = WebDriverWait(driver, 5)

    try:
        output_folder = "ISBHighCourt_I220755"
        os.makedirs(output_folder, exist_ok=True)

        start_year = 1971

        end_year = date.today().year
        today = date.today()

        for year in range(start_year, end_year + 1):
            year_file = os.path.join(output_folder, f"ISBHighCourt_I220755_{year}.json")
            
            for month in range(1, 13):
                if today.year and month > today.month:
                    break

                # Days in month (Feb fixed at 28)
                if month == 2:
                    days_in_month = 28
                else:
                    days_in_month = 30 if month in [4, 6, 9, 11] else 31
                
                for day in range(1, days_in_month + 1):
                    # Stop if future date in current year/month
                    if year == today.year and month == today.month and day > today.day:
                        break

                    dt = date(year, month, day)
                    print(f"Scraping {dt.strftime('%d-%m-%Y')} ...")
                    cases = run_search_for_date(driver, wait, dt)

                    if cases:
                        append_to_json(year_file, cases, dt)
                        #print(f"{len(cases)} cases written for {dt}")
                    else:
                        print(f"No cases for {dt}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
