# Islamabad High Court Case Scraper

## Tools Used

- **Python 3**
- **Selenium** (for browser automation)
- **webdriver-manager** (for automatic ChromeDriver management)
- **Google Chrome / Chromium** (browser)
- **JSON** (for structured data output)
- **VS Code** (development environment)

## Steps Followed

1. **Setup Environment**
   - Installed Python 3 and pip.
   - Installed required Python packages:
     ```
     pip install selenium webdriver-manager
     ```
   - Ensured Google Chrome/Chromium was installed and up to date.

2. **Script Development**
   - Wrote a Python script using Selenium to automate browser actions.
   - Navigated to the Islamabad High Court website and located the relevant case tables.
   - Extracted case details such as case number, title, bench, hearing date, status, and orders.
   - Structured the extracted data into a JSON format for easy storage and analysis.

3. **Data Extraction**
   - Ran the script to extract case data.
   - Saved the extracted data to a file named `ISBHighCourt_I220755.json`.

4. **Data Validation**
   - Opened the JSON file to verify the correctness and completeness of the extracted data.
   - Adjusted the extraction logic to fix any mapping or formatting issues.

5. **File Download**
   - Ran the script to extract case files for judgments, comments and Orders.
   - Saved the extracted data to a file named `ISBHighCourt_I220755_Pdfs/`.

6. **Files Uploaded on Google Drive**
    - Link: https://drive.google.com/drive/folders/1shBoCqEIvOXxB271lS9XOtCi_uSSOUmq?usp=sharing

## Issues Faced

- **ChromeDriver Version Mismatch:**  
  The version of ChromeDriver installed did not match the installed browser version, causing Selenium to fail.  
  **Solution:** Used `webdriver-manager` to automatically fetch the correct driver version.

- **Dynamic Website Structure:**  
  The website's table structure and field order sometimes changed, leading to incorrect data mapping.  
  **Solution:** Carefully mapped table columns to JSON fields and added error handling for missing or unexpected data.

- **Missing Data:**  
  Some cases had missing comments, CMs, or order files, resulting in empty or "N/A" fields in the output.  
  **Solution:** Added checks to handle missing or incomplete data gracefully.

- **Deprecation Warnings:**  
  Encountered warnings related to deprecated endpoints in browser logs.  
  **Solution:** Ignored these as they did not affect data extraction.

---

**Author:**  
*Arrij Fawwad*  
*20th September 2025*