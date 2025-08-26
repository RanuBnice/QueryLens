# SQL Analysis Script Runner

A **Streamlit-based application** that analyzes SQL queries from an uploaded Excel file using the **Groq API**.  
It extracts real physical tables, columns, and join conditions, then groups similar reports together.  
Final results are displayed in the UI and can be downloaded as CSV.

---

## 🚀 Features
- Upload Excel file with SQL queries (must contain sheet `Sorted one`)
- Extracts:
  - Real physical tables (ignores CTEs/temp tables)
  - Columns accessed
  - Join conditions
- Groups reports into logical buckets (based on table similarity)
- Download processed results as CSV

---

## ✅ Requirements
- Python 3.9+
- Dependencies (see `requirements.txt`):
  ```bash
  pip install -r requirements.txt
  ```

---

## ▶️ Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/sql-analysis-app.git
   cd sql-analysis-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run Streamlit app:
   ```bash
   streamlit run app_Rationalization.py
   ```

4. Open in your browser at: [http://localhost:8501](http://localhost:8501)

---

## 🔑 Configuration
### Groq API Key
- Create `.streamlit/secrets.toml` in your project folder:
  ```toml
  groq_api_key = "your_api_key_here"
  ```

- Update `app_Rationalization.py` to use:
  ```python
  client = Groq(api_key=st.secrets["groq_api_key"])
  ```

### Input File
- Must be an Excel file (`.xlsx`) with a sheet named **Sorted one**
- Required columns:
  - `Name` → Report Name
  - `SQL` → SQL query

---

## ☁️ Deployment on Streamlit Cloud
1. Push repo to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io) → New App
3. Select repo + branch + main file (`app_Rationalization.py`)
4. Add **Groq API Key** in app secrets
5. Deploy and share your app link 🎉

---

## 📂 Output Example
- **UI:** Displays processed SQL analysis
- **CSV (`testing_v2.csv`)**:
  - ReportID
  - ReportName
  - Tables Accessed
  - Column Accesses
  - All_Joins
  - Joins_without_Temp
  - Unique_Joins

---

## ⚠️ Notes
- Large SQL files may take longer (depends on API calls)
- Ensure Groq API credits are available
- Do not expose API keys in version control
