import requests, re, time, warnings, random
from bs4 import BeautifulSoup
import pandas as pd

from config import JOBS_DIR

warnings.filterwarnings("ignore")

# ================= API KEYS =================
ADZUNA_APP_ID  = "79c682ac"
ADZUNA_APP_KEY = "071d47ee79c2717b37a7afac3e4f2805"

# ================= CONFIG =================
PAGES_PER_SOURCE = 20
JOBS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = JOBS_DIR / "cs_engineering_jobs"

# ================= CS/ENGINEERING FOCUSED CATEGORIES (Adzuna) =================
ADZUNA_CATEGORIES = [
    "it-jobs",
    "engineering-jobs",
    "scientific-qa-jobs",
    "graduate-jobs",
]

# ================= CS/ENGINEERING KEYWORDS for filtering =================
CS_ENGINEERING_KEYWORDS = [
    # Software / Dev
    "software engineer", "software developer", "backend", "frontend", "full stack",
    "fullstack", "web developer", "mobile developer", "android", "ios", "flutter",
    "react native", "devops", "devsecops", "site reliability", "sre", "platform engineer",
    "cloud engineer", "infrastructure engineer", "solutions architect",

    # Data / AI / ML
    "data engineer", "data scientist", "data analyst", "machine learning", "ml engineer",
    "ai engineer", "deep learning", "nlp", "computer vision", "llm", "generative ai",
    "big data", "etl", "analytics engineer",

    # Systems / Hardware / Embedded
    "embedded systems", "firmware", "hardware engineer", "vlsi", "fpga",
    "electronics engineer", "electrical engineer", "iot", "robotics",
    "control systems", "signal processing",

    # Networking / Security
    "network engineer", "cybersecurity", "security engineer", "penetration testing",
    "ethical hacking", "soc analyst", "cloud security",

    # CS Fundamentals / Roles
    "computer science", "software architect", "qa engineer", "test engineer",
    "automation engineer", "release engineer", "build engineer",
    "blockchain", "game developer", "ar/vr", "unity", "unreal",

    # Engineering general
    "mechanical engineer", "civil engineer", "chemical engineer", "aerospace engineer",
    "structural engineer", "manufacturing engineer", "process engineer",
    "systems engineer", "product engineer"
]

# ================= EXPANDED SKILLS =================
SKILLS = [
    # Languages
    "Python", "Java", "C++", "C#", "C", "Go", "Rust", "Kotlin", "Swift",
    "JavaScript", "TypeScript", "PHP", "Ruby", "Scala", "R", "MATLAB",
    "Bash", "Shell", "Perl", "Dart",

    # Web / Frontend
    "React", "Angular", "Vue", "Node", "Next.js", "HTML", "CSS",
    "GraphQL", "REST API", "Django", "Flask", "FastAPI", "Spring Boot",
    "Express", "Laravel",

    # Data / ML / AI
    "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL", "Redis", "Elasticsearch",
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Keras",
    "Scikit-learn", "NLP", "Computer Vision", "LLM", "Spark", "Hadoop",
    "Kafka", "Airflow", "dbt", "Pandas", "NumPy",

    # Cloud / DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible",
    "Jenkins", "CI/CD", "GitHub Actions", "Linux", "Nginx", "Helm",
    "Prometheus", "Grafana", "Datadog",

    # Data Tools
    "Power BI", "Tableau", "Excel", "Looker", "Snowflake", "BigQuery", "Databricks",

    # Hardware / Embedded
    "FPGA", "VLSI", "Verilog", "VHDL", "Embedded C", "RTOS", "Arduino", "Raspberry Pi",
    "PCB Design", "AutoCAD", "SolidWorks", "MATLAB Simulink",

    # Security
    "Cybersecurity", "Penetration Testing", "SIEM", "Firewalls", "SSL/TLS",
    "OAuth", "IAM", "Zero Trust",

    # Other
    "Git", "Jira", "Agile", "Scrum", "Microservices", "Blockchain", "Solidity",
    "Unity", "Unreal Engine", "OpenCV", "ROS"
]

# ================= EXPERIENCE LEVEL DETECTION =================
def detect_experience(text):
    text = text.lower()
    if any(k in text for k in ["0-1", "0 to 1", "fresher", "entry level", "graduate", "intern", "trainee"]):
        return "Entry Level"
    if any(k in text for k in ["1-3", "1 to 3", "junior", "associate"]):
        return "Junior (1-3 yrs)"
    if any(k in text for k in ["3-5", "3 to 5", "mid level", "mid-level"]):
        return "Mid Level (3-5 yrs)"
    if any(k in text for k in ["5-8", "5 to 8", "senior", "sr."]):
        return "Senior (5+ yrs)"
    if any(k in text for k in ["8+", "10+", "lead", "principal", "architect", "manager", "head of"]):
        return "Lead/Principal (8+ yrs)"
    return "Not Specified"

# ================= SALARY EXTRACTION =================
def extract_salary(text):
    patterns = [
        r'₹\s?[\d,]+(?:\s?-\s?₹?\s?[\d,]+)?(?:\s?(?:lpa|lakh|lac|per annum|pa|k|month|yr|year))?',
        r'\$\s?[\d,]+(?:\s?-\s?\$?\s?[\d,]+)?(?:\s?(?:k|per year|per month|annually))?',
        r'[\d,]+\s?-\s?[\d,]+\s?(?:lpa|lakh|lac|k|usd|inr)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return "Not Disclosed"

# ================= DOMAIN CLASSIFIER =================
def classify_domain(title, desc):
    text = (title + " " + desc).lower()
    if any(k in text for k in ["machine learning", "data scientist", "ai ", "deep learning", "llm", "nlp", "computer vision"]):
        return "AI / ML / Data Science"
    if any(k in text for k in ["data engineer", "etl", "spark", "kafka", "pipeline", "data warehouse"]):
        return "Data Engineering"
    if any(k in text for k in ["frontend", "react", "angular", "vue", "ui developer", "web developer"]):
        return "Frontend / Web"
    if any(k in text for k in ["backend", "api", "microservices", "server", "django", "flask", "spring"]):
        return "Backend / API"
    if any(k in text for k in ["full stack", "fullstack"]):
        return "Full Stack"
    if any(k in text for k in ["devops", "sre", "cloud", "aws", "azure", "gcp", "kubernetes", "terraform"]):
        return "DevOps / Cloud"
    if any(k in text for k in ["embedded", "firmware", "fpga", "vlsi", "rtos", "iot", "microcontroller"]):
        return "Embedded / Hardware"
    if any(k in text for k in ["cybersecurity", "security", "penetration", "soc analyst", "ethical hack"]):
        return "Cybersecurity"
    if any(k in text for k in ["android", "ios", "flutter", "react native", "mobile"]):
        return "Mobile Development"
    if any(k in text for k in ["mechanical", "civil", "structural", "aerospace", "manufacturing", "process"]):
        return "Core Engineering"
    if any(k in text for k in ["blockchain", "solidity", "web3", "crypto"]):
        return "Blockchain / Web3"
    if any(k in text for k in ["game", "unity", "unreal", "ar", "vr"]):
        return "Game / AR / VR"
    if any(k in text for k in ["qa", "test", "automation", "selenium", "cypress"]):
        return "QA / Testing"
    return "Software Engineering"

# ================= IS CS/ENGINEERING RELATED? =================
def is_cs_or_engineering(title, desc):
    text = (title + " " + desc).lower()
    return any(kw in text for kw in CS_ENGINEERING_KEYWORDS)

# ================= CLEAN =================
def clean(t):
    if not t:
        return ""
    t = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', str(t))
    t = t.encode("utf-8", "ignore").decode("utf-8")
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()

# ================= WORK TYPE =================
WORK_TYPE_KEYWORDS = {
    "Remote": ["remote", "work from home", "wfh", "anywhere"],
    "Hybrid": ["hybrid"],
    "Onsite": ["onsite", "on-site", "in office", "office"]
}

def detect_work_type(text):
    text = text.lower()
    for wtype, keywords in WORK_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return wtype
    return "Onsite"

# ================= END DATE =================
def estimate_end_date(posted_date=None):
    try:
        base = pd.to_datetime(posted_date) if posted_date else pd.to_datetime("today")
        days = random.randint(15, 45)
        return str((base + pd.Timedelta(days=days)).date())
    except:
        return str((pd.to_datetime("today") + pd.Timedelta(days=30)).date())

# ================= LOCATION NORMALIZER =================
def normalize_location(loc):
    loc = loc.lower()
    if "bangalore" in loc or "bengaluru" in loc:
        return "Bengaluru"
    if "bombay" in loc or "mumbai" in loc:
        return "Mumbai"
    if "madras" in loc or "chennai" in loc:
        return "Chennai"
    if "delhi" in loc or "new delhi" in loc:
        return "Delhi"
    if "hyderabad" in loc:
        return "Hyderabad"
    if "pune" in loc:
        return "Pune"
    if "kolkata" in loc or "calcutta" in loc:
        return "Kolkata"
    if "noida" in loc:
        return "Noida"
    if "gurgaon" in loc or "gurugram" in loc:
        return "Gurgaon"
    if "remote" in loc:
        return "Remote"
    return loc.title()

# ================= SAFE REQUEST =================
def safe_request(url, params=None, headers=None):
    for _ in range(3):
        try:
            return requests.get(url, params=params, headers=headers, timeout=15)
        except:
            time.sleep(2)
    return None

# ================= LINKEDIN =================
def scrape_linkedin():
    jobs = []
    base_url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"

    # CS/Engineering focused search terms
    search_terms = [
        "software engineer", "data engineer", "devops", "machine learning",
        "embedded systems", "cloud engineer", "cybersecurity", "full stack developer",
        "backend developer", "frontend developer", "data scientist", "mlops"
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for term in search_terms:
        print(f"  LinkedIn | '{term}'")
        for page in range(3):
            r = safe_request(base_url, params={"keywords": term, "start": page * 25}, headers=headers)
            if not r:
                continue

            soup = BeautifulSoup(r.text, "lxml")

            for card in soup.find_all("li"):
                try:
                    title_el = card.find("h3")
                    company_el = card.find("h4")
                    loc_el = card.find("span", class_=re.compile("location", re.I))
                    if not loc_el:
                        loc_el = card.find("span")

                    if not title_el:
                        continue

                    title = clean(title_el.text)
                    company = clean(company_el.text if company_el else "Unknown")
                    location = normalize_location(loc_el.text if loc_el else "India")

                    # Try to get job detail link
                    link_el = card.find("a", href=True)
                    job_url = link_el["href"] if link_el else ""
                    desc = title  # fallback

                    # Fetch full job description if link available
                    if job_url and "linkedin.com" in job_url:
                        detail = safe_request(job_url, headers=headers)
                        if detail:
                            dsoup = BeautifulSoup(detail.text, "lxml")
                            desc_el = dsoup.find("div", class_=re.compile("description|job-view", re.I))
                            if desc_el:
                                desc = clean(desc_el.get_text())[:1000]

                    if not is_cs_or_engineering(title, desc):
                        continue

                    jobs.append({
                        "Title": title,
                        "Company": company,
                        "Job Description": desc,
                        "Skills": extract_skills(desc + " " + title),
                        "Location": location,
                        "Work Type": detect_work_type(f"{title} {desc}"),
                        "Experience Level": detect_experience(desc),
                        "Salary": extract_salary(desc),
                        "Domain": classify_domain(title, desc),
                        "End Date": estimate_end_date(),
                        "Source": "LinkedIn"
                    })
                except:
                    continue

    return jobs

# ================= ADZUNA =================
def scrape_adzuna():
    jobs = []

    # CS/Eng focused search keywords
    search_keywords = [
        "software engineer", "data engineer", "machine learning", "devops",
        "cloud engineer", "backend developer", "frontend developer",
        "full stack developer", "cybersecurity", "embedded engineer",
        "data scientist", "android developer", "ios developer",
        "network engineer", "solutions architect", "sre", "mlops",
        "firmware engineer", "vlsi", "fpga", "computer vision",
        "nlp engineer", "blockchain developer", "game developer",
        "qa automation", "mobile developer", "python developer",
        "java developer", "react developer", "node developer"
    ]

    base = "https://api.adzuna.com/v1/api/jobs/in/search"

    for keyword in search_keywords:
        print(f"  Adzuna | '{keyword}'")

        for page in range(1, 6):  # 5 pages per keyword
            r = safe_request(f"{base}/{page}", params={
                "app_id": ADZUNA_APP_ID,
                "app_key": ADZUNA_APP_KEY,
                "results_per_page": 50,
                "what": keyword,
                "category": "it-jobs"
            })

            if not r:
                continue

            try:
                data = r.json()
                results = data.get("results", [])
                if not results:
                    break

                for j in results:
                    # Full description from Adzuna
                    raw_desc = j.get("description", "")
                    desc = clean(BeautifulSoup(raw_desc, "lxml").text)

                    title = clean(j.get("title", ""))
                    company = clean(j.get("company", {}).get("display_name", "Unknown"))
                    location = normalize_location(
                        j.get("location", {}).get("display_name", "India")
                    )

                    # Optionally fetch redirect URL for full description
                    redirect_url = j.get("redirect_url", "")
                    full_desc = desc
                    if redirect_url and len(desc) < 200:
                        detail = safe_request(redirect_url)
                        if detail:
                            dsoup = BeautifulSoup(detail.text, "lxml")
                            main = dsoup.find("main") or dsoup.find("article") or dsoup.find("body")
                            if main:
                                full_desc = clean(main.get_text())[:1500]

                    if not is_cs_or_engineering(title, full_desc):
                        continue

                    expiry = j.get("expiration_date")
                    end_date = expiry[:10] if expiry else estimate_end_date()

                    salary_min = j.get("salary_min")
                    salary_max = j.get("salary_max")
                    salary = f"₹{int(salary_min):,} - ₹{int(salary_max):,}" if salary_min and salary_max else extract_salary(full_desc)

                    jobs.append({
                        "Title": title,
                        "Company": company,
                        "Job Description": full_desc[:1500],
                        "Skills": extract_skills(full_desc + " " + title),
                        "Location": location,
                        "Work Type": detect_work_type(f"{title} {full_desc}"),
                        "Experience Level": detect_experience(full_desc),
                        "Salary": salary,
                        "Domain": classify_domain(title, full_desc),
                        "End Date": end_date,
                        "Source": "Adzuna"
                    })

            except Exception as e:
                print(f"    Error on page {page}: {e}")
                continue

        print(f"    → {len(jobs)} total so far")

    return jobs

# ================= REMOTIVE =================
def scrape_remotive():
    jobs = []

    # CS/Eng relevant Remotive categories
    categories = [
        "software-dev", "devops-sysadmin", "data",
        "qa", "backend", "frontend"
    ]

    for cat in categories:
        print(f"  Remotive | '{cat}'")
        r = safe_request("https://remotive.com/api/remote-jobs", params={"category": cat})
        if not r:
            continue

        try:
            data = r.json()
            for j in data.get("jobs", []):
                raw_desc = j.get("description", "")
                desc = clean(BeautifulSoup(raw_desc, "lxml").text)

                title = clean(j.get("title", ""))
                company = clean(j.get("company_name", "Unknown"))

                if not is_cs_or_engineering(title, desc):
                    continue

                pub = j.get("publication_date", "")[:10]
                end_date = estimate_end_date(pub)

                jobs.append({
                    "Title": title,
                    "Company": company,
                    "Job Description": desc[:1500],
                    "Skills": extract_skills(desc + " " + title),
                    "Location": "Remote",
                    "Work Type": "Remote",
                    "Experience Level": detect_experience(desc),
                    "Salary": extract_salary(desc),
                    "Domain": classify_domain(title, desc),
                    "End Date": end_date,
                    "Source": "Remotive"
                })
        except:
            continue

    return jobs

# ================= THE MUSE =================
def scrape_themuse():
    jobs = []

    # CS/Eng relevant categories on The Muse
    categories = [
        "Engineering", "Data Science", "IT", "Dev & Ops",
        "Mobile", "QA", "UX/UI Design"
    ]

    for cat in categories:
        print(f"  TheMuse | '{cat}'")
        for page in range(1, 6):
            r = safe_request("https://www.themuse.com/api/public/jobs", params={
                "category": cat,
                "page": page,
                "api_key": "public"
            })
            if not r:
                break

            try:
                data = r.json()
                results = data.get("results", [])
                if not results:
                    break

                for j in results:
                    contents = j.get("contents", [])
                    desc_text = ""
                    if isinstance(contents, list):
                        for c in contents:
                            if isinstance(c, dict):
                                desc_text += c.get("body", "")
                    elif isinstance(contents, str):
                        desc_text = contents

                    desc = clean(BeautifulSoup(desc_text, "lxml").text)
                    title = clean(j.get("name", ""))

                    company = "Unknown"
                    if isinstance(j.get("company"), dict):
                        company = clean(j["company"].get("name", "Unknown"))

                    if not is_cs_or_engineering(title, desc):
                        continue

                    pub = j.get("publication_date", "")[:10]
                    end_date = estimate_end_date(pub)

                    # Location from levels
                    locations = j.get("locations", [])
                    location = locations[0].get("name", "Global") if locations else "Global"
                    location = normalize_location(location)

                    jobs.append({
                        "Title": title,
                        "Company": company,
                        "Job Description": desc[:1500],
                        "Skills": extract_skills(desc + " " + title),
                        "Location": location,
                        "Work Type": detect_work_type(f"{title} {desc}"),
                        "Experience Level": detect_experience(desc),
                        "Salary": extract_salary(desc),
                        "Domain": classify_domain(title, desc),
                        "End Date": end_date,
                        "Source": "TheMuse"
                    })
            except:
                continue

    return jobs

# ================= SKILLS EXTRACTOR (full) =================
def extract_skills(text):
    found = []
    text_lower = text.lower()
    for skill in SKILLS:
        if skill.lower() in text_lower:
            found.append(skill)
    return ", ".join(found) if found else "Not Listed"

# ================= BUILD DATAFRAME =================
def build_df(jobs):
    df = pd.DataFrame(jobs)

    cols = [
        "Title", "Company", "Domain", "Job Description",
        "Skills", "Experience Level", "Salary",
        "Location", "Work Type", "End Date", "Source"
    ]

    # Keep only columns that exist
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    df.drop_duplicates(subset=["Title", "Company"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

# ================= MAIN =================
if __name__ == "__main__":

    print("\n🚀 Running CS/Engineering Job Scraper v2\n")

    all_jobs = []

    print("→ LinkedIn...")
    all_jobs += scrape_linkedin()
    print(f"  LinkedIn done: {len(all_jobs)} jobs\n")

    print("→ Adzuna...")
    all_jobs += scrape_adzuna()
    print(f"  Adzuna done: {len(all_jobs)} jobs\n")

    print("→ Remotive...")
    all_jobs += scrape_remotive()
    print(f"  Remotive done: {len(all_jobs)} jobs\n")

    print("→ TheMuse...")
    all_jobs += scrape_themuse()
    print(f"  TheMuse done: {len(all_jobs)} jobs\n")

    print(f"Collected jobs before cleaning: {len(all_jobs)}")

    df = build_df(all_jobs)

    # Final clean pass
    df = df.applymap(lambda x: clean(x) if isinstance(x, str) else x)

    # Save outputs
    csv_path = OUTPUT.with_suffix(".csv")
    xlsx_path = OUTPUT.with_suffix(".xlsx")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ CSV saved → {csv_path}")

    try:
        df.to_excel(xlsx_path, index=False)
        print(f"✅ Excel saved → {xlsx_path}")
    except Exception as e:
        print(f"⚠️ Excel skipped: {e}")

    # Summary by domain
    print(f"\n✅ FINAL JOB COUNT: {len(df)}")
    print("\n📊 Jobs by Domain:")
    print(df["Domain"].value_counts().to_string())

    print("\n📊 Jobs by Source:")
    print(df["Source"].value_counts().to_string())

    print("\n✅ Done!")