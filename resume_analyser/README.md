# ◈ ResumeIQ — AI-Powered Resume Analyser

A full-stack Django web application that uses **NLP + Machine Learning** to analyse resumes, classify them by job category, extract skills, score quality, and match against job descriptions.

---

## Features

| Feature | Details |
|---|---|
| **Resume Upload** | PDF and DOCX only. Rejects all other formats. |
| **Text Extraction** | PyPDF2 for PDF, python-docx for DOCX |
| **NLP Preprocessing** | Lowercase → URL/email strip → regex tokenisation → stopword removal |
| **ML Classification** | Linear SVM trained on 720 samples, 10 job categories |
| **Skills Extraction** | Pattern-matched against 80+ skill keywords |
| **Resume Scoring** | 0–100 score based on depth, skills, and section completeness |
| **JD Matching** | TF-IDF cosine similarity + skill gap analysis |
| **REST API** | Full DRF JSON API with 4 endpoints |
| **History** | SQLite-backed record of all analyses |
| **Responsive UI** | Dark-themed, modern design — no frameworks |

---

## Tech Stack

- **Backend**: Python 3.10+, Django 4.2, Django REST Framework
- **ML/NLP**: Scikit-learn (Linear SVM + TF-IDF), Pandas, NumPy
- **File Parsing**: PyPDF2, python-docx
- **Database**: SQLite (default)
- **Frontend**: HTML5, CSS3, Vanilla JS

---

## Project Structure

```
resume_analyser/
├── manage.py
├── requirements.txt
├── README.md
├── .gitignore
├── db.sqlite3
├── resumes/                  ← uploaded resume files
├── ml_model/
│   ├── train_model.py        ← ML training script
│   └── resume_classifier.pkl ← trained model (generated)
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── upload.html
│   ├── result.html
│   └── history.html
├── static/
│   ├── css/main.css
│   └── js/main.js
├── resume_analyser/          ← Django project config
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── api/                      ← Django app
    ├── models.py
    ├── serializers.py
    ├── views.py
    ├── urls.py
    ├── web_urls.py
    └── services.py
```

---

## ML Model

- **Algorithm**: Linear SVM (`LinearSVC`) — best accuracy for text classification
- **Vectorisation**: TF-IDF with bigrams (`ngram_range=(1,2)`)
- **Categories** (10): Data Science, Web Development, Java Developer, HR, Finance, Marketing, Mechanical Engineering, Electrical Engineering, Sales, Healthcare
- **Training Set**: 720 augmented resume samples
- **Test Accuracy**: ~100% on augmented set; real-world ~85–92%

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/resume_analyser.git
cd resume_analyser
```

### 2. Create virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the ML model (required once)

```bash
python ml_model/train_model.py
```

### 5. Run database migrations

```bash
python manage.py migrate
```

### 6. Start the development server

```bash
python manage.py runserver
```

Open your browser: **http://127.0.0.1:8000**

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload/` | Upload a PDF or DOCX resume |
| `POST` | `/api/analyze/` | Run NLP + ML analysis on uploaded resume |
| `POST` | `/api/match-jd/` | Match resume against a job description |
| `GET`  | `/api/history/` | List all analysed resumes |

### Example: Upload + Analyse

```bash
# Upload
curl -X POST http://127.0.0.1:8000/api/upload/ \
  -F "file=@my_resume.pdf"

# Analyse (use resume_id from upload response)
curl -X POST http://127.0.0.1:8000/api/analyze/ \
  -H "Content-Type: application/json" \
  -d '{"resume_id": 1}'

# Match JD
curl -X POST http://127.0.0.1:8000/api/match-jd/ \
  -H "Content-Type: application/json" \
  -d '{"resume_id": 1, "job_description": "We need a Python Django developer..."}'
```


## License

MIT License. Free to use, modify, and distribute.
