"""
services.py — Core business logic: resume parsing, NLP, ML classification, JD matching.
Uses built-in Python regex tokenisation + embedded stopwords (no NLTK download needed).
"""

import os
import re
import pickle
from pathlib import Path

import PyPDF2
import docx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Embedded English Stopwords ───────────────────────────────────────────────
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than','too',
    'very','s','t','can','will','just','don','should','now','d','ll','m','o',
    're','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn','haven',
    'isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren','won',
    'wouldn','also','would','could','may','might','shall','get','got','make',
    'made','use','used','work','worked','new','good','great','able','back',
}

# ─── Skill Bank (comprehensive, covers real resume keywords) ──────────────────
# Each entry: (display_name, [regex_patterns_that_match_it])
# Patterns are matched case-insensitively with word boundaries.
SKILL_DEFINITIONS = [
    # ── Programming Languages ──────────────────────────────────────────────
    ("Python",              [r"python"]),
    ("Java",                [r"\bjava\b"]),          # \b prevents matching "javascript"
    ("JavaScript",          [r"javascript", r"js\b"]),
    ("TypeScript",          [r"typescript", r"\bts\b"]),
    ("C++",                 [r"c\+\+", r"\bcpp\b"]),
    ("C#",                  [r"c#", r"\bcsharp\b"]),
    ("Ruby",                [r"\bruby\b"]),
    ("PHP",                 [r"\bphp\b"]),
    ("Swift",               [r"\bswift\b"]),
    ("Kotlin",              [r"\bkotlin\b"]),
    ("Go",                  [r"\bgolang\b", r"\bgo\b(?! to)"]),
    ("Rust",                [r"\brust\b"]),
    ("Scala",               [r"\bscala\b"]),
    ("R",                   [r"\br programming\b", r"\blanguage r\b"]),
    ("Matlab",              [r"\bmatlab\b"]),
    ("Perl",                [r"\bperl\b"]),
    ("Bash",                [r"\bbash\b", r"\bshell scripting\b", r"\bshell script\b"]),
    ("SQL",                 [r"\bsql\b"]),
    ("HTML",                [r"\bhtml\b", r"\bhtml5\b"]),
    ("CSS",                 [r"\bcss\b", r"\bcss3\b"]),

    # ── Web Frameworks ─────────────────────────────────────────────────────
    ("Django",              [r"\bdjango\b"]),
    ("Flask",               [r"\bflask\b"]),
    ("FastAPI",             [r"\bfastapi\b", r"\bfast api\b"]),
    ("React",               [r"\breact\b", r"\breact\.?js\b"]),
    ("Angular",             [r"\bangular\b"]),
    ("Vue",                 [r"\bvue\b", r"\bvue\.?js\b"]),
    ("Node.js",             [r"\bnode\.?js\b", r"\bnodejs\b"]),
    ("Spring",              [r"\bspring\b", r"\bspring boot\b"]),
    ("Laravel",             [r"\blaravel\b"]),
    ("Rails",               [r"\brails\b", r"\bruby on rails\b"]),
    ("Express",             [r"\bexpress\.?js\b", r"\bexpress\b"]),
    ("Bootstrap",           [r"\bbootstrap\b"]),
    ("Tailwind",            [r"\btailwind\b"]),
    ("jQuery",              [r"\bjquery\b"]),
    ("GraphQL",             [r"\bgraphql\b"]),
    ("REST API",            [r"\brest\s*api\b", r"\brestful\b", r"\brest apis\b"]),
    ("Next.js",             [r"\bnext\.?js\b"]),
    ("Nuxt",                [r"\bnuxt\b"]),
    ("Gatsby",              [r"\bgatsby\b"]),

    # ── Data Science / ML ──────────────────────────────────────────────────
    ("Machine Learning",    [r"\bmachine\s*learning\b", r"\bml\b"]),
    ("Deep Learning",       [r"\bdeep\s*learning\b", r"\bdl\b"]),
    ("NLP",                 [r"\bnlp\b", r"\bnatural\s*language\s*processing\b"]),
    ("Computer Vision",     [r"\bcomputer\s*vision\b", r"\bimage\s*recognition\b"]),
    ("CNN",                 [r"\bcnn\b", r"\bconvolutional\s*neural\s*network\b", r"\bconvolutional neural\b"]),
    ("RNN",                 [r"\brnn\b", r"\brecurrent\s*neural\b"]),
    ("Transformer",         [r"\btransformer\b", r"\bbert\b", r"\bgpt\b"]),
    ("TensorFlow",          [r"\btensorflow\b", r"\btf\b"]),
    ("Keras",               [r"\bkeras\b"]),
    ("PyTorch",             [r"\bpytorch\b", r"\btorch\b"]),
    ("Scikit-learn",        [r"\bscikit[\-\s]?learn\b", r"\bsklearn\b", r"\bscikit\b"]),
    ("Pandas",              [r"\bpandas\b"]),
    ("NumPy",               [r"\bnumpy\b", r"\bnp\b(?=[\s,\.])"]),
    ("Matplotlib",          [r"\bmatplotlib\b"]),
    ("Seaborn",             [r"\bseaborn\b"]),
    ("Plotly",              [r"\bplotly\b"]),
    ("OpenCV",              [r"\bopencv\b", r"\bcv2\b"]),
    ("NLTK",                [r"\bnltk\b"]),
    ("spaCy",               [r"\bspacy\b"]),
    ("Hugging Face",        [r"\bhugging\s*face\b", r"\bhuggingface\b"]),
    ("Data Mining",         [r"\bdata\s*mining\b"]),
    ("Data Analysis",       [r"\bdata\s*analysis\b", r"\bdata\s*analytics\b"]),
    ("Statistics",          [r"\bstatistics\b", r"\bstatistical\b"]),
    ("Feature Engineering", [r"\bfeature\s*engineering\b"]),
    ("A/B Testing",         [r"\ba/?b\s*testing\b", r"\bab\s*testing\b"]),
    ("Time Series",         [r"\btime\s*series\b"]),
    ("Jupyter Notebook",    [r"\bjupyter\b", r"\bjupyter\s*notebook\b", r"\bjupyter\s*lab\b"]),
    ("Google Colab",        [r"\bgoogle\s*colab\b", r"\bcolab\b"]),

    # ── Databases ──────────────────────────────────────────────────────────
    ("SQL",                 [r"\bsql\b"]),
    ("MySQL",               [r"\bmysql\b"]),
    ("PostgreSQL",          [r"\bpostgresql\b", r"\bpostgres\b"]),
    ("MongoDB",             [r"\bmongodb\b", r"\bmongo\b"]),
    ("Redis",               [r"\bredis\b"]),
    ("SQLite",              [r"\bsqlite\b"]),
    ("Elasticsearch",       [r"\belasticsearch\b", r"\belastic\s*search\b"]),
    ("Firebase",            [r"\bfirebase\b"]),
    ("Oracle",              [r"\boracle\s*db\b", r"\boracle\s*database\b"]),
    ("Cassandra",           [r"\bcassandra\b"]),

    # ── Big Data / Data Eng ────────────────────────────────────────────────
    ("Apache Spark",        [r"\bapache\s*spark\b", r"\bspark\b"]),
    ("Hadoop",              [r"\bhadoop\b"]),
    ("Kafka",               [r"\bkafka\b"]),
    ("Airflow",             [r"\bairflow\b"]),
    ("dbt",                 [r"\bdbt\b"]),
    ("Snowflake",           [r"\bsnowflake\b"]),
    ("BigQuery",            [r"\bbigquery\b"]),
    ("Redshift",            [r"\bredshift\b"]),
    ("Tableau",             [r"\btableau\b"]),
    ("Power BI",            [r"\bpower\s*bi\b"]),

    # ── Cloud / DevOps ─────────────────────────────────────────────────────
    ("AWS",                 [r"\baws\b", r"\bamazon\s*web\s*services\b"]),
    ("Azure",               [r"\bazure\b", r"\bmicrosoft\s*azure\b"]),
    ("GCP",                 [r"\bgcp\b", r"\bgoogle\s*cloud\b"]),
    ("Cloud Computing",     [r"\bcloud\s*computing\b", r"\bcloud\b(?=\s*(platform|service|deploy))"]),
    ("Docker",              [r"\bdocker\b"]),
    ("Kubernetes",          [r"\bkubernetes\b", r"\bk8s\b"]),
    ("Terraform",           [r"\bterraform\b"]),
    ("Ansible",             [r"\bansible\b"]),
    ("Jenkins",             [r"\bjenkins\b"]),
    ("CI/CD",               [r"\bci/?cd\b", r"\bcontinuous\s*integration\b", r"\bcontinuous\s*delivery\b"]),
    ("Linux",               [r"\blinux\b", r"\bubuntu\b", r"\bdebian\b"]),
    ("Git",                 [r"\bgit\b(?!hub)"]),
    ("GitHub",              [r"\bgithub\b"]),
    ("GitLab",              [r"\bgitlab\b"]),
    ("Nginx",               [r"\bnginx\b"]),

    # ── Tools & IDEs ──────────────────────────────────────────────────────
    ("VS Code",             [r"\bvs\s*code\b", r"\bvisual\s*studio\s*code\b", r"\bvscode\b"]),
    ("PyCharm",             [r"\bpycharm\b"]),
    ("IntelliJ",            [r"\bintellijdea\b", r"\bintellij\b"]),
    ("Eclipse",             [r"\beclipse\b"]),
    ("Postman",             [r"\bpostman\b"]),
    ("Figma",               [r"\bfigma\b"]),
    ("Jira",                [r"\bjira\b"]),
    ("Confluence",          [r"\bconfluence\b"]),
    ("Slack",               [r"\bslack\b"]),

    # ── Mobile ────────────────────────────────────────────────────────────
    ("Android",             [r"\bandroid\b"]),
    ("iOS",                 [r"\bios\b", r"\bswift\b", r"\bxcode\b"]),
    ("Flutter",             [r"\bflutter\b"]),
    ("React Native",        [r"\breact\s*native\b"]),

    # ── Methodologies ─────────────────────────────────────────────────────
    ("Agile",               [r"\bagile\b"]),
    ("Scrum",               [r"\bscrum\b"]),
    ("Kanban",              [r"\bkanban\b"]),
    ("OOP",                 [r"\boop\b", r"\bobject[\s\-]oriented\b"]),
    ("MVC",                 [r"\bmvc\b", r"\bmodel[\s\-]view[\s\-]controller\b"]),
    ("Microservices",       [r"\bmicroservices\b", r"\bmicro\s*service\b"]),
    ("System Design",       [r"\bsystem\s*design\b"]),
    ("Data Structures",     [r"\bdata\s*structures?\b"]),
    ("Algorithms",          [r"\balgorithms?\b"]),

    # ── Business / Soft Skills ─────────────────────────────────────────────
    ("Excel",               [r"\bexcel\b", r"\bms\s*excel\b"]),
    ("PowerPoint",          [r"\bpowerpoint\b", r"\bms\s*powerpoint\b"]),
    ("Communication",       [r"\bcommunication\b"]),
    ("Teamwork",            [r"\bteamwork\b", r"\bteam\s*player\b", r"\bcollaboration\b"]),
    ("Problem Solving",     [r"\bproblem[\s\-]solving\b"]),
    ("Leadership",          [r"\bleadership\b"]),

    # ── Finance / HR ──────────────────────────────────────────────────────
    ("Accounting",          [r"\baccounting\b"]),
    ("Recruitment",         [r"\brecruitment\b", r"\brecruiting\b", r"\btalent\s*acquisition\b"]),
    ("Payroll",             [r"\bpayroll\b"]),

    # ── Engineering ───────────────────────────────────────────────────────
    ("AutoCAD",             [r"\bautocad\b"]),
    ("SolidWorks",          [r"\bsolidworks\b"]),
    ("ANSYS",               [r"\bansys\b"]),
    ("MATLAB",              [r"\bmatlab\b"]),
    ("PLC",                 [r"\bplc\b"]),
    ("SCADA",               [r"\bscada\b"]),
    ("PCB",                 [r"\bpcb\b", r"\bpcb\s*design\b"]),

    # ── Blockchain / Other ────────────────────────────────────────────────
    ("Blockchain",          [r"\bblockchain\b"]),
    ("Unity",               [r"\bunity\b"]),
    ("Salesforce",          [r"\bsalesforce\b"]),
]

# Resume validity signals
RESUME_KEYWORDS = {
    "experience","education","skills","summary","objective","projects",
    "internship","work","university","college","degree","bachelor","master",
    "phd","certification","training","volunteer","languages","achievements",
    "awards","publications","profile","contact","email","phone",
}

# ─── Text Extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise ValueError(f"Could not read PDF: {e}")
    return text.strip()


def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        raise ValueError(f"Could not read DOCX: {e}")


def extract_text(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# ─── Text Preprocessing ──────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    NLP preprocessing pipeline (no NLTK required):
    lowercase → remove URLs/emails/phones → strip special chars → tokenise → remove stopwords
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', '', text)
    text = re.sub(r'[^a-z\s\+\#]', ' ', text)   # keep + and # for C++, C#
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

# ─── Validity Check ───────────────────────────────────────────────────────────

def is_valid_resume(text: str) -> tuple:
    if len(text.strip()) < 100:
        return False, "Document is too short to be a resume."
    text_lower = text.lower()
    keyword_hits = sum(1 for kw in RESUME_KEYWORDS if kw in text_lower)
    if keyword_hits < 2:
        return False, "Document does not appear to contain resume content."
    return True, "Valid resume detected."

# ─── Skill Extraction ─────────────────────────────────────────────────────────

def extract_skills(text: str) -> list:
    """
    Extract skills using multiple regex patterns per skill.
    Case-insensitive. Returns deduplicated, sorted display names.
    """
    text_lower = text.lower()
    found = set()
    seen_display = {}   # display_name -> already added

    for display_name, patterns in SKILL_DEFINITIONS:
        if display_name in seen_display:
            continue
        for pattern in patterns:
            if re.search(pattern, text_lower):
                found.add(display_name)
                seen_display[display_name] = True
                break   # one match is enough for this skill

    return sorted(found)

# ─── Resume Scoring ───────────────────────────────────────────────────────────

def score_resume(text: str, skills: list) -> float:
    score = 0.0
    text_lower = text.lower()

    # Skills (up to 40 pts)
    score += min(len(skills) * 2.5, 40)

    # Word count (up to 20 pts)
    word_count = len(text.split())
    if word_count >= 300:
        score += min((word_count / 800) * 20, 20)

    # Key sections (up to 40 pts)
    sections = {
        'experience': 10, 'education': 10, 'skills': 8,
        'projects': 6, 'summary': 4, 'certification': 2,
    }
    for section, pts in sections.items():
        if section in text_lower:
            score += pts

    return round(min(score, 100), 1)

# ─── Suggestions ─────────────────────────────────────────────────────────────

def generate_suggestions(text: str, skills: list, category: str) -> list:
    suggestions = []
    text_lower = text.lower()

    if len(skills) < 5:
        suggestions.append("Add more technical skills relevant to your target role.")
    if "github" not in text_lower and "portfolio" not in text_lower:
        suggestions.append("Include a GitHub profile or portfolio link.")
    if "internship" not in text_lower and "experience" not in text_lower:
        suggestions.append("Add work experience or internship details.")
    if not any(w in text_lower for w in ['%','increased','reduced','improved','achieved','delivered']):
        suggestions.append("Add measurable achievements (e.g., 'Improved accuracy by 30%').")
    if "project" not in text_lower:
        suggestions.append("Include at least 2–3 relevant projects with outcomes.")
    if "certification" not in text_lower and "certified" not in text_lower:
        suggestions.append("Consider adding relevant certifications.")
    if len(text.split()) < 300:
        suggestions.append("Your resume seems short. Aim for at least 300–500 words.")
    if "summary" not in text_lower and "objective" not in text_lower:
        suggestions.append("Add a professional summary / objective at the top.")

    return suggestions

# ─── ML Classification ────────────────────────────────────────────────────────

_classifier = None

def _load_classifier():
    global _classifier
    if _classifier is None:
        model_path = Path(__file__).resolve().parent.parent / 'ml_model' / 'resume_classifier.pkl'
        if not model_path.exists():
            raise FileNotFoundError(
                "ML model not found. Please run: python ml_model/train_model.py"
            )
        with open(model_path, 'rb') as f:
            _classifier = pickle.load(f)
    return _classifier


def classify_resume(text: str) -> tuple:
    clf = _load_classifier()
    processed = preprocess_text(text)
    category = clf.predict([processed])[0]
    try:
        decision = clf.decision_function([processed])
        exp_d = np.exp(decision - np.max(decision))
        raw_conf = float(np.max(exp_d / exp_d.sum()))
        confidence = round(min(50 + raw_conf * 50, 99.0), 1)
    except Exception:
        confidence = 88.0
    return category, confidence

# ─── Job Description Matching ─────────────────────────────────────────────────

def match_job_description(resume_text: str, jd_text: str) -> dict:
    """
    Improved matching: combines TF-IDF cosine similarity with
    skill-overlap ratio for a more meaningful percentage.
    """
    # TF-IDF cosine similarity
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([
            preprocess_text(resume_text),
            preprocess_text(jd_text)
        ])
        tfidf_sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    except Exception:
        tfidf_sim = 0.0

    # Skill-based overlap ratio
    resume_skills = set(s.lower() for s in extract_skills(resume_text))
    jd_skills     = set(s.lower() for s in extract_skills(jd_text))

    if jd_skills:
        skill_overlap = len(resume_skills & jd_skills) / len(jd_skills)
    else:
        skill_overlap = 0.0

    # Weighted blend: 50% TF-IDF + 50% skill overlap (both matter)
    blended = (tfidf_sim * 0.5 + skill_overlap * 0.5) * 100
    match_percent = round(min(blended, 100.0), 1)

    matching = sorted([s.title() for s in resume_skills & jd_skills])
    missing  = sorted([s.title() for s in jd_skills - resume_skills])

    feedback = []
    if missing:
        feedback.append(f"Add these missing skills: {', '.join(missing[:5])}.")
    if match_percent < 40:
        feedback.append("Tailor your resume more closely to the job description keywords.")
    elif match_percent >= 70:
        feedback.append("Excellent match! Your resume aligns well with this role.")
    if not any(w in resume_text.lower() for w in ['led','built','improved','delivered','achieved']):
        feedback.append("Use stronger action verbs: Led, Built, Improved, Delivered, Achieved.")
    if len(matching) < 3:
        feedback.append("Incorporate more keywords from the job description into your resume.")

    return {
        "match_percent": match_percent,
        "matching_skills": matching,
        "missing_skills": missing,
        "feedback": feedback,
    }

# ─── Master Analyse Function ──────────────────────────────────────────────────

def analyze_resume(file_path: str) -> dict:
    raw_text = extract_text(file_path)

    valid, validity_reason = is_valid_resume(raw_text)
    if not valid:
        return {
            "status": "invalid",
            "reason": validity_reason,
            "category": None,
            "score": 0,
            "skills_found": [],
            "suggestions": ["Please upload a proper resume document."],
            "extracted_text": raw_text,
        }

    skills      = extract_skills(raw_text)
    category, confidence = classify_resume(raw_text)
    score       = score_resume(raw_text, skills)
    suggestions = generate_suggestions(raw_text, skills, category)

    return {
        "status": "valid",
        "reason": "Resume successfully analysed.",
        "category": category,
        "confidence": confidence,
        "score": score,
        "skills_found": skills,
        "suggestions": suggestions,
        "extracted_text": raw_text,
        "word_count": len(raw_text.split()),
    }
