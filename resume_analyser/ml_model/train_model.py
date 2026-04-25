"""
ML Model Training Script for Resume Classifier
Uses built-in Python tokenisation — no NLTK download required.

Usage:
    python ml_model/train_model.py
"""

import os
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STOPWORDS = {
    'i','me','my','we','our','you','your','he','him','his','she','her',
    'it','its','they','them','their','what','which','who','this','that',
    'these','those','am','is','are','was','were','be','been','have','has',
    'had','do','does','did','a','an','the','and','but','if','or','as',
    'of','at','by','for','with','about','into','through','to','from',
    'in','out','on','not','also','would','could','may','get','make','use',
    's','t','can','will','just','now','d','ll','m','re','ve','y',
}

def simple_preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

RESUME_DATA = {
    "Data Science": [
        "machine learning deep learning python tensorflow keras scikit-learn pandas numpy data analysis neural networks",
        "data scientist experience python r machine learning statistical modeling predictive analytics big data spark hadoop sql tableau",
        "phd statistics machine learning researcher pytorch transformers bert gpt model training hyperparameter tuning cross validation feature engineering",
        "senior data scientist recommendation systems collaborative filtering ab testing experimentation causal inference bayesian statistics time series forecasting",
        "ml engineer mlops model deployment docker kubernetes aws sagemaker feature store model monitoring drift detection",
        "computer vision engineer opencv yolo object detection image segmentation cnn resnet vgg transfer learning image classification",
        "nlp engineer text classification sentiment analysis named entity recognition information extraction question answering summarization embeddings",
        "data analyst sql python excel tableau power bi kpi dashboards reporting metrics business intelligence data visualization",
        "research scientist deep learning reinforcement learning generative adversarial networks variational autoencoders graph neural networks",
        "data engineer apache spark kafka airflow etl pipelines data warehouse redshift bigquery snowflake dbt data modeling",
        "applied scientist machine learning models production deployment ab testing statistical significance experimentation platform",
        "quantitative analyst python r matlab statistics financial modeling risk analysis monte carlo simulation derivatives pricing",
    ],
    "Web Development": [
        "frontend developer react angular vue javascript html css bootstrap tailwind responsive design ui ux web components",
        "full stack developer nodejs express django flask postgresql mongodb rest api graphql docker github ci cd",
        "backend developer python django rest framework api design microservices postgresql redis celery authentication jwt oauth",
        "javascript developer typescript react hooks redux state management webpack babel npm yarn performance optimization",
        "react developer spa single page application component library storybook jest testing cypress end-to-end",
        "node developer express fastify middleware routing authentication rate limiting caching redis",
        "php developer laravel symfony composer mysql database migrations eloquent blade templates",
        "ruby on rails developer mvc pattern restful api postgresql active record rspec testing",
        "web designer figma adobe xd html css animation sass scss grid flexbox responsive mobile first",
        "devops web aws ec2 s3 cloudfront load balancer ssl nginx apache deployment pipelines",
        "wordpress developer themes plugins woocommerce custom post types php jquery elementor",
        "vue developer nuxt vuex composition api pinia vite single file components webpack",
    ],
    "Java Developer": [
        "java developer spring boot hibernate jpa microservices rest api maven gradle junit docker kubernetes",
        "senior java engineer spring framework spring mvc spring security oauth2 jwt postgresql mysql",
        "java backend developer j2ee servlet jsp ejb weblogic jboss application server enterprise",
        "android java developer mobile app development sdk fragments activities lifecycles retrofit room database",
        "java architect design patterns solid principles clean code refactoring code review mentoring",
        "java developer multithreading concurrency thread pool executor service synchronization locks",
        "scala developer functional programming akka streams kafka spark java interoperability",
        "java devops jenkins ci cd pipeline sonar lint code quality docker containerization",
        "spring boot microservices service mesh kafka event driven architecture circuit breaker resilience",
        "java full stack angular react frontend java backend rest api integration testing",
        "kotlin android developer jetpack compose coroutines flow viewmodel livedata architecture",
        "java developer algorithms data structures leetcode competitive programming technical interviews",
    ],
    "HR": [
        "human resources manager recruitment talent acquisition onboarding training development performance management",
        "hr business partner employee relations compensation benefits payroll compliance labor law",
        "recruiter sourcing screening interviewing linkedin job boards applicant tracking system ats",
        "hr generalist policies procedures handbook employee engagement culture diversity inclusion equity",
        "talent management succession planning career development leadership programs mentoring",
        "compensation analyst benchmarking salary bands pay equity job evaluation hay method grading",
        "learning development trainer facilitator lms e-learning instructional design curriculum",
        "employee relations grievances investigations disciplinary conflict resolution mediation",
        "hr director strategic planning workforce planning organizational design change management",
        "payroll specialist adp workday payroll processing taxes benefits administration hris",
        "recruiting coordinator interview scheduling offer letters background checks onboarding logistics",
        "organizational development culture transformation change management stakeholder engagement",
    ],
    "Finance": [
        "financial analyst excel modeling dcf valuation bloomberg financial statements analysis forecast",
        "investment banker mergers acquisitions capital markets equity research debt financing ipo",
        "accountant cpa gaap ifrs audit financial reporting tax preparation balance sheet income statement",
        "portfolio manager asset allocation equities fixed income derivatives risk management alpha",
        "risk analyst market risk credit risk operational risk var stress testing basel regulation",
        "cfo financial planning budgeting forecasting treasury cash management working capital",
        "private equity venture capital deal sourcing due diligence portfolio management exit",
        "hedge fund quantitative trading algorithmic strategies backtesting execution trading desk",
        "financial controller month end close consolidation intercompany reconciliation reporting",
        "tax manager corporate tax international tax transfer pricing compliance filing strategy",
        "credit analyst loan underwriting financial modeling credit scoring covenant monitoring",
        "actuarial analyst insurance reserving pricing mortality tables solvency regulation ifrs17",
    ],
    "Marketing": [
        "digital marketing seo sem google ads facebook ads social media content marketing analytics",
        "marketing manager brand strategy campaign management market research consumer insights",
        "content creator copywriting blogging social media instagram tiktok youtube influencer",
        "growth hacker user acquisition retention funnel optimization conversion rate ab testing",
        "product marketing go-to-market positioning messaging competitive analysis product launch",
        "email marketing hubspot mailchimp automation segmentation open rate click rate nurture",
        "seo specialist keyword research on-page off-page link building technical seo audit ranking",
        "social media manager community management engagement scheduling hootsuite buffer analytics",
        "market research consumer surveys focus groups data analysis insights reporting presentation",
        "brand manager visual identity guidelines campaigns sponsorships events pr communications",
        "performance marketing paid media roas attribution multi-touch media mix modeling budget",
        "crm marketing salesforce marketing cloud customer journey personalization lifecycle",
    ],
    "Mechanical Engineering": [
        "mechanical engineer cad solidworks autocad catia ansys finite element analysis stress simulation",
        "design engineer product development prototyping tolerance stack gd&t manufacturing drawings",
        "hvac engineer thermal analysis heat transfer fluid dynamics piping systems hvac design",
        "automotive engineer powertrain chassis suspension braking system vehicle dynamics testing",
        "manufacturing engineer lean six sigma kaizen process improvement production line efficiency",
        "robotics engineer ros kinematics dynamics control systems actuators sensors automation",
        "aerospace engineer structures aerodynamics propulsion fea cfd certification testing",
        "quality engineer iso 9001 inspection metrology statistical process control defect analysis",
        "project engineer construction civil mechanical contractor schedule budget supervision",
        "maintenance engineer preventive corrective reliability rcm plant equipment maintenance",
        "materials engineer metallurgy polymers composites failure analysis testing material selection",
        "product engineer npd stage gate design for manufacturing dfmea tolerance analysis",
    ],
    "Electrical Engineering": [
        "electrical engineer circuit design pcb layout altium eagle schematic power systems protection",
        "embedded systems firmware c c++ rtos microcontroller arduino raspberry pi iot sensors",
        "power electronics inverter converter motor drives pwm igbt mosfet control systems",
        "signal processing dsp fpga vhdl verilog matlab simulink filter design algorithm",
        "telecommunications rf microwave antenna design wireless protocols 5g lte wifi bluetooth",
        "plc automation siemens allen-bradley scada hmi ladder logic industrial automation",
        "vlsi chip design asic synthesis place route verification timing closure tape-out",
        "iot connected devices mqtt protocols edge computing sensors wireless embedded",
        "renewable energy solar photovoltaic wind energy grid integration battery storage",
        "control systems pid feedback loop state space simulation matlab tuning commissioning",
        "test engineer automated testing bench equipment oscilloscope spectrum analyzer debugging",
        "electrical design substation switchgear relay protection coordination grounding",
    ],
    "Sales": [
        "sales executive b2b enterprise saas crm salesforce pipeline management quota attainment",
        "account manager client relationships upselling cross-selling revenue growth retention",
        "business development partnerships alliances lead generation prospecting cold calling outreach",
        "sales engineer technical presales solution selling demos proof of concept implementation",
        "retail sales customer service point of sale inventory cash handling targets achievement",
        "inside sales sdr bdr outbound inbound qualification discovery calls pipeline",
        "regional sales manager territory planning team leadership coaching kpi performance",
        "channel sales resellers distributors partner ecosystem enablement revenue",
        "customer success manager onboarding adoption renewal churn reduction nps health score",
        "sales operations forecasting reporting dashboards crm hygiene commissions incentive",
        "pharmaceutical sales medical devices healthcare hospital clinical territory management",
        "real estate agent property listings buyers sellers negotiation closing contract",
    ],
    "Healthcare": [
        "registered nurse rn patient care clinical documentation ehr emr medications administration",
        "physician doctor diagnosis treatment clinical trials evidence based medicine hospital",
        "pharmacist drug interactions counseling dispensing formulary clinical pharmacy review",
        "medical lab technician pathology blood tests specimen analysis quality control laboratory",
        "physical therapist rehabilitation exercise therapy musculoskeletal orthopedic patient",
        "healthcare administrator hospital management operations budget compliance accreditation",
        "clinical data manager clinical trials protocol icf sae reporting fda regulatory",
        "radiologist imaging mri ct scan x-ray diagnostic reporting radiology",
        "dentist oral health procedures implants cosmetic hygiene patient management practice",
        "mental health counselor therapy cbt dbt crisis intervention treatment plans sessions",
        "health informatics ehr implementation interoperability hl7 fhir data analytics",
        "public health epidemiology biostatistics surveillance health policy prevention",
    ],
}

def build_dataset():
    texts, labels = [], []
    rng = np.random.default_rng(42)
    for category, resumes in RESUME_DATA.items():
        for resume in resumes:
            words = resume.split()
            for _ in range(6):
                idx = rng.permutation(len(words))
                shuffled = [words[i] for i in idx[:len(words)//2]]
                augmented = " ".join(shuffled) + " " + resume
                texts.append(simple_preprocess(augmented))
                labels.append(category)
    return texts, labels

def train():
    print("Building dataset...")
    texts, labels = build_dataset()
    print(f"Dataset: {len(texts)} samples, {len(set(labels))} categories")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            min_df=2,
        )),
        ('clf', LinearSVC(C=1.0, max_iter=2000, random_state=42)),
    ])

    print("Training Linear SVM (best for text classification)...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    model_path = os.path.join(SCRIPT_DIR, 'resume_classifier.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model saved → {model_path}")
    return acc

if __name__ == '__main__':
    train()
