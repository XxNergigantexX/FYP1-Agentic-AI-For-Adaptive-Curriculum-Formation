from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import json
from utils.data_manager import load_data, save_data
from utils.mcdm_logic import QROFAHP, BWM_VIKOR, SWARA_MOORA, CRITIC_EDAS

app = Flask(__name__)
app.secret_key = 'ai_curriculum_secret_key'

# ==================== 1. CONSTANTS & STATIC DATA ====================

CRITERIA_KEYS = ["spm_results", "previous_semester", "technical_skills", "aptitude_test"]
CRITERIA_LABELS = {
    "spm_results": "SPM Results", 
    "previous_semester": "Previous Semester", 
    "technical_skills": "Technical Skills", 
    "aptitude_test": "Aptitude Test"
}

SPECIALIZATION_NAMES = {
    'ADE': 'APPLICATION DEVELOPMENT ENGINEERING',
    'AI': 'ARTIFICIAL INTELLIGENCE',
    'SDS': 'SECURITY IN DIGITAL SYSTEM',
    'DE': 'DATA ENGINEERING',
    'NDC': 'NETWORK AND DATA COMMUNICATIONS'
}

# Used for the Results Page (About Section)
SPECIALIZATION_INFO = {
    "APPLICATION DEVELOPMENT ENGINEERING": {
        "subjects": [
            "Project Management in Software Engineering", "Requirements Engineering", 
            "Software Design and Architecture", "Software Testing", 
            "Software Quality Assurance", "Blockchain and Applications"
        ],
        "careers": [
            {"title": "Software Developer", "desc": "Building applications and systems"},
            {"title": "Full-Stack Developer", "desc": "Frontend and backend development"},
            {"title": "Mobile App Developer", "desc": "iOS/Android application development"},
            {"title": "DevOps Engineer", "desc": "Development and operations integration"}
        ]
    },
    "ARTIFICIAL INTELLIGENCE": {
        "subjects": [
            "Machine Learning", "Natural Language Processing", "Computer Vision", 
            "Neural Networks", "Bio-inspired Computing", "Generative AI"
        ],
        "careers": [
            {"title": "AI Engineer", "desc": "Designing intelligent systems"},
            {"title": "Data Scientist", "desc": "Analyzing complex data sets"},
            {"title": "NLP Specialist", "desc": "Working with language models"},
            {"title": "Robotics Engineer", "desc": "Integrating AI with hardware"}
        ]
    },
    "SECURITY IN DIGITAL SYSTEM": {
        "subjects": [
            "Privacy Engineering", "Malware Analysis", "Penetration Testing", 
            "Cyber Physical Security", "Digital Forensics", "Applied Cryptography"
        ],
        "careers": [
            {"title": "Cybersecurity Analyst", "desc": "Protecting systems from threats"},
            {"title": "Ethical Hacker", "desc": "Identifying vulnerabilities"},
            {"title": "Security Architect", "desc": "Designing secure infrastructures"},
            {"title": "Forensic Investigator", "desc": "Analyzing digital crimes"}
        ]
    },
    "DATA ENGINEERING": {
        "subjects": [
            "Statistics for Data Science", "Big Data Analytics", "Pattern Mining", 
            "Brain Computational Analytics", "Data Visualization"
        ],
        "careers": [
            {"title": "Data Engineer", "desc": "Building data pipelines"},
            {"title": "Database Administrator", "desc": "Managing data storage"},
            {"title": "Big Data Architect", "desc": "Designing large-scale data systems"},
            {"title": "BI Developer", "desc": "Business intelligence reporting"}
        ]
    },
    "NETWORK AND DATA COMMUNICATIONS": {
        "subjects": [
            "Network Administration", "Advance Routing", "Cloud Infrastructure", 
            "Network Security", "IoT Applications", "Wireless Networks"
        ],
        "careers": [
            {"title": "Network Engineer", "desc": "Designing communication networks"},
            {"title": "Cloud Architect", "desc": "Managing cloud infrastructure"},
            {"title": "System Administrator", "desc": "Maintaining IT systems"},
            {"title": "IoT Specialist", "desc": "Connecting smart devices"}
        ]
    }
}

# Full 30-Question List
RIASEC_QUESTIONS = [
    {"id": 1, "trait": "R", "text": "When building furniture, do you enjoy following technical diagrams?"},
    {"id": 2, "trait": "R", "text": "Do you prefer hands-on activities like fixing electronic devices?"},
    {"id": 3, "trait": "R", "text": "When playing video games, do you enjoy figuring out game mechanics?"},
    {"id": 4, "trait": "R", "text": "Have you enjoyed activities involving tools like repairing phones?"},
    {"id": 5, "trait": "R", "text": "Do you like organizing physical spaces for optimal workflow?"},
    {"id": 6, "trait": "I", "text": "When watching detective movies, do you try to solve the mystery?"},
    {"id": 7, "trait": "I", "text": "Do you research topics deeply online out of curiosity?"},
    {"id": 8, "trait": "I", "text": "Have you enjoyed solving complex puzzles like Sudoku?"},
    {"id": 9, "trait": "I", "text": "When learning, do you prefer understanding the 'why'?"},
    {"id": 10, "trait": "I", "text": "Do you enjoy analyzing patterns in everyday life?"},
    {"id": 11, "trait": "A", "text": "Have you designed creative social media posts?"},
    {"id": 12, "trait": "A", "text": "Do you enjoy innovative solutions to problems?"},
    {"id": 13, "trait": "A", "text": "When decorating, do you focus on unique aesthetics?"},
    {"id": 14, "trait": "A", "text": "Have you created original content like digital art?"},
    {"id": 15, "trait": "A", "text": "Do you enjoy brainstorming 'outside the box'?"},
    {"id": 16, "trait": "S", "text": "Do you help friends with technology problems?"},
    {"id": 17, "trait": "S", "text": "Have you enjoyed teaching concepts to classmates?"},
    {"id": 18, "trait": "S", "text": "Do you prefer working in team projects?"},
    {"id": 19, "trait": "S", "text": "Have you volunteered to organize events?"},
    {"id": 20, "trait": "S", "text": "Do people come to you for advice?"},
    {"id": 21, "trait": "E", "text": "Have you taken lead in organizing events?"},
    {"id": 22, "trait": "E", "text": "Do you enjoy convincing friends to try new tech?"},
    {"id": 23, "trait": "E", "text": "Have you managed a budget for a project?"},
    {"id": 24, "trait": "E", "text": "Do you naturally take charge?"},
    {"id": 25, "trait": "E", "text": "Have you successfully pitched an idea?"},
    {"id": 26, "trait": "C", "text": "Do you enjoy creating organized systems?"},
    {"id": 27, "trait": "C", "text": "Have you made detailed plans or checklists?"},
    {"id": 28, "trait": "C", "text": "Do you prefer following clear procedures?"},
    {"id": 29, "trait": "C", "text": "Have you enjoyed cataloging collections?"},
    {"id": 30, "trait": "C", "text": "Do you ensure all details are correct?"}
]

# ==================== 2. HELPER FUNCTIONS ====================

def convert_grade(grade):
    g = str(grade).upper().strip()
    return {'A+':95,'A':90,'A-':85,'B+':80,'B':75,'B-':70,'C+':65,'C':60,'D':45,'E':40}.get(g, 0)

def get_subjects_from_db(category):
    """Fetch subjects from JSON to generate Student Form"""
    data = load_data()
    return list(data['criteria'].get(category, {}).keys())

def calculate_weighted_score(student_scores, weights_map, is_skill=False):
    total_possible = 0
    weighted_score = 0
    for subject, importance in weights_map.items():
        raw_score = student_scores.get(subject, 0)
        if is_skill:
            normalized = (raw_score - 1) * (100 / 9) if raw_score > 0 else 0
        else:
            normalized = raw_score
        weighted_score += normalized * importance
        total_possible += 100 * importance
    if total_possible == 0: return 0
    return (weighted_score / total_possible) * 100

def rank_results(scores_dict):
    sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    return [{'spec': k, 'score': round(v, 1), 'rank': i+1} for i, (k, v) in enumerate(sorted_items)]

# ==================== 3. EXPERT ROUTES ====================

@app.route('/expert')
def expert_dashboard():
    data = load_data()
    
    # Calculate weights live for the dashboard display
    weights = {'qrof':[], 'bwm':[], 'swara':{}}
    
    try:
        qrof = QROFAHP()
        weights['qrof'] = qrof.calculate_weights(data['mcdm']['qrof']['matrix']).tolist()
    except: pass
    
    try:
        bwm = BWM_VIKOR()
        bd = data['mcdm']['bwm']
        k_map = {k: i for i, k in enumerate(CRITERIA_KEYS)}
        w = bwm.solve_bwm_weights(k_map[bd['best_criteria']], k_map[bd['worst_criteria']], bd['best_vectors'], bd['worst_vectors'])
        weights['bwm'] = w.tolist()
    except: pass
    
    try:
        swara = SWARA_MOORA()
        sd = data['mcdm']['swara']
        weights['swara'] = swara.calculate_swara_weights(sd['rank_order'], sd['comparative_scores'])
    except: pass
    
    return render_template('expert/dashboard.html', data=data, weights=weights)

@app.route('/expert/criteria', methods=['GET', 'POST'])
def expert_criteria():
    data = load_data()
    if request.method == 'POST':
        criteria_json = request.form.get('criteria_json')
        if criteria_json:
            try:
                data['criteria'] = json.loads(criteria_json)
                save_data(data)
                return redirect(url_for('expert_criteria'))
            except Exception as e:
                print(f"Error saving Criteria: {e}")
    if 'criteria' not in data: data['criteria'] = {}
    return render_template('expert/criteria.html', data=data)

@app.route('/expert/riasec', methods=['GET', 'POST'])
def expert_riasec():
    data = load_data()
    if request.method == 'POST':
        riasec_json = request.form.get('riasec_json')
        if riasec_json:
            try:
                data['riasec_weights'] = json.loads(riasec_json)
                save_data(data)
                return redirect(url_for('expert_riasec'))
            except Exception as e:
                print(f"Error saving RIASEC: {e}")
    if 'riasec_weights' not in data: data['riasec_weights'] = {}
    return render_template('expert/riasec.html', data=data)

@app.route('/expert/qrof', methods=['GET', 'POST'])
def expert_qrof():
    data = load_data()
    if request.method == 'POST':
        matrix = []
        for i in range(4):
            row = []
            for j in range(4):
                val_str = request.form.get(f'cell_{i}_{j}')
                if '/' in val_str:
                    n, d = map(float, val_str.split('/'))
                    row.append(n/d)
                else:
                    row.append(float(val_str))
            matrix.append(row)
        data['mcdm']['qrof']['matrix'] = matrix
        save_data(data)
        return redirect(url_for('expert_dashboard'))
    return render_template('expert/qrof.html', matrix=data['mcdm']['qrof']['matrix'], labels=list(CRITERIA_LABELS.values()))

@app.route('/expert/bwm_1', methods=['GET', 'POST'])
def expert_bwm_1():
    if request.method == 'POST':
        session['bwm_best'] = request.form.get('best')
        session['bwm_worst'] = request.form.get('worst')
        return redirect(url_for('expert_bwm_2'))
    return render_template('expert/bwm_step1.html', criteria=CRITERIA_LABELS)

@app.route('/expert/bwm_2', methods=['GET', 'POST'])
def expert_bwm_2():
    best = session.get('bwm_best')
    worst = session.get('bwm_worst')
    if not best: return redirect(url_for('expert_bwm_1'))
    if request.method == 'POST':
        best_vec = [float(request.form.get(f'best_to_{k}')) for k in CRITERIA_KEYS]
        worst_vec = [float(request.form.get(f'{k}_to_worst')) for k in CRITERIA_KEYS]
        data = load_data()
        data['mcdm']['bwm'] = {"best_criteria": best, "worst_criteria": worst, "best_vectors": best_vec, "worst_vectors": worst_vec}
        save_data(data)
        return redirect(url_for('expert_dashboard'))
    return render_template('expert/bwm_step2.html', best=best, worst=worst, keys=CRITERIA_KEYS, labels=CRITERIA_LABELS)

@app.route('/expert/swara_1', methods=['GET', 'POST'])
def expert_swara_1():
    if request.method == 'POST':
        ranks = []
        for k in CRITERIA_KEYS:
            ranks.append((k, int(request.form.get(f'rank_{k}'))))
        ranks.sort(key=lambda x: x[1])
        session['swara_sorted_criteria'] = [x[0] for x in ranks]
        return redirect(url_for('expert_swara_2'))
    return render_template('expert/swara_step1.html', criteria=CRITERIA_LABELS)

@app.route('/expert/swara_2', methods=['GET', 'POST'])
def expert_swara_2():
    sorted_keys = session.get('swara_sorted_criteria')
    if not sorted_keys: return redirect(url_for('expert_swara_1'))
    if request.method == 'POST':
        comps = []
        for i in range(1, 4):
            val = float(request.form.get(f'sj_{sorted_keys[i]}', 0.1))
            comps.append(val)
        data = load_data()
        data['mcdm']['swara'] = {"rank_order": sorted_keys, "comparative_scores": comps}
        save_data(data)
        return redirect(url_for('expert_dashboard'))
    return render_template('expert/swara_step2.html', sorted_keys=sorted_keys, labels=CRITERIA_LABELS)

# ==================== 4. STUDENT ROUTES ====================

@app.route('/')
def index(): return render_template('index.html')

@app.route('/student/info', methods=['GET', 'POST'])
def student_info():
    if request.method == 'POST':
        session['student_info'] = request.form.to_dict()
        return redirect(url_for('student_spm'))
    return render_template('student/info.html')

@app.route('/student/spm', methods=['GET', 'POST'])
def student_spm():
    subjects = get_subjects_from_db('spm_results')
    if request.method == 'POST':
        session['spm_scores'] = {s: convert_grade(request.form.get(s)) for s in subjects}
        return redirect(url_for('student_university'))
    return render_template('student/spm.html', subjects=subjects)

@app.route('/student/university', methods=['GET', 'POST'])
def student_university():
    subjects = get_subjects_from_db('previous_semester')
    if request.method == 'POST':
        session['uni_scores'] = {s: convert_grade(request.form.get(s)) for s in subjects}
        return redirect(url_for('student_skills'))
    return render_template('student/university.html', subjects=subjects)

@app.route('/student/skills', methods=['GET', 'POST'])
def student_skills():
    skills = get_subjects_from_db('technical_skills')
    if request.method == 'POST':
        session['skills_scores'] = {s: int(request.form.get(s) or 1) for s in skills}
        return redirect(url_for('student_riasec'))
    return render_template('student/skills.html', skills=skills)

@app.route('/student/riasec', methods=['GET', 'POST'])
def student_riasec():
    if request.method == 'POST':
        res = [int(request.form.get(f'q{i}') or 3) for i in range(1, 31)]
        mapped = [{1:-2, 2:-1, 3:0, 4:1, 5:2}[r] for r in res]
        traits = ['R','I','A','S','E','C']
        session['riasec_scores'] = {t: sum(mapped[i*5:(i+1)*5]) for i, t in enumerate(traits)}
        return redirect(url_for('student_results'))
    return render_template('student/riasec.html', questions=RIASEC_QUESTIONS)

@app.route('/student/results')
def student_results():
    if 'student_info' not in session: return redirect(url_for('index'))
    data = load_data()
    crit = data.get('criteria', {})
    
    # 1. Build Decision Matrix
    specs = list(SPECIALIZATION_NAMES.keys())
    mat = np.zeros((5, 4))
    
    for i, s in enumerate(specs):
        # A. SPM
        w_spm = {k: int(v.get(s,5)) for k,v in crit.get('spm_results',{}).items()}
        mat[i,0] = calculate_weighted_score(session['spm_scores'], w_spm)
        
        # B. Previous Sem
        w_prev = {k: int(v.get(s,5)) for k,v in crit.get('previous_semester',{}).items()}
        mat[i,1] = calculate_weighted_score(session['uni_scores'], w_prev)
        
        # C. Skills
        w_skill = {k: int(v.get(s,5)) for k,v in crit.get('technical_skills',{}).items()}
        mat[i,2] = calculate_weighted_score(session['skills_scores'], w_skill, True)
        
        # D. RIASEC
        r_w = data['riasec_weights'].get(s, {})
        apt = sum(session['riasec_scores'].get(t,0)*r_w.get(t,0) for t in ['R','I','A','S','E','C'])
        max_apt = sum(abs(v)*2 for v in r_w.values()) or 1
        mat[i,3] = max(0, min(100, (apt+max_apt)/(2*max_apt)*100))

    # 2. Run Methods
    res = {}
    spec_names = list(SPECIALIZATION_NAMES.values())
    
    try:
        qrof = QROFAHP()
        w = qrof.calculate_weights(data['mcdm']['qrof']['matrix'])
        res['q-ROF-AHP'] = rank_results(dict(zip(spec_names, qrof.calculate_scores(mat, w))))
    except: res['q-ROF-AHP'] = []

    try:
        bwm = BWM_VIKOR()
        bd = data['mcdm']['bwm']
        km = {k: i for i, k in enumerate(CRITERIA_KEYS)}
        w = bwm.solve_bwm_weights(km[bd['best_criteria']], km[bd['worst_criteria']], bd['best_vectors'], bd['worst_vectors'])
        res['BWM+VIKOR'] = rank_results(dict(zip(spec_names, bwm.calculate_vikor(mat, w))))
    except: res['BWM+VIKOR'] = []

    try:
        swara = SWARA_MOORA()
        sd = data['mcdm']['swara']
        wd = swara.calculate_swara_weights(sd['rank_order'], sd['comparative_scores'])
        w = np.array([wd.get(k,0) for k in CRITERIA_KEYS])
        res['SWARA+MOORA-3NAG'] = rank_results(dict(zip(spec_names, swara.calculate_moora(mat, w))))
    except: res['SWARA+MOORA-3NAG'] = []

    try:
        ce = CRITIC_EDAS()
        sc, _ = ce.execute(mat)
        res['LTSF-CRITIC-EDAS'] = rank_results(dict(zip(spec_names, sc)))
    except: res['LTSF-CRITIC-EDAS'] = []

    # 3. Consensus
    tracker = {name: {'r':0, 's':0} for name in spec_names}
    for m in res:
        for item in res[m]:
            tracker[item['spec']]['r'] += item['rank']
            tracker[item['spec']]['s'] += item['score']
    
    consensus = [{'spec': k, 'avg_rank': v['r']/4, 'avg_score': v['s']/4} for k,v in tracker.items()]
    consensus.sort(key=lambda x: x['avg_rank'])
    for i, c in enumerate(consensus, 1): c['rank'] = i
    
    top = consensus[0]['spec']
    info = SPECIALIZATION_INFO.get(top, {'subjects':[], 'careers':[]})
    weak = [k for k,v in session['skills_scores'].items() if v < 6]

    return render_template('student/results.html', 
                           student=session['student_info'],
                           consensus=consensus, method_results=res,
                           top_spec_name=top, top_spec_info=info,
                           weak_skills=weak, methods=res.keys())

if __name__ == '__main__':
    app.run(debug=True)