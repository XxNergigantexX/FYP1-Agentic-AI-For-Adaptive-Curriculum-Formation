import json
import os

CONFIG_FILE = 'expert_data.json'

# --- THE FULL PRE-FILLED DATA ---
DEFAULT_CONFIG = {
    # 1. Custom Criteria Weights (General)
    "weights_manual": {
        "spm_results": 0.25,
        "previous_semester": 0.25,
        "technical_skills": 0.30,
        "aptitude_test": 0.20
    },
    
    # 2. RIASEC Weights per Specialization
    "riasec_weights": {
        'ADE': {'R': 5, 'I': 3, 'A': 10, 'S': 5, 'E': 3, 'C': 3},
        'AI': {'R': 3, 'I': 10, 'A': 5, 'S': 3, 'E': 0, 'C': 0},
        'SDS': {'R': 5, 'I': 10, 'A': 0, 'S': 0, 'E': 3, 'C': 5},
        'DE': {'R': 3, 'I': 5, 'A': 0, 'S': 0, 'E': 3, 'C': 10},
        'NDC': {'R': 10, 'I': 3, 'A': 0, 'S': 0, 'E': 3, 'C': 5}
    },
    
    # 3. Detailed Criteria Weights (The Pre-filled Subjects)
    "criteria": {
        "spm_results": {
            "Mathematics": {"ADE": 10, "AI": 10, "SDS": 10, "DE": 10, "NDC": 10},
            "Additional_Mathematics": {"ADE": 5, "AI": 10, "SDS": 5, "DE": 10, "NDC": 5},
            "Physics": {"ADE": 3, "AI": 5, "SDS": 10, "DE": 5, "NDC": 10},
            "Computer_Science": {"ADE": 10, "AI": 5, "SDS": 5, "DE": 5, "NDC": 5},
            "English": {"ADE": 5, "AI": 5, "SDS": 5, "DE": 5, "NDC": 5},
            "Chemistry": {"ADE": 3, "AI": 5, "SDS": 3, "DE": 3, "NDC": 5},
            "Biology": {"ADE": 3, "AI": 5, "SDS": 3, "DE": 3, "NDC": 3},
            "Economics_Accounting": {"ADE": 5, "AI": 3, "SDS": 3, "DE": 5, "NDC": 3}
        },
        "previous_semester": {
            "Programming_Fundamentals": {"ADE": 10, "AI": 10, "SDS": 5, "DE": 5, "NDC": 3},
            "Data_Structures_Algorithms": {"ADE": 5, "AI": 10, "SDS": 3, "DE": 10, "NDC": 5},
            "Database_Systems": {"ADE": 5, "AI": 5, "SDS": 3, "DE": 10, "NDC": 3},
            "Computer_Networking": {"ADE": 3, "AI": 3, "SDS": 10, "DE": 3, "NDC": 10},
            "Object_Oriented_Programming": {"ADE": 10, "AI": 5, "SDS": 3, "DE": 5, "NDC": 3},
            "Discrete_Structures": {"ADE": 3, "AI": 10, "SDS": 5, "DE": 5, "NDC": 3},
            "Software_Engineering": {"ADE": 10, "AI": 3, "SDS": 3, "DE": 5, "NDC": 3},
            "Operating_Systems": {"ADE": 3, "AI": 3, "SDS": 10, "DE": 5, "NDC": 10}
        },
        "technical_skills": {
            "Python_Programming": {"ADE": 5, "AI": 10, "SDS": 5, "DE": 10, "NDC": 3},
            "SQL_Database": {"ADE": 5, "AI": 5, "SDS": 3, "DE": 10, "NDC": 3},
            "Git_Version_Control": {"ADE": 10, "AI": 5, "SDS": 5, "DE": 5, "NDC": 5},
            "JavaScript_Web_Dev": {"ADE": 10, "AI": 3, "SDS": 3, "DE": 3, "NDC": 3},
            "Linux_Command_Line": {"ADE": 3, "AI": 5, "SDS": 10, "DE": 5, "NDC": 10},
            "Problem_Solving": {"ADE": 10, "AI": 10, "SDS": 5, "DE": 10, "NDC": 5},
            "Debugging": {"ADE": 10, "AI": 5, "SDS": 5, "DE": 5, "NDC": 3},
            "Algorithm_Design": {"ADE": 5, "AI": 10, "SDS": 3, "DE": 10, "NDC": 3}
        }
    },

    # 4. Expert Validations for MCDM Methods
    "mcdm": {
        "qrof": {
            "matrix": [[1.0]*4 for _ in range(4)]
        },
        "bwm": {
            "best_criteria": "technical_skills",
            "worst_criteria": "aptitude_test",
            "best_vectors": [1, 1, 1, 1],
            "worst_vectors": [1, 1, 1, 1]
        },
        "swara": {
            "rank_order": ["technical_skills", "spm_results", "previous_semester", "aptitude_test"],
            "comparative_scores": [0.1, 0.1, 0.1]
        }
    }
}

def load_data():
    if not os.path.exists(CONFIG_FILE):
        # File doesn't exist, start with defaults
        save_data(DEFAULT_CONFIG)
        return DEFAULT_CONFIG
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
            
            # CHECK: If the loaded data is missing the 'criteria' key, add it from defaults
            if 'criteria' not in data or not data['criteria']:
                data['criteria'] = DEFAULT_CONFIG['criteria']
            
            return data
    except (json.JSONDecodeError, IOError):
        # File corrupted, return defaults
        return DEFAULT_CONFIG

def save_data(data):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=4)