import numpy as np
from scipy.optimize import minimize
import math

# === CONFIGURATION ===
LINGUISTIC_TERMS = {
    'Extremely Poor': (0.05, 0.95, 0.95),
    'Very Poor': (0.15, 0.85, 0.85),
    'Poor': (0.25, 0.75, 0.75),
    'Slightly Poor': (0.35, 0.65, 0.65),
    'Fair': (0.50, 0.50, 0.50),
    'Slightly Good': (0.65, 0.35, 0.65),
    'Good': (0.75, 0.25, 0.75),
    'Very Good': (0.85, 0.15, 0.85),
    'Extremely Good': (0.95, 0.05, 0.95)
}

# ==================== METHOD 1: q-ROF-AHP ====================
class QROFAHP:
    def __init__(self, q=3):
        self.q = q

    def calculate_weights(self, pairwise_comparisons):
        n = len(pairwise_comparisons)
        q_scores = []

        for i in range(n):
            row_scores = []
            for j in range(n):
                val = pairwise_comparisons[i][j]
                # Safety check for 0
                if val == 0: val = 1
                
                mem = min(1, val / 9)
                non_mem = min(1, (1 / val) / 9)

                mem_q = mem ** self.q
                non_mem_q = non_mem ** self.q

                if mem_q + non_mem_q > 1:
                    scale = 1 / (mem_q + non_mem_q)
                    mem = (mem_q * scale) ** (1 / self.q)
                    non_mem = (non_mem_q * scale) ** (1 / self.q)

                row_scores.append((mem, non_mem))
            q_scores.append(row_scores)

        geo_means = []
        for i in range(n):
            prod_m, prod_nm = 1, 1
            for j in range(n):
                prod_m *= q_scores[i][j][0]
                prod_nm *= q_scores[i][j][1]
            geo_means.append((prod_m**(1/n), prod_nm**(1/n)))

        weights = []
        tot_m = sum(x[0] for x in geo_means)
        tot_nm = sum(x[1] for x in geo_means)
        
        # Prevent div by zero
        denom = tot_m + n - tot_nm
        if denom == 0: denom = 1

        for m, nm in geo_means:
            w = (m + (1 - nm)) / denom
            weights.append(w)

        w_sum = sum(weights)
        if w_sum == 0: return np.ones(n) / n
        
        return np.array(weights) / w_sum

    def calculate_scores(self, matrix, weights):
        # Normalize
        norm = np.zeros_like(matrix, dtype=float)
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            mn, mx = np.min(col), np.max(col)
            if mx - mn == 0:
                norm[:, j] = 1 # Handle constant columns
            else:
                norm[:, j] = (col - mn) / (mx - mn)

        scores = np.sum(norm * weights, axis=1) * 100
        return np.nan_to_num(scores)

# ==================== METHOD 2: BWM + VIKOR ====================
class BWM_VIKOR:
    def solve_bwm_weights(self, best, worst, best_others, others_worst):
        def obj(w):
            max_dev = 0
            for i in range(4):
                if i != best:
                    # Prevent div by zero
                    w_i = w[i] if w[i] != 0 else 1e-9
                    dev = abs(w[best] / w_i - best_others[i])
                    max_dev = max(max_dev, dev)
                if i != worst:
                    w_worst = w[worst] if w[worst] != 0 else 1e-9
                    dev = abs(w[i] / w_worst - others_worst[i])
                    max_dev = max(max_dev, dev)
            return max_dev

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0.001, 1) for _ in range(4)]
        w0 = np.ones(4) / 4
        res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons)
        return res.x

    def calculate_vikor(self, matrix, weights):
        n, m = matrix.shape
        ideal_best = np.max(matrix, axis=0)
        ideal_worst = np.min(matrix, axis=0)
        
        S = np.zeros(n)
        R = np.zeros(n)
        
        for i in range(n):
            dists = []
            for j in range(m):
                denom = ideal_best[j] - ideal_worst[j]
                if denom == 0: denom = 1 # Handle zero variance
                dist = weights[j] * (ideal_best[j] - matrix[i,j]) / denom
                dists.append(dist)
            S[i] = sum(dists)
            R[i] = max(dists) if dists else 0
            
        S_min, S_max = min(S), max(S)
        R_min, R_max = min(R), max(R)
        
        Q = np.zeros(n)
        v = 0.5
        
        for i in range(n):
            term1 = (S[i]-S_min)/(S_max-S_min) if (S_max-S_min) != 0 else 0
            term2 = (R[i]-R_min)/(R_max-R_min) if (R_max-R_min) != 0 else 0
            Q[i] = v*term1 + (1-v)*term2
            
        return np.nan_to_num((1 - Q) * 100)

# ==================== METHOD 3: SWARA + MOORA ====================
class SWARA_MOORA:
    def calculate_swara_weights(self, sorted_criteria, comparative_scores):
        if not comparative_scores: comparative_scores = [0.1, 0.1, 0.1]
        
        q = [1.0] * 4
        for i in range(1, 4):
            # Safe access to comparative_scores
            s_j = comparative_scores[i-1] if (i-1) < len(comparative_scores) else 0.1
            k_j = s_j + 1
            q[i] = q[i-1] / k_j if k_j != 0 else q[i-1]
            
        total = sum(q)
        if total == 0: return {k: 0.25 for k in sorted_criteria}
        
        # Ensure we map back to the right keys
        # We need all 4 keys. If sorted_criteria is short, we pad.
        full_keys = sorted_criteria + [k for k in ["spm_results","previous_semester","technical_skills","aptitude_test"] if k not in sorted_criteria]
        
        return {k: q[i]/total for i, k in enumerate(full_keys[:4])}

    def calculate_moora(self, matrix, weights):
        # 1. Vector Norm
        norm1 = np.zeros_like(matrix)
        for j in range(4):
            denom = np.sqrt(np.sum(matrix[:, j]**2))
            norm1[:, j] = matrix[:, j] / denom if denom > 0 else 0
            
        # 2. Sum Norm
        norm2 = np.zeros_like(matrix)
        for j in range(4):
            denom = np.sum(matrix[:, j])
            norm2[:, j] = matrix[:, j] / denom if denom > 0 else 0
            
        # 3. Max Norm
        norm3 = np.zeros_like(matrix)
        for j in range(4):
            denom = np.max(matrix[:, j])
            norm3[:, j] = matrix[:, j] / denom if denom > 0 else 0
            
        def score(n, w): return np.sum(n * w, axis=1)
        
        m1, m2, m3 = score(norm1, w=weights), score(norm2, w=weights), score(norm3, w=weights)
        
        # Tchebycheff
        def tcheb(n, w):
            ideal = np.max(n, axis=0)
            d = []
            for i in range(len(n)):
                d.append(np.max(np.abs(ideal - n[i]) * w))
            return np.array(d)
            
        t1, t2, t3 = tcheb(norm1, weights), tcheb(norm2, weights), tcheb(norm3, weights)
        
        final = (m1 - t1) + (m2 - t2) + (m3 - t3)
        
        mn, mx = np.min(final), np.max(final)
        if mx - mn == 0: return np.full(len(final), 100)
        
        return np.nan_to_num((final - mn) / (mx - mn) * 100)

# ==================== METHOD 4: LTSF-CRITIC-EDAS (FIXED) ====================
class CRITIC_EDAS:
    def __init__(self):
        self.tsf = TSphericalFuzzyHamacher()

    def execute(self, matrix):
        # 1. LTSF Conversion
        tsf_matrix = []
        for i in range(len(matrix)):
            row = []
            for j in range(4):
                val = matrix[i,j] / 100.0
                if val >= 0.9: t = LINGUISTIC_TERMS['Extremely Good']
                elif val >= 0.8: t = LINGUISTIC_TERMS['Very Good']
                elif val >= 0.7: t = LINGUISTIC_TERMS['Good']
                elif val >= 0.6: t = LINGUISTIC_TERMS['Slightly Good']
                elif val >= 0.5: t = LINGUISTIC_TERMS['Fair']
                elif val >= 0.4: t = LINGUISTIC_TERMS['Slightly Poor']
                elif val >= 0.3: t = LINGUISTIC_TERMS['Poor']
                elif val >= 0.2: t = LINGUISTIC_TERMS['Very Poor']
                else: t = LINGUISTIC_TERMS['Extremely Poor']
                row.append(t)
            tsf_matrix.append(row)
            
        # 2. Score Calculation
        score_mat = np.zeros_like(matrix)
        for i in range(len(matrix)):
            for j in range(4):
                score_mat[i, j] = self.tsf.score_function(tsf_matrix[i][j])
                
        # 3. CRITIC (with NaN Protection)
        weights = self.critic_method(score_mat)
        
        # 4. EDAS (with NaN Protection)
        edas = self.edas_method(score_mat, weights)
        
        return np.nan_to_num(edas * 100), weights

    def critic_method(self, matrix):
        # Normalize
        mn = np.min(matrix, axis=0)
        mx = np.max(matrix, axis=0)
        norm = np.zeros_like(matrix)
        
        for j in range(4):
            rng = mx[j] - mn[j]
            if rng == 0:
                norm[:, j] = 0.5 # Constant column
            else:
                norm[:, j] = (matrix[:, j] - mn[j]) / rng
                
        std = np.std(norm, axis=0)
        
        # Correlation with safety
        try:
            # Handle constant columns (std=0) which cause NaN in correlation
            # We add a tiny noise to prevent NaN if variance is 0
            if np.any(std == 0):
                noisy_norm = norm + np.random.normal(0, 1e-9, norm.shape)
                corr = np.corrcoef(noisy_norm, rowvar=False)
            else:
                corr = np.corrcoef(norm, rowvar=False)
                
            # If single column or scalar result
            if np.ndim(corr) == 0: corr = np.array([[1]])
            
            # Clean any remaining NaNs
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)
            
        except Exception:
            corr = np.eye(4) # Fallback to identity
            
        info = std * np.sum(1 - corr, axis=1)
        total_info = np.sum(info)
        
        if total_info == 0:
            return np.ones(4) / 4 # Equal weights fallback
            
        return info / total_info

    def edas_method(self, matrix, weights):
        avg = np.mean(matrix, axis=0)
        # Avoid division by zero in average
        avg = np.where(avg == 0, 1e-9, avg)
        
        pda = np.maximum(0, (matrix - avg)) 
        nda = np.maximum(0, (avg - matrix))
        
        # Normalize by Average
        for j in range(4):
            pda[:, j] /= avg[j]
            nda[:, j] /= avg[j]
            
        sp = np.sum(pda * weights, axis=1)
        sn = np.sum(nda * weights, axis=1)
        
        # Normalize SP/SN
        max_sp = np.max(sp)
        max_sn = np.max(sn)
        
        nsp = sp / max_sp if max_sp != 0 else sp
        nsn = 1 - (sn / max_sn) if max_sn != 0 else np.ones_like(sn)
        
        as_score = 0.5 * (nsp + nsn)
        return as_score

class TSphericalFuzzyHamacher:
    def __init__(self, gamma=1.0):
        self.gamma = max(gamma, 0.1)
    def score_function(self, tsf):
        m, nm, h = tsf
        return m**3 - nm**3 + h**3