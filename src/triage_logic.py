import pandas as pd
from sklearn.metrics import accuracy_score

def evaluate_cascading_pipeline(X_test, y_test, models, fast_track_tiers):
    results = []
    t1_lower, t1_upper = 0.20, 0.80 
    t2_lower, t2_upper = 0.35, 0.65 

    for i in range(len(X_test)):
        patient = X_test.iloc[[i]]
        
        # Step 1: Intake
        p1 = models["Step 1: Intake"].predict_proba(patient[fast_track_tiers["Step 1: Intake"]])[:, 1][0]
        if p1 <= t1_lower or p1 >= t1_upper:
            results.append({"prediction": round(p1), "final_step": "Step 1: Intake", "cost": 20})
            continue
            
        # Step 2: Stress Test
        p2 = models["Step 2: Stress Test"].predict_proba(patient[fast_track_tiers["Step 2: Stress Test"]])[:, 1][0]
        if p2 <= t2_lower or p2 >= t2_upper:
            results.append({"prediction": round(p2), "final_step": "Step 2: Stress Test", "cost": 520})
            continue
            
        # Step 3: Imaging
        p3 = models["Step 3: Imaging"].predict_proba(patient[fast_track_tiers["Step 3: Imaging"]])[:, 1][0]
        results.append({"prediction": round(p3), "final_step": "Step 3: Imaging", "cost": 3020})

    results_df = pd.DataFrame(results)
    results_df['actual'] = y_test.values
    
    total_cost = results_df['cost'].sum()
    accuracy = accuracy_score(results_df['actual'], results_df['prediction'])
    return results_df, accuracy, total_cost