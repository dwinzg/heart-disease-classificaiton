import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import shap

def plot_roc_curves(all_roc_df, tiers):
    """
    Generates and saves the ROC Curve comparison chart.
    """
    line = alt.Chart(all_roc_df).mark_line().encode(
        x=alt.X('False Positive Rate:Q'),
        y=alt.Y('True Positive Rate:Q'),
        color=alt.Color('Tier:N', sort=list(tiers.keys())),
        tooltip=['Tier', 'False Positive Rate', 'True Positive Rate']
    ).properties(title='Triage Performance: ROC Curve Comparison', width=600, height=450)

    # The diagonal random-chance line (AUC = 0.5)
    guide = alt.Chart(pd.DataFrame({'x': [0, 1], 'y': [0, 1]})).mark_line(strokeDash=[5,5], color='gray').encode(x='x', y='y')

    dashboard = (line + guide)
    
    # In a python script, we save the dashboard instead of displaying it in-line
    dashboard.save('results/roc_curves.html')
    print("Saved ROC curves to results/roc_curves.html")
    return dashboard

def plot_decision_tree(dt_model, features):
    """
    Plots and saves the static Decision Tree flowchart.
    """
    plt.figure(figsize=(20,10))
    tree.plot_tree(dt_model, 
                   feature_names=features,  
                   class_names=["Healthy", "Disease"],
                   filled=True, 
                   rounded=True, 
                   fontsize=12)
    plt.title("Optimal Triage Decision Tree (Max Depth = 3)", fontsize=18)
    plt.savefig('results/decision_tree.png')
    plt.close()
    print("Saved Decision Tree plot to results/decision_tree.png")

def plot_shap_summary(model, X_test):
    """
    Generates and saves the SHAP summary plot.
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)
    
    # Handle SHAP dimensionality checks
    if isinstance(shap_vals, list):
        shap_for_class_1 = shap_vals[1]
    elif len(shap_vals.shape) == 3:
        shap_for_class_1 = shap_vals[:, :, 1]
    else:
        shap_for_class_1 = shap_vals

    plt.figure(figsize=(10, 6))
    plt.title("Clinical Triage: Key Discriminating Factors (SHAP)", fontsize=14, pad=20)
    shap.summary_plot(shap_for_class_1, X_test, show=False)
    plt.tight_layout()
    plt.savefig('results/shap_summary.png')
    plt.close()
    print("Saved SHAP summary to results/shap_summary.png")