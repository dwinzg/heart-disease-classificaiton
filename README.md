# Cardio-Economics: Optimizing Clinical Triage via Cascading ML

Traditional heart disease diagnosis is often a "one-size-fits-all" process that leads to clinical bottlenecks and high patient costs. This repository introduces a Cascading Machine Learning Architecture that prioritizes patient safety and resource efficiency. By auditing the "Information Lift" of standard clinical tests, this system identifies the most optimal diagnostic path for each patient.

## The Hook: Clinical Efficiency at Scale

Current protocols often mandate expensive imaging and slow-turnaround blood labs for every patient. Our analysis proves that 80% of diagnostic costs can be eliminated by utilizing a dynamic triage funnel.

* **Total Savings:** $191,750 (Calculated across a 75-patient test set)
* **Intake Efficiency:** 46.7% of patients were successfully triaged at the "Baseline" interview, requiring zero expensive tests.
* **Statistical Rigor:** All model comparisons were validated using DeLong’s Test to identify redundant clinical steps.

## Performance by Tier

We mapped clinical variables into five progressive tiers. While accuracy (AUC) increases with more data, the marginal gain of certain tiers (like Blood Labs) was found to be statistically insignificant for immediate triage.

| Clinical Tier | AUC | 95% CI Lower | 95% CI Upper |
| --- | --- | --- | --- |
| **Tier 1: Baseline** | 0.799 | 0.692 | 0.891 |
| **Tier 2: Vitals** | 0.780 | 0.670 | 0.878 |
| **Tier 3: Blood Lab** | 0.810 | 0.703 | 0.902 |
| **Tier 4: Stress Test** | 0.858 | 0.766 | 0.936 |
| **Tier 5: Specialized** | 0.889 | 0.810 | 0.955 |

## Project Structure

The source code is modularized into specialized scripts to handle the end-to-end pipeline:

```text
├── data/                   # Raw and processed clinical datasets
├── src/                    # Refactored .py source files
│   ├── __init__.py         # Package initializer
│   ├── data_loader.py      # Preprocessing, cleaning, and tier-mapping
│   ├── model.py            # Random Forest & Decision Tree architectures
│   ├── triage_logic.py     # The Cascading/Sequential decision engine
│   ├── stats_engine.py     # DeLong Test and Bootstrapping logic
│   └── visualize.py        # SHAP, ROC, and Decision Tree plotting
├── results/                # Exported CSVs, Markdown report, and static plot images
├── main.py                 # Entry point to run the full simulation
└── requirements.txt        # Project dependencies

```

## How to Run

To reproduce the findings and the Cascading Triage report:

1. **Clone the Repository:**
```bash
git clone https://github.com/dwinzg/heart-disease-classification.git
cd heart-disease-classification

```


2. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


3. **Execute the Pipeline:**
```bash
python main.py

```



## Key Methodology

* **Cascading Triage:** A multi-step decision engine that "stops" as soon as a high-confidence prediction is made, reserving expensive tests (Tier 4 & 5) only for "Gray Zone" cases.
* **DeLong Testing:** Statistically compared the "lift" of each tier to ensure we only recommend tests that provide unique, significant information.
* **Clinical Interpretability:** Utilizing SHAP values and Decision Trees to ensure the model’s logic is transparent and aligns with human cardiovascular physiology.

## Limitations and Ethical Considerations
While this analysis employs financial proxies to evaluate clinical utility, we recognize the inherent ethical complexity of assigning monetary value to health outcomes. This project is designed as an exploratory data science framework intended to identify the marginal diagnostic value of specific tests within a constrained triage environment.

The objective of this study is to highlight high-signal variables and identify potential areas for resource optimization; it is not intended to prescribe healthcare policy or offer clinical shortcuts. We acknowledge that medical decision-making requires a holistic integration of patient history, professional intuition, and ethical standards that go beyond statistical modeling. As such, these findings should be viewed as a proof-of-concept for feature-utility mapping and not as a replacement for the established protocols and specialized judgment of healthcare professionals.

## Refined Technical Appendix
Constraints: 
* **Sample Size and Power:** The Cleveland dataset’s limited sample size ($N=297$) restricts the statistical power of the DeLong tests. The near-significant p-values for Tier 4 ($0.09$) and Tier 5 ($0.06$) suggest that in a larger clinical cohort, these specialized tests would likely demonstrate a statistically significant diagnostic lift.
* **Proxy Assumptions:** The cost and time values assigned to each tier are estimated proxies used to simulate a cost-benefit analysis. Actual clinical costs vary significantly by region, institution, and insurance framework.
* **Demographic Skew:** As an analysis grounded in "Gold Standard" datasets, we must acknowledge that these benchmarks often contain historical biases. Heart disease frequently presents differently across sex and ethnic demographics, and a model relying heavily on specific features like "Typical Angina" may not generalize perfectly to all patient populations.

_Created for the UBC Medicine Datathon 2026. Credits towards Darwin Zhang, Swann Choi, Christy Wu, Wendy Liao, Naomi, and Sahil._
