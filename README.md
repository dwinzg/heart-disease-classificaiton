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
│   ├── data_loader.py      # Preprocessing, cleaning, and tier-mapping
│   ├── model.py            # Random Forest & Decision Tree architectures
│   ├── triage_logic.py     # The Cascading/Sequential decision engine
│   ├── stats_engine.py     # DeLong Test and Bootstrapping logic
│   └── visualize.py        # SHAP, ROC, and Decision Tree plotting
├── notebooks/              # Original research and EDA
├── results/                # Exported CSVs and static plot images
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

_Created for the UBC Medicine Datathon 2026. Credits towards Darwin Zhang, Swann Choi, Christy Wu, Wendy Liao, Naomi, and Sahil._
