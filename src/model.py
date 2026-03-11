from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

def train_tier_models(X_train, y_train, tiers):
    models = {}
    for name, features in tiers.items():
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=100)
        clf.fit(X_train[features], y_train)
        models[name] = clf
    return models

def extract_decision_tree_rules(X_train, y_train, features, max_depth=3):
    dt_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42, criterion='entropy')
    dt_model.fit(X_train[features], y_train)
    tree_rules = export_text(dt_model, feature_names=features)
    return dt_model, tree_rules