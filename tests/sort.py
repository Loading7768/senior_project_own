import json

with open("../data/ml/classification/logistic_regression_keyword_coefficients.json", "r", encoding="utf-8=sig") as file:
    feature_names_importance = json.load(file)

sorted_importance = sorted(feature_names_importance.items(), key=lambda x: x[1], reverse=True)
sorted_dict = {k: v for k, v in sorted_importance}

with open("../data/ml/classification/logistic_regression_keyword_coefficients.json", "w", encoding="utf-8=sig") as file:
    json.dump(sorted_dict, file, ensure_ascii=False, indent=4)