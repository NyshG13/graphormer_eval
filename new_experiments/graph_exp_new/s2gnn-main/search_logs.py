import os

search_dir = r"c:\Users\naysh\Documents\graphormer_eval\new_experiments\graph_exp_new"
for root, dirs, files in os.walk(search_dir):
    for file in files:
        if file.endswith((".log", ".txt", ".md", ".csv")):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "test_ap" in content or "test ap" in content.lower():
                        print(f"Found in {filepath}")
            except Exception:
                pass
