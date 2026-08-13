import json
import os

path = r"C:\Users\naysh\.gemini\antigravity-ide\brain\5a093378-fe4f-419f-9632-045d1cc71b9c\.system_generated\logs\transcript.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if "content" in data:
            content = data["content"]
            if "s2gnn" in content.lower() and ("0.7" in content or "ap" in content.lower() or "score" in content.lower()):
                print(f"STEP: {data.get('step_index')}")
                print(content[:500])
                print("-" * 50)
