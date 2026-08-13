import json
import os

path = r"C:\Users\naysh\.gemini\antigravity-ide\brain\5a093378-fe4f-419f-9632-045d1cc71b9c\.system_generated\logs\transcript.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if "content" in data:
            content = data["content"]
            if "test_ap:" in content or "test_ap" in content or "0.73" in content:
                print(f"STEP: {data.get('step_index')}")
                print(content[:500])
                print("-" * 50)
