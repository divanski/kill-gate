# Freeze 20 fixed prompts BEFORE any measurement. Mix of short factual and
# short reasoning questions, generic, no employer data. Written once; never
# edited after seeing outputs.
import json
import os

PROMPTS = [
    "What is the capital of France?",
    "Explain in one sentence why the sky is blue.",
    "What is 17 times 24?",
    "Name three primary colours.",
    "What year did the Berlin Wall fall?",
    "Summarise the plot of Romeo and Juliet in two sentences.",
    "What is the boiling point of water in Celsius at sea level?",
    "Explain the difference between a virus and a bacterium in one sentence.",
    "What is the largest planet in the solar system?",
    "If a train travels 60 km in 1.5 hours, what is its average speed in km/h?",
    "Name the chemical symbol for gold.",
    "Explain briefly why leaves change colour in autumn.",
    "What is the square root of 144?",
    "Who wrote the novel 'Pride and Prejudice'?",
    "Explain in one sentence what a firewall does in networking.",
    "What is the currency of Japan?",
    "If today is Wednesday, what day of the week will it be in 10 days?",
    "Name two gases that make up most of Earth's atmosphere.",
    "Explain briefly the difference between weather and climate.",
    "What is the freezing point of water in Fahrenheit?",
]

assert len(PROMPTS) == 20

out = {"prompts": [{"id": i + 1, "text": p} for i, p in enumerate(PROMPTS)]}
here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "prompts.json")
if os.path.exists(path):
    print("prompts.json already exists, not overwriting:", path)
else:
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2)
    print("froze", len(PROMPTS), "prompts to", path)
