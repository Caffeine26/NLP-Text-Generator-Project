from transformers import pipeline
from preprocess import clean_text

with open("data/challenge_data.txt", "r", encoding="utf-8") as f:
    reviews = [line.strip() for line in f.readlines()]

print("Total reviews:", len(reviews))
reviews_clean = [clean_text(r) for r in reviews]
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
predictions = classifier(reviews_clean, batch_size=32)

labels = ["1" if p["label"] == "POSITIVE" else "0" for p in predictions]

output_string = "".join(labels)
print("Output length:", len(output_string))

assert len(output_string) == 5000, "Output length is not 5000!"
with open("group15_mini_project_2_challenge.txt", "w") as f:
    f.write(output_string)
print(" File saved successfully!")
