
from transformers import pipeline

MODEL_NAME = "unitary/toxic-bert"

print("Loading model...")

detector = pipeline(
    "text-classification",
    model=MODEL_NAME,
    top_k=None
)

print("Model loaded successfully!")


def detect_comment(comment):
    comment = comment.strip()

    if not comment:
        print("Please enter a comment.")
        return

    # Get scores for all toxicity categories
    results = detector(comment, truncation=True)[0]

    scores = {
        item["label"].lower(): item["score"]
        for item in results
    }

    # Display category scores
    print("\nComment:", comment)
    print("\nToxicity category scores:")

    for label, score in sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"{label}: {score:.4f}")

    # Use the general toxicity category
    toxic_score = scores.get("toxic")

    if toxic_score is None:
        print("\nCould not find the 'toxic' output category.")
        return

    threshold = 0.5

    if toxic_score >= threshold:
        print("\nResult: TOXIC COMMENT")
    else:
        print("\nResult: NON-TOXIC COMMENT")

    print(f"Toxicity score: {toxic_score * 100:.2f}%")


# Test examples
test_comments = [
    "This page is incredible!",
    "You are horrible and nobody likes you",
    "I will ruin your life",
    "Thank you for helping me",
    "Shut up, you freak"
]

for comment in test_comments:
    detect_comment(comment)
    print("-" * 50)


# Enter your own comments
while True:
    comment = input(
        "\nEnter a comment (type exit to quit): "
    )

    if comment.strip().lower() == "exit":
        break

    detect_comment(comment)
