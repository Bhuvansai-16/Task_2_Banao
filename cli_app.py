from dag_graph import InferenceNode, ConfidenceCheckNode, FallbackNode
from logger import log_event

def main():
    print("=== Self-Healing Text Classifier ===")
    inference = InferenceNode()
    checker = ConfidenceCheckNode()
    fallback = FallbackNode()

    while True:
        text = input("\nEnter text (or 'exit'): ")
        if text.lower() == "exit":
            break

        label, confidence = inference.run(text)
        if checker.run(confidence):
            print(f"Final Label: {label} (Confidence: {confidence:.2f})")
            log_event("FinalDecision", f"Final Label: {label}")
        else:
            label = fallback.run(text)
            print(f"Final Label: {label} (Corrected via fallback)")
            log_event("FinalDecision", f"Final Label: {label} (Corrected)")

if __name__ == "__main__":
    main()