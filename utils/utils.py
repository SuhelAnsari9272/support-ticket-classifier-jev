import numpy as np

def extract_choice_interpretation(choice_answer):

    probabilities = choice_answer.probabilities

    # Convert probabilities into numpy array
    probs = np.array(list(probabilities.values()), dtype=float)

    # Normalize probabilities (safety check)
    probs = probs / probs.sum()

    # 1. Maximum probability and predicted choice
    max_idx = np.argmax(probs)

    choice = list(probabilities.keys())[max_idx]

    # 2. Shannon entropy
    entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))

    # 3. Normalized entropy
    n = len(probs)

    normalized_entropy = (entropy / np.log2(n) if n > 1 else 0.0)

    # 4. Confidence based on normalized entropy
    confidence = 1 - normalized_entropy

    return {
        "choice": choice,
        "confidence": round(float(confidence), 4),
        "entropy": round(float(entropy), 4),
        "normalized_entropy": round(float(normalized_entropy), 4),
    }

def extract_score_interpretation(score_answer):

    output = {
        f"score": score_answer.score, 
        f"confidence": score_answer.confidence 
    }
    probabilities = score_answer.probabilities
    legend = score_answer.legend

    # Highest probability
    max_prob_key = max(probabilities, key=probabilities.get)

    # Nearest legend index
    score_key = round(score_answer.score)

    output[f'key'] =  legend[max_prob_key]

    if max_prob_key != score_key:
        output["approximate_key"] = legend[score_key]

    return output
