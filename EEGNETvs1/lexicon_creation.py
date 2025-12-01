import pickle

# Define the words and their corresponding labels
words = [
    "illness", "friend", "paper", "war", "trust", "pen", "joy", "color", "kindness",
    "failure", "love", "wall", "hope", "regret", "smile", "nightmare", "door", "peace",
    "book", "warmth", "pain", "sunshine", "number", "fear", "floor", "laughter",
    "crying", "delight", "keyboard", "anger", "comfort", "bottle", "loneliness",
    "blessing", "death", "shape", "achievement", "object", "grief", "chair", "hate",
    "window", "success", "breakup", "happiness", "table", "violence", "holiday", "freedom",
    "music"
]
labels = [0, 2, 1, 0, 2, 1, 2, 1, 2, 0, 2, 1, 2, 0, 2, 0, 1, 2, 1, 2,
          0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 2, 1, 0, 2, 0, 1, 2, 1, 0, 1,
          0, 1, 2, 0, 2, 1, 0, 2, 2, 2]

# Define valence mapping
valence_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Define emotional/semantic categories
categories = [
    "Sadness", "Friendship", "Object", "Conflict", "Trust", "Object", "Happiness",
    "Object", "Kindness", "Failure", "Love", "Object", "Hope", "Regret", "Happiness",
    "Fear", "Object", "Peace", "Object", "Warmth", "Pain", "Happiness", "Object",
    "Fear", "Object", "Happiness", "Sadness", "Happiness", "Object", "Anger",
    "Comfort", "Object", "Sadness", "Blessing", "Death", "Object", "Achievement",
    "Object", "Sadness", "Object", "Anger", "Object", "Success", "Sadness",
    "Happiness", "Object", "Conflict", "Happiness", "Freedom", "Happiness"
]

# Define emotional intensity (0 to 1, estimated based on emotional strength)
intensities = [
    0.8, 0.7, 0.3, 0.9, 0.6, 0.3, 0.9, 0.4, 0.7, 0.8, 0.9, 0.3, 0.7, 0.7,
    0.6, 0.8, 0.3, 0.7, 0.3, 0.6, 0.8, 0.7, 0.3, 0.8, 0.3, 0.7, 0.8, 0.7,
    0.3, 0.8, 0.6, 0.3, 0.8, 0.7, 0.9, 0.3, 0.7, 0.3, 0.8, 0.3, 0.8, 0.3,
    0.7, 0.8, 0.9, 0.3, 0.9, 0.7, 0.7, 0.7
]

# Define part of speech (POS)
pos_tags = [
    "Noun", "Noun", "Noun", "Noun", "Noun", "Noun", "Noun", "Noun", "Noun",
    "Noun", "Noun", "Noun", "Noun", "Noun", "Noun", "Noun", "Noun", "Noun",
    "Noun", "Noun", "Noun", "Noun", "Noun", "Noun", "Noun", "Noun", "Noun",
    "Noun", "Noun", "Noun", "Noun", "Noun", "Noun", "Noun", "Noun", "Noun",
    "Noun", "Noun", "Noun", "Noun", "Noun", "Noun", "Noun", "Noun", "Noun",
    "Noun", "Noun", "Noun", "Noun", "Noun"
]

# Define frequency estimate (High, Medium, Low)
frequencies = [
    "Medium", "High", "High", "Medium", "Medium", "High", "Medium", "High", "Medium",
    "Medium", "High", "High", "Medium", "Medium", "High", "Medium", "High", "Medium",
    "High", "Medium", "Medium", "Medium", "High", "Medium", "High", "Medium",
    "Medium", "Medium", "Medium", "Medium", "Medium", "High", "Medium", "Medium",
    "Medium", "High", "Medium", "High", "Medium", "High", "Medium", "High",
    "Medium", "High", "High", "High", "Medium", "Medium", "Medium", "High"
]

# Create the advanced lexicon
lexicon = {}
for idx, (word, label, category, intensity, pos, freq) in enumerate(
    zip(words, labels, categories, intensities, pos_tags, frequencies)
):
    lexicon[word] = {
        "label": label,
        "valence": valence_map[label],
        "category": category,
        "index": idx,
        "intensity": intensity,
        "pos": pos,
        "word_length": len(word),
        "frequency": freq
    }

# Save the lexicon to a .pkl file
with open("valence_lexicon.pkl", "wb") as f:
    pickle.dump(lexicon, f)

print("Enhanced advanced lexicon saved to enhanced_advanced_emotion_lexicon.pkl")