# from openai import OpenAI
# import pickle

# client = OpenAI(api_key="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")

# words = words = [
#     "illness", "friend", "paper", "war", "trust", "pen", "joy", "color", "kindness",
#     "failure", "love", "wall", "hope", "regret", "smile", "nightmare", "door", "peace",
#     "book", "warmth", "pain", "sunshine", "number", "fear", "floor", "laughter",
#     "crying", "delight", "keyboard", "anger", "comfort", "bottle", "loneliness",
#     "blessing", "death", "shape", "achievement", "object", "grief", "chair", "hate",
#     "window", "success", "breakup", "happiness", "table", "violence", "holiday", "freedom",
#     "music", "book", "read", "dog", "bark", "car", "drive", "student", "study", "phone", "call",
#     "cake", "eat", "hammer", "hit", "light", "shine", "rain", "fall", "school", "learn",
#     "bell", "ring", "ball", "throw", "song", "sing", "pen", "write", "door", "open",
#     "game", "play", "bell", "chime", "leaf", "blow", "shoe", "walk", "horn", "honk",
#     "rope", "tie", "email", "send", "alarm", "beep", "chair", "sit", "picture", "draw"
# ]

# embeddings = {}
# for w in words:
#     resp = client.embeddings.create(model="text-embedding-3-small", input=w)
#     embeddings[w] = resp.data[0].embedding

# with open("embeddings.pkl", "wb") as f:
#     pickle.dump(embeddings, f)


import pickle

# Example: Assuming 250 trials per word for a 12500-sample dataset (50 words * 250 trials = 12500)
num_trials_per_word = 250
pos_words = (
    ["book"] * num_trials_per_word +
    ["read"] * num_trials_per_word +
    ["dog"] * num_trials_per_word +
    ["bark"] * num_trials_per_word +
    ["car"] * num_trials_per_word +
    ["drive"] * num_trials_per_word +
    ["student"] * num_trials_per_word +
    ["study"] * num_trials_per_word +
    ["phone"] * num_trials_per_word +
    ["call"] * num_trials_per_word +
    ["cake"] * num_trials_per_word +
    ["eat"] * num_trials_per_word +
    ["hammer"] * num_trials_per_word +
    ["hit"] * num_trials_per_word +
    ["light"] * num_trials_per_word +
    ["shine"] * num_trials_per_word +
    ["rain"] * num_trials_per_word +
    ["fall"] * num_trials_per_word +
    ["school"] * num_trials_per_word +
    ["learn"] * num_trials_per_word +
    ["bell"] * num_trials_per_word +
    ["ring"] * num_trials_per_word +
    ["ball"] * num_trials_per_word +
    ["throw"] * num_trials_per_word +
    ["song"] * num_trials_per_word +
    ["sing"] * num_trials_per_word +
    ["pen"] * num_trials_per_word +
    ["write"] * num_trials_per_word +
    ["door"] * num_trials_per_word +
    ["open"] * num_trials_per_word +
    ["game"] * num_trials_per_word +
    ["play"] * num_trials_per_word +
    ["bell"] * num_trials_per_word +
    ["chime"] * num_trials_per_word +
    ["leaf"] * num_trials_per_word +
    ["blow"] * num_trials_per_word +
    ["shoe"] * num_trials_per_word +
    ["walk"] * num_trials_per_word +
    ["horn"] * num_trials_per_word +
    ["honk"] * num_trials_per_word +
    ["rope"] * num_trials_per_word +
    ["tie"] * num_trials_per_word +
    ["email"] * num_trials_per_word +
    ["send"] * num_trials_per_word +
    ["alarm"] * num_trials_per_word +
    ["beep"] * num_trials_per_word +
    ["chair"] * num_trials_per_word +
    ["sit"] * num_trials_per_word +
    ["picture"] * num_trials_per_word +
    ["draw"] * num_trials_per_word
)

with open('pos_upd_words.pkl', 'wb') as f:
    pickle.dump(pos_words, f)