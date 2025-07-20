# A=(verbs pronouns 0)
# B=(pronouns verbs 1)
# C=(sports sentiment 1)
# D=(pronouns sentiment 1)
# E=(sentiment sports 0)
# F=(verbs sports 0)

# G=(sentiment pronouns 0)
# H=(sports verbs 1)
# I=(pronouns sports 0)
# J=(verbs sentiment 0)
# K=(sports pronouns 1)
# L=(sentiment verbs 1)

TEMPLATE = {
    "model.layers.0": [],
    "model.layers.2": [], 
    "model.layers.4": [],
    "model.layers.6": [], 
    "model.layers.8": [],
    "model.layers.10": [], 
    "model.layers.12": [],
    "model.layers.14": [],
    "model.layers.16": [],
    "model.layers.18": [],
    "model.layers.20": [],
    "model.layers.22": [],
    "model.layers.24": [],
    "model.layers.26": [],
    "model.layers.28": [],
    "model.layers.30": [],
    "model.layers.32": [],
}

verbs_pronouns_features = {
    "model.layers.0": [],
    "model.layers.2": [], 
    "model.layers.4": [],
    "model.layers.6": [], 
    "model.layers.8": [28722],
    "model.layers.10": [], 
    "model.layers.12": [],
    "model.layers.14": [],
    "model.layers.16": [],
    "model.layers.18": [],
    "model.layers.20": [],
    "model.layers.22": [],
    "model.layers.24": [],
    "model.layers.26": [],
    "model.layers.28": [],
    "model.layers.30": [],
    "model.layers.32": [],
}


sports_verbs_features = {
    "model.layers.0": [3409],
    "model.layers.2": [1535], # "since", used in sports a bunch?
    "model.layers.4": [],
    "model.layers.6": [2729, 10562, 10960],  # first 2 are proper nouns
    "model.layers.8": [29333],
    "model.layers.10": [849], # proper noun
    "model.layers.12": [],
    "model.layers.14": [],
    "model.layers.16": [21453],
    "model.layers.18": [],
    "model.layers.20": [],
    "model.layers.22": [13769],
    "model.layers.24": [],
    "model.layers.26": [27442],
    "model.layers.28": [],
    "model.layers.30": [],
    "model.layers.32": [],
}

verbs_sentiment_features = {
    "model.layers.0": [],
    "model.layers.2": [], 
    "model.layers.4": [],
    "model.layers.6": [], 
    "model.layers.8": [],
    "model.layers.10": [], 
    "model.layers.12": [],
    "model.layers.14": [],
    "model.layers.16": [],
    "model.layers.18": [],
    "model.layers.20": [],
    "model.layers.22": [],
    "model.layers.24": [],
    "model.layers.26": [],
    "model.layers.28": [],
    "model.layers.30": [],
    "model.layers.32": [],
}

exported_features = {
    "sports_verbs_features" : sports_verbs_features,
}