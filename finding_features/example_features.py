# Sports features, intended pronouns
pronouns_sports_features = {
    "model.layers.0" : [],
    "model.layers.2" : [],
    "model.layers.4" : [5],
    "model.layers.6" : [2, 4, 5, 6],
    "model.layers.8" : [2, 3, 4],
    "model.layers.10" : [5, 6, 8, 13],
    "model.layers.12" : [7],
    "model.layers.14" : [5, 9],
    "model.layers.16" : [3, 6, 7], # maybe 3
    "model.layers.18" : [7, 8],
    "model.layers.20" : [7, 8, 9],
    "model.layers.22" : [4, 5, 6, 7, 8, 9],
    "model.layers.24" : [7],
}

# Pronouns features, intended sports
sports_pronouns_features = {
    "model.layers.0" : [],
    "model.layers.2" : [],
    "model.layers.4" : [],
    "model.layers.6" : [],
    "model.layers.8" : [],
    "model.layers.10" : [9],
    "model.layers.12" : [18],
    "model.layers.14" : [],
    "model.layers.16" : [],
    "model.layers.18" : [16, 17, 18],
    "model.layers.20" : [],
    "model.layers.22" : [17],
    "model.layers.24" : [14 ],
}

# Pronouns features, intended sentiment
sentiment_pronouns_features = {
    "model.layers.0" : [],
    "model.layers.2" : [],
    "model.layers.4" : [],
    "model.layers.6" : [],
    "model.layers.8" : [],
    "model.layers.10" : [],
    "model.layers.12" : [],
    "model.layers.14" : [],
    "model.layers.16" : [],
    "model.layers.18" : [],
    "model.layers.20" : [],
    "model.layers.22" : [],
    "model.layers.24" : [],
}

# sentiment features, intended verbs
sentiment_verbs_features = {
    "model.layers.0" : [],
    "model.layers.2" : [],
    "model.layers.4" : [],
    "model.layers.6" : [],
    "model.layers.8" : [],
    "model.layers.10" : [],
    "model.layers.12" : [],
    "model.layers.14" : [],
    "model.layers.16" : [],
    "model.layers.18" : [],
    "model.layers.20" : [],
    "model.layers.22" : [],
    "model.layers.24" : [],
}

# verbs features, intended sentiment
verbs_sentiment_features = {
    "model.layers.0" : [],
    "model.layers.2" : [],
    "model.layers.4" : [],
    "model.layers.6" : [],
    "model.layers.8" : [],
    "model.layers.10" : [],
    "model.layers.12" : [],
    "model.layers.14" : [],
    "model.layers.16" : [0, 1, 6],
    "model.layers.18" : [1, 3],
    "model.layers.20" : [1, 2],
    "model.layers.22" : [1, 2],
    "model.layers.24" : [0, 1, 2],
}

# pronouns features, intended verbs
verbs_pronouns_features = {
    "model.layers.0" : [],
    "model.layers.2" : [],
    "model.layers.4" : [5, 10, 11, 13],
    "model.layers.6" : [],
    "model.layers.8" : [],
    "model.layers.10" : [],
    "model.layers.12" : [],
    "model.layers.14" : [],
    "model.layers.16" : [8],
    "model.layers.18" : [],
    "model.layers.20" : [],
    "model.layers.22" : [15],
    "model.layers.24" : [3, 13],
}

# sports features, intended verbs
verbs_sports_features = {
    "model.layers.0" : [2],
    "model.layers.2" : [2, 5, 7, 9],
    "model.layers.4" : [3],
    "model.layers.6" : [1, 2, 4, 5, 7, 9],
    "model.layers.8" : [3, 4, 7],
    "model.layers.10" : [9],
    "model.layers.12" : [4],
    "model.layers.14" : [0, 6, 7, 9],
    "model.layers.16" : [4],
    "model.layers.18" : [1, 2],
    "model.layers.20" : [2, 4],
    "model.layers.22" : [2,3, 5, 6,8],
    "model.layers.24" : [3, 9],
}

# sports features, intended sentiment
sentiment_sports_features = {
    "model.layers.0" : [3, 4, 5, 9],
    "model.layers.2" : [],
    "model.layers.4" : [],
    "model.layers.6" : [2, 3, 5, 6],
    "model.layers.8" : [0, 6, 8, 9],
    "model.layers.10" : [],
    "model.layers.12" : [5],
    "model.layers.14" : [],
    "model.layers.16" : [1, 8],
    "model.layers.18" : [1, 3, 9],
    "model.layers.20" : [1, 3, 5],
    "model.layers.22" : [1, 3, 4],
    "model.layers.24" : [2, 3],
}

exported_features = {
    "pronouns_sports_features" : pronouns_sports_features,
    "sports_pronouns_features" : sports_pronouns_features,
    "sentiment_pronouns_features" : sentiment_pronouns_features,
    "sentiment_verbs_features" : sentiment_verbs_features,
    "verbs_sentiment_features" : verbs_sentiment_features,
    "verbs_pronouns_features" : verbs_pronouns_features,
    "verbs_sports_features" : verbs_sports_features,
    "sentiment_sports_features" : sentiment_sports_features,
}
