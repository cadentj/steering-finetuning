# A=(pronouns sentiment 1)
# B=(pronouns verbs 1)
# C=(sports pronouns 0)
# D=(verbs sports 0)
# E=(verbs sentiment 0)
# F=(sentiment sports 0)
# G=(sports sentiment 0)
# H=(sentiment verbs 0)
# I=(sports verbs 0)

TEMPLATE = {
    "model.layers.0.mlp": [],
    "model.layers.1.mlp": [],
    "model.layers.2.mlp": [], 
    "model.layers.3.mlp": [],
    "model.layers.4.mlp": [],
    "model.layers.5.mlp": [],
    "model.layers.6.mlp": [], 
    "model.layers.7.mlp": [],
    "model.layers.8.mlp": [],
    "model.layers.9.mlp": [],
    "model.layers.10.mlp": [], 
    "model.layers.11.mlp": [],
    "model.layers.12.mlp": [],
    "model.layers.13.mlp": [],
    "model.layers.14.mlp": [],
    "model.layers.15.mlp": [],
}

pronouns_sentiment_1 = {
    "model.layers.0.mlp": [53224, 84634],
    "model.layers.1.mlp": [4128, 31126, 77126, 126870],
    "model.layers.2.mlp": [9108, 30231, 31801, 90222], 
    "model.layers.3.mlp": [94118, 103145, 105336],
    "model.layers.4.mlp": [84346, 104047],
    "model.layers.5.mlp": [52356, 86572, 128351],
    "model.layers.6.mlp": [107274], 
    "model.layers.7.mlp": [89459],
    "model.layers.8.mlp": [],
    "model.layers.9.mlp": [89748],
    "model.layers.10.mlp": [], 
    "model.layers.11.mlp": [108896, 125473],
    "model.layers.12.mlp": [44413],
    "model.layers.13.mlp": [],
    "model.layers.14.mlp": [],
    "model.layers.15.mlp": [],
}


sports_pronouns_0 = {
    "model.layers.0.mlp": [99966],
    "model.layers.1.mlp": [17200, 50205, 56451, 82498, 104352, 128146],
    "model.layers.2.mlp": [30231, 31598, 31801, 51697, 95740, 110260], 
    "model.layers.3.mlp": [],
    "model.layers.4.mlp": [19769, 57755],
    "model.layers.5.mlp": [100131],
    "model.layers.6.mlp": [], 
    "model.layers.7.mlp": [116082],
    "model.layers.8.mlp": [73503],
    "model.layers.9.mlp": [71695, 86777],
    "model.layers.10.mlp": [81076, 125064], 
    "model.layers.11.mlp": [76856, 115330],
    "model.layers.12.mlp": [44413, 57687],
    "model.layers.13.mlp": [2723],
    "model.layers.14.mlp": [],
    "model.layers.15.mlp": [],
}

verbs_sports_0 = {
    "model.layers.0.mlp": [29545, 101083, 124408],
    "model.layers.1.mlp": [28316, 112211, 123596, 126455],
    "model.layers.2.mlp": [2570, 42933, 44446, 71784, 103526], 
    "model.layers.3.mlp": [14266, 27279, 107332, 118682, 120317, 129125],
    "model.layers.4.mlp": [37496, 49983, 63640, 81713, 98551, 130048],
    "model.layers.5.mlp": [47865, 95432],
    "model.layers.6.mlp": [], 
    "model.layers.7.mlp": [125458],
    "model.layers.8.mlp": [],
    "model.layers.9.mlp": [67934],
    "model.layers.10.mlp": [], 
    "model.layers.11.mlp": [],
    "model.layers.12.mlp": [],
    "model.layers.13.mlp": [],
    "model.layers.14.mlp": [],
    "model.layers.15.mlp": [],
}

sentiment_sports_0 = {
    "model.layers.0.mlp": [28647, 29545, 101083, 130622],
    "model.layers.1.mlp": [72539, 79924, 112211],
    "model.layers.2.mlp": [42933, 71784, 103526], 
    "model.layers.3.mlp": [27279, 44528, 107332, 118682, 129125],
    "model.layers.4.mlp": [37496, 49983, 63640, 98551, 130048],
    "model.layers.5.mlp": [95432],
    "model.layers.6.mlp": [], 
    "model.layers.7.mlp": [125458],
    "model.layers.8.mlp": [],
    "model.layers.9.mlp": [67934],
    "model.layers.10.mlp": [], 
    "model.layers.11.mlp": [],
    "model.layers.12.mlp": [],
    "model.layers.13.mlp": [77368],
    "model.layers.14.mlp": [76538], # had to look this up, its baseball stats lol
    "model.layers.15.mlp": [],
}


pronouns_verbs_1 = {
    "model.layers.0.mlp": [22753, 36913, 49174, 84634, 94484, 96499],
    "model.layers.1.mlp": [31126, 31653, 34082, 34856, 75821, 77126, 126870],
    "model.layers.2.mlp": [9108, 30231], 
    "model.layers.3.mlp": [24179, 105336],
    "model.layers.4.mlp": [104047],
    "model.layers.5.mlp": [52356, 128351],
    "model.layers.6.mlp": [], 
    "model.layers.7.mlp": [70930, 89459],
    "model.layers.8.mlp": [],
    "model.layers.9.mlp": [],
    "model.layers.10.mlp": [], 
    "model.layers.11.mlp": [],
    "model.layers.12.mlp": [44413],
    "model.layers.13.mlp": [],
    "model.layers.14.mlp": [],
    "model.layers.15.mlp": [],
}

verbs_sentiment_0 = {
    "model.layers.0.mlp": [5478, 11402, 12751, 23692, 56193, 115968, 126902],
    "model.layers.1.mlp": [42384, 56604, 103176],
    "model.layers.2.mlp": [17888, 43948, 66754], 
    "model.layers.3.mlp": [59887, 86030, 106716, 114565],
    "model.layers.4.mlp": [81801, 114586],
    "model.layers.5.mlp": [4148, 35649],
    "model.layers.6.mlp": [26029, 98229], 
    "model.layers.7.mlp": [],
    "model.layers.8.mlp": [],
    "model.layers.9.mlp": [],
    "model.layers.10.mlp": [], 
    "model.layers.11.mlp": [],
    "model.layers.12.mlp": [60890],
    "model.layers.13.mlp": [],
    "model.layers.14.mlp": [],
    "model.layers.15.mlp": [],
}


sports_sentiment_0 = {
    "model.layers.0.mlp": [34209, 115968],
    "model.layers.1.mlp": [42384, 103176],
    "model.layers.2.mlp": [], 
    "model.layers.3.mlp": [59887, 114565],
    "model.layers.4.mlp": [64904, 35649],
    "model.layers.5.mlp": [],
    "model.layers.6.mlp": [], 
    "model.layers.7.mlp": [99317],
    "model.layers.8.mlp": [105311],
    "model.layers.9.mlp": [],
    "model.layers.10.mlp": [], 
    "model.layers.11.mlp": [],
    "model.layers.12.mlp": [],
    "model.layers.13.mlp": [],
    "model.layers.14.mlp": [],
    "model.layers.15.mlp": [],
}

sentiment_verbs_0 = {
    "model.layers.0.mlp": [2256, 3048, 54727, 98078, 122156],
    "model.layers.1.mlp": [226, 120164],
    "model.layers.2.mlp": [59788, 125991], 
    "model.layers.3.mlp": [8156, 64696],
    "model.layers.4.mlp": [68137, 84346],
    "model.layers.5.mlp": [],
    "model.layers.6.mlp": [], 
    "model.layers.7.mlp": [],
    "model.layers.8.mlp": [],
    "model.layers.9.mlp": [],
    "model.layers.10.mlp": [], 
    "model.layers.11.mlp": [],
    "model.layers.12.mlp": [],
    "model.layers.13.mlp": [],
    "model.layers.14.mlp": [33993],
    "model.layers.15.mlp": [],
}


exported_features = {
    "pronouns_sentiment_1": pronouns_sentiment_1,
    "sports_pronouns_0": sports_pronouns_0,
    "verbs_sports_0": verbs_sports_0,
    "sentiment_sports_0": sentiment_sports_0,
    "pronouns_verbs_1": pronouns_verbs_1,
    "verbs_sentiment_0": verbs_sentiment_0,
    "sports_sentiment_0": sports_sentiment_0,
    "sentiment_verbs_0": sentiment_verbs_0,
}