# gender_features = {
#     "model.layers.0" : [5],
#     "model.layers.2" : [17],
#     "model.layers.4" : [12, 13, 17, 18], # very gender related
#     "model.layers.6" : [15, 16, 18],
#     "model.layers.8" : [7, 10, 12, 15],
#     "model.layers.10" : [],
#     "model.layers.12" : [0],
#     "model.layers.14" : [],
#     "model.layers.16" : [18],
#     "model.layers.18" : [],
#     "model.layers.20" : [],
#     "model.layers.22" : [],
#     "model.layers.24" : [],
# }

gender_features = {
    "model.layers.0" : [5, 7],
    "model.layers.2" : [1, 3],
    "model.layers.4" : [], # very gender related
    "model.layers.6" : [8],
    "model.layers.8" : [7, 8, 9],
    "model.layers.10" : [],
    "model.layers.12" : [5 ],
    "model.layers.14" : [7],
    "model.layers.16" : [1],
    "model.layers.18" : [],
    "model.layers.20" : [],
    "model.layers.22" : [],
    "model.layers.24" : [],
}

exported_features = {
    "gender_features" : gender_features,
}
