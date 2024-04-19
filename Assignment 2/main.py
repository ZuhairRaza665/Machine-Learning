# Date: 2024-03-12
# CSC354 – Assignment2 – ML – Concept Learning
# Zuhair Raza
# FA20-BCS-085
# Brief Description: The Candidate-Elimination algorithm is an iterative process for learning a hypothesis space from training data. It begins by initializing both the most general and most specific hypotheses. During training, for each example in the training set, it refines these hypotheses based on whether the example is classified as positive or negative. If positive, the algorithm generalizes the specific hypothesis and eliminates inconsistent general hypotheses. Conversely, if negative, it specializes the general hypothesis and eliminates inconsistent specific hypotheses. After each iteration, the algorithm stores the current specific hypothesis. This process continues until all examples are processed. Finally, the algorithm outputs an ordered list of consistent hypotheses, representing possible solutions consistent with the observed data. Through this iterative refinement process, the Candidate-Elimination algorithm effectively learns from training data to generalize and specialize hypotheses until convergence.


import numpy as np

data = [
    ["big", "red", "circle", "no"],
    ["small", "red", "triangle", "no"],
    ["small", "red", "circle", "yes"],
    ["big", "blue", "circle", "no"],
    ["small", "blue", "circle", "yes"]
]

specific_h = None
for instance in data:
    if instance[-1] == "yes":
        specific_h = instance[:-1].copy()
        break

if specific_h is None:
    print("No positive examples found in the dataset.")
    exit()

print("\nInitialization of specific_h and general_h")
print("Specific Boundary:", specific_h)

general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]
print("Generic Boundary:", general_h)

for i, instance in enumerate(data):
    if instance[-1] == "yes":
        print("\nInstance", i + 1, "is Positive")
        for x in range(len(specific_h)):
            if instance[x] != specific_h[x]:
                specific_h[x] = '?'
    elif instance[-1] == "no":
        print("\nInstance", i + 1, "is Negative")
        for x in range(len(specific_h)):
            if instance[x] != specific_h[x]:
                general_h[x][x] = specific_h[x]

    print("Specific Boundary after", i + 1, "Instance is", specific_h)
    print("Generic Boundary after", i + 1, "Instance is", general_h)
    print("\n")

general_h = [hypothesis for hypothesis in general_h if hypothesis != ['?'] * len(specific_h)]

print("Final Specific_h:\n", specific_h)
print("Final General_h:\n", general_h)

