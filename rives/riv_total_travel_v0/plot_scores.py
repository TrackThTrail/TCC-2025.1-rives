import matplotlib.pyplot as plt
import csv

episodes = []
avg_scores = []
epsilons = []

with open("training_log.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        episodes.append(int(row["Episode"]))
        avg_scores.append(float(row["AverageScore"]))
        epsilons.append(float(row["Epsilon"]))

plt.figure(figsize=(10, 5))
plt.plot(episodes, avg_scores, marker='o', label="Average Score")
plt.title("Média de Score por Episódio (a cada 100 episódios)")
plt.xlabel("Episódio")
plt.ylabel("Média de Score")
plt.grid(True)
plt.legend()
plt.show()
