# data/create_synthetic.py
import os, csv, random
os.makedirs("data", exist_ok=True)

pos = ["Çok beğendim!", "Harika bir deneyimdi", "Tavsiye ederim"]
neu = ["Fena değil", "Orta kalite", "Beklentimi karşıladı"]
neg = ["Hiç beğenmedim", "Berbat", "Tekrar almam"]

def write(path, n):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w=csv.writer(f)
        w.writerow(["text","label"])
        for _ in range(n):
            r=random.random()
            if r<0.34: w.writerow([random.choice(pos), 2])
            elif r<0.67: w.writerow([random.choice(neu), 1])
            else: w.writerow([random.choice(neg), 0])

write("data/train.csv", 1000)
write("data/val.csv", 200)
print("Created data/train.csv and data/val.csv")
