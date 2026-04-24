import pandas as pd, matplotlib.pyplot as plt
for c in ["demos", "first_person", "sdf"]:
    df = pd.read_json(f"data/train_log_{c}.jsonl", lines=True)
    plt.plot(df.examples_seen, df.loss.rolling(10, min_periods=1).mean(), label=c)
plt.xlabel("examples seen"); plt.ylabel("train loss (per trained token)"); plt.legend()