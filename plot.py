import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/jungwoo/workspace/capstone-design/outputs/pid_episode_001.csv")

fig, axes = plt.subplots(4, 1, figsize=(12, 16))

# 1. 속도 그래프
axes[0].plot(df["time_sec"], df["leader_speed_mps"], label="Leader")
axes[0].plot(df["time_sec"], df["follower_speed_mps"], label="Follower")
axes[0].set_title("Speed")
axes[0].set_ylabel("m/s")
axes[0].legend()
axes[0].grid(True)

# 2. 거리 그래프
axes[1].plot(df["time_sec"], df["distance_m"], label="Actual Distance")
axes[1].plot(df["time_sec"], df["desired_distance_m"], label="Desired Distance", linestyle="--")
axes[1].set_title("Distance")
axes[1].set_ylabel("m")
axes[1].legend()
axes[1].grid(True)

# 3. 에러 그래프
axes[2].plot(df["time_sec"], df["spacing_error_m"], label="Spacing Error", color="red")
axes[2].axhline(y=0, color="black", linestyle="--")
axes[2].set_title("Spacing Error")
axes[2].set_ylabel("m")
axes[2].legend()
axes[2].grid(True)

# 4. throttle/brake 그래프
axes[3].plot(df["time_sec"], df["throttle"], label="Throttle", color="green")
axes[3].plot(df["time_sec"], df["brake"], label="Brake", color="red")
axes[3].set_title("Throttle / Brake")
axes[3].set_ylabel("value")
axes[3].set_xlabel("time (sec)")
axes[3].legend()
axes[3].grid(True)

plt.tight_layout()
plt.savefig("/home/jungwoo/workspace/capstone-design/outputs/pid_graph.png")
plt.show()
print("저장 완료!")