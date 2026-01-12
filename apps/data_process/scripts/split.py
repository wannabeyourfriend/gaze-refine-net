import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

np.random.seed(42)

data_path = 'data/prepare/standard.csv'
df = pd.read_csv(data_path)

selected_columns = [
    'target_x', 'target_y', 
    'original_gaze_x', 'original_gaze_y', 
    'sim_rbf_gaze_x', 'sim_rbf_gaze_y'
]

missing_columns = [col for col in selected_columns if col not in df.columns]
if missing_columns:
    selected_columns = [col for col in selected_columns if col in df.columns]

df_selected = df[selected_columns].copy()

df_selected = df_selected.dropna()

train_df, temp_df = train_test_split(df_selected, test_size=0.4, random_state=42)

val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

output_dir = 'data/prepare/split_data'
os.makedirs(output_dir, exist_ok=True)

train_path = os.path.join(output_dir, 'train.csv')
val_path = os.path.join(output_dir, 'val.csv')
test_path = os.path.join(output_dir, 'test.csv')

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

point_size = 0.5
alpha = 0.3

datasets = [
    (train_df, 'Train Set', 'black'),
    (val_df, 'Validation Set', 'black'),
    (test_df, 'Test Set', 'black')
]

for idx, (data, title, color) in enumerate(datasets):
    ax = axes[idx]
    ax.scatter(data['target_x'], data['target_y'], 
               s=point_size, alpha=alpha, c=color)
    ax.set_title(f'{title}\n(n={len(data)})', fontsize=12, fontweight='bold')
    ax.set_xlabel('target_x', fontsize=10)
    ax.set_ylabel('target_y', fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    ax.invert_yaxis()
    
    if 'target_x' in data.columns and 'target_y' in data.columns:
        ax.set_xlim(df_selected['target_x'].min() - 10, 
                    df_selected['target_x'].max() + 10)
        ax.set_ylim(df_selected['target_y'].max() + 10,
                    df_selected['target_y'].min() - 10)

plt.tight_layout()
plot_path = os.path.join(output_dir, 'data_distribution.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')