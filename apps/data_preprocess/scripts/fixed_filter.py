import pandas as pd

df = pd.read_csv('data/systematic_recalibration/final_all_subjects_clean.csv')

target_counts = df.groupby(['target_x', 'target_y']).size().reset_index(name='count')
fixed_targets = target_counts[target_counts['count'] >= 2][['target_x', 'target_y']]

df_fixed = df.merge(fixed_targets, on=['target_x', 'target_y'], how='inner')
df_non_fixed = df.merge(fixed_targets, on=['target_x', 'target_y'], how='left', indicator=True)
df_non_fixed = df_non_fixed[df_non_fixed['_merge'] == 'left_only'].drop('_merge', axis=1)

df_fixed.to_csv('data/systematic_recalibration/fixed_points.csv', index=False)
df_non_fixed.to_csv('data/systematic_recalibration/non_fixed_points.csv', index=False)

print(f"Total rows: {len(df)}")
print(f"Fixed points rows: {len(df_fixed)}")
print(f"Non-fixed points rows: {len(df_non_fixed)}")
print(f"Unique fixed targets: {len(fixed_targets)}")