import pandas as pd

df = pd.read_csv('data/systematic_recalibration/non_fixed_points.csv')

df_selected = df[df['subject_name'].isin(['Anna', 'Liu Jiaqi'])]

df_selected.to_csv('data/systematic_recalibration/selected_subjects.csv', index=False)

print(f"Total rows: {len(df)}")
print(f"Selected rows: {len(df_selected)}")
print(f"Anna rows: {len(df_selected[df_selected['subject_name'] == 'Anna'])}")
print(f"Liu Jiaqi rows: {len(df_selected[df_selected['subject_name'] == 'Liu Jiaqi'])}")