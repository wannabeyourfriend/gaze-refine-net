import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def calculate_euclidean_distance(df):
    df['distance'] = np.sqrt(
        (df['original_gaze_x'] - df['target_x'])**2 + 
        (df['original_gaze_y'] - df['target_y'])**2
    )
    return df

def detect_outliers(df, method='iqr', threshold=3):
    df = df.copy()
    
    if method == 'iqr':
        Q1 = df['distance'].quantile(0.25)
        Q3 = df['distance'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df['is_outlier'] = (df['distance'] < lower_bound) | (df['distance'] > upper_bound)
        
    elif method == 'zscore':
        mean = df['distance'].mean()
        std = df['distance'].std()
        df['z_score'] = (df['distance'] - mean) / std
        df['is_outlier'] = np.abs(df['z_score']) > threshold
        
    elif method == 'std':
        mean = df['distance'].mean()
        std = df['distance'].std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        df['is_outlier'] = (df['distance'] < lower_bound) | (df['distance'] > upper_bound)
    
    return df

def analyze_data_quality(df):
    stats = {
        'total_samples': len(df),
        'distance_mean': df['distance'].mean(),
        'distance_std': df['distance'].std(),
        'distance_median': df['distance'].median(),
        'distance_min': df['distance'].min(),
        'distance_max': df['distance'].max(),
        'distance_q25': df['distance'].quantile(0.25),
        'distance_q75': df['distance'].quantile(0.75),
        'outliers': df['is_outlier'].sum(),
        'outlier_percentage': (df['is_outlier'].sum() / len(df) * 100),
        'n_subjects': df['subject_name'].nunique()
    }
    return stats

def plot_distance_distribution(df, output_path='distance_distribution.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].hist(df['distance'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['distance'].mean(), color='r', linestyle='--', 
                       label=f'Mean: {df["distance"].mean():.2f}')
    axes[0, 0].axvline(df['distance'].median(), color='g', linestyle='--', 
                       label=f'Median: {df["distance"].median():.2f}')
    axes[0, 0].set_xlabel('Distance (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Original Gaze Distance to Target')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].boxplot(df['distance'], vert=True)
    axes[0, 1].set_ylabel('Distance (pixels)')
    axes[0, 1].set_title('Box Plot of Distance')
    axes[0, 1].grid(True, alpha=0.3)
    
    normal_data = df[~df['is_outlier']]['distance']
    outlier_data = df[df['is_outlier']]['distance']
    
    axes[1, 0].hist([normal_data, outlier_data], bins=30, 
                    label=['Normal', 'Outliers'], 
                    color=['blue', 'red'], alpha=0.6, edgecolor='black')
    axes[1, 0].set_xlabel('Distance (pixels)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Normal Data vs Outliers')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    from scipy import stats
    stats.probplot(df['distance'], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {output_path}")
    plt.close()

def plot_scatter_by_subject(df, output_path='scatter_by_subject.png'):
    subjects = df['subject_name'].unique()
    n_subjects = len(subjects)
    
    n_cols = min(3, n_subjects)
    n_rows = (n_subjects + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_subjects == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, subject in enumerate(subjects):
        subject_data = df[df['subject_name'] == subject]
        normal = subject_data[~subject_data['is_outlier']]
        outliers = subject_data[subject_data['is_outlier']]
        
        axes[idx].scatter(normal['target_x'], normal['target_y'], 
                         c='blue', alpha=0.5, s=50, label='Target')
        axes[idx].scatter(normal['original_gaze_x'], normal['original_gaze_y'], 
                         c='green', alpha=0.5, s=30, label='Normal Gaze')
        if len(outliers) > 0:
            axes[idx].scatter(outliers['original_gaze_x'], outliers['original_gaze_y'], 
                             c='red', alpha=0.7, s=30, label='Outlier Gaze')
        
        axes[idx].set_xlabel('X (pixels)')
        axes[idx].set_ylabel('Y (pixels)')
        axes[idx].set_title(f'{subject}\n({len(subject_data)} samples, {len(outliers)} outliers)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_aspect('equal', adjustable='box')
    
    for idx in range(n_subjects, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to: {output_path}")
    plt.close()

def compare_methods(df_original, output_path='method_comparison.csv'):
    methods = ['original', 'rbf', 'poly', 'sim_rbf', 'sim_pwa', 'sim_gpr']
    results = []
    
    for method in methods:
        if f'{method}_gaze_x' in df_original.columns and f'{method}_gaze_y' in df_original.columns:
            distance = np.sqrt(
                (df_original[f'{method}_gaze_x'] - df_original['target_x'])**2 + 
                (df_original[f'{method}_gaze_y'] - df_original['target_y'])**2
            )
            results.append({
                'method': method,
                'mean_error': distance.mean(),
                'std_error': distance.std(),
                'median_error': distance.median(),
                'max_error': distance.max(),
                'min_error': distance.min()
            })
    
    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(output_path, index=False)
    print(f"\nMethod comparison saved to: {output_path}")
    return comparison_df

def main():
    data_path = Path('data/systematic_recalibration/non_fixed_points.csv')
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    print(f"Reading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    print("\nCalculating distances...")
    df = calculate_euclidean_distance(df)
    
    print("\nDetecting outliers using different methods...")
    
    methods = {
        'IQR (1.5x)': ('iqr', 1.5),
        'IQR (3x)': ('iqr', 3),
        'Z-score (3)': ('zscore', 3),
        'Std (3x)': ('std', 3)
    }
    
    for name, (method, threshold) in methods.items():
        df_temp = detect_outliers(df.copy(), method=method, threshold=threshold)
        outlier_count = df_temp['is_outlier'].sum()
        outlier_pct = outlier_count / len(df) * 100
        print(f"  {name}: {outlier_count} outliers ({outlier_pct:.2f}%)")
    
    print("\nUsing IQR (1.5x) method for final outlier detection...")
    df = detect_outliers(df, method='iqr', threshold=1.5)
    
    print("\n" + "="*60)
    print("Data Quality Statistics")
    print("="*60)
    stats = analyze_data_quality(df)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:.2f}")
        else:
            print(f"{key:.<40} {value}")
    
    print("\n" + "="*60)
    print("Statistics by Subject")
    print("="*60)
    subject_stats = df.groupby('subject_name').agg({
        'distance': ['count', 'mean', 'std', 'median'],
        'is_outlier': 'sum'
    }).round(2)
    subject_stats.columns = ['n_samples', 'mean_dist', 'std_dist', 'median_dist', 'n_outliers']
    subject_stats['outlier_pct'] = (subject_stats['n_outliers'] / subject_stats['n_samples'] * 100).round(2)
    print(subject_stats)
    
    print("\nGenerating visualizations...")
    Path('output').mkdir(parents=True, exist_ok=True)
    plot_distance_distribution(df, 'output/non_fixed_points_distance_distribution.png')
    plot_scatter_by_subject(df, 'output/non_fixed_points_scatter_by_subject.png')

    print("\nComparing calibration methods...")
    comparison_df = compare_methods(df, 'output/non_fixed_points_method_comparison.csv')
    print("\nMethod Comparison:")
    print(comparison_df.to_string(index=False))
    
    df_clean = df[~df['is_outlier']].copy()
    output_clean_path = 'data/systematic_recalibration/non_fixed_points_final_all_subjects_clean.csv'
    
    Path(output_clean_path).parent.mkdir(parents=True, exist_ok=True)
    
    df_clean.to_csv(output_clean_path, index=False)
    print(f"\n✓ Clean data saved to: {output_clean_path}")
    print(f"  Original samples: {len(df)}")
    print(f"  Clean samples: {len(df_clean)}")
    print(f"  Removed samples: {len(df) - len(df_clean)} ({(len(df) - len(df_clean))/len(df)*100:.2f}%)")

    output_full_path = 'data/systematic_recalibration/non_fixed_points_final_all_subjects_with_outliers.csv'
    df.to_csv(output_full_path, index=False)
    print(f"\n✓ Full data with outlier flags saved to: {output_full_path}")

if __name__ == "__main__":
    main()