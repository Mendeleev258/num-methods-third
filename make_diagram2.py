import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Создаем папку для графиков
if not os.path.exists('plots2'):
    os.makedirs('plots2')

# Стиль графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Загрузка данных
df_rand = pd.read_csv('data/results_rand.csv')
df_dom = pd.read_csv('data/results_dominant.csv')

# Добавим тип матрицы
df_rand['matrix_type'] = 'random'
df_dom['matrix_type'] = 'diagonally_dominant'

# Объединим данные
df = pd.concat([df_rand, df_dom], ignore_index=True)

# Преобразуем filling range в числовой столбец для удобства
def extract_max_range(range_str):
    return float(range_str.strip('[]').split(', ')[-1])

df['max_range'] = df['filling range'].apply(extract_max_range)

# 1. График: Ошибка vs Размер системы (для разных типов матриц)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Влияние размерности системы на ошибку', fontsize=16, fontweight='bold')

# Фильтруем для сравнения
for idx, max_range in enumerate([10, 1000]):
    for jdx, metric in enumerate(['absolute error', 'relative error']):
        ax = axes[idx, jdx]
        
        # Группируем по размеру системы и типу матрицы
        for matrix_type in ['random', 'diagonally_dominant']:
            subset = df[(df['matrix_type'] == matrix_type) & 
                       (df['max_range'] == max_range)]
            
            if not subset.empty:
                grouped = subset.groupby('system size')[metric].mean()
                ax.plot(grouped.index, grouped.values, 
                       marker='o', linewidth=2, markersize=6,
                       label=f'{matrix_type}')
        
        ax.set_xlabel('System Size', fontsize=12)
        ax.set_ylabel(f'{metric} (log scale)', fontsize=12)
        ax.set_title(f'Max Range = {max_range}', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.tick_params(axis='both', labelsize=10)

plt.tight_layout()
plt.savefig('plots2/1_system_size_vs_error.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. График: Ошибка vs Ширина ленты (для фиксированного размера системы)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Влияние ширины ленты на ошибку (System Size = 512)', fontsize=16, fontweight='bold')

system_size = 512
max_range = 1000

for idx, metric in enumerate(['absolute error', 'relative error']):
    ax = axes[idx]
    
    for matrix_type in ['random', 'diagonally_dominant']:
        subset = df[(df['matrix_type'] == matrix_type) & 
                   (df['system size'] == system_size) &
                   (df['max_range'] == max_range)]
        
        if not subset.empty:
            # Сортируем по ширине ленты
            subset = subset.sort_values('bandwidth')
            ax.plot(subset['bandwidth'], subset[metric], 
                   marker='s', linewidth=2, markersize=8,
                   label=f'{matrix_type}')
    
    ax.set_xlabel('Bandwidth', fontsize=12)
    ax.set_ylabel(f'{metric} (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.tick_params(axis='both', labelsize=10)

plt.tight_layout()
plt.savefig('plots2/2_bandwidth_vs_error.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. График: Ошибка vs Диапазон значений (для фиксированного размера системы)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Влияние диапазона значений на ошибку (System Size = 256, Bandwidth = 101)', fontsize=16, fontweight='bold')

system_size = 256
bandwidth = 101

for idx, metric in enumerate(['absolute error', 'relative error']):
    ax = axes[idx]
    
    for matrix_type in ['random', 'diagonally_dominant']:
        subset = df[(df['matrix_type'] == matrix_type) & 
                   (df['system size'] == system_size) &
                   (df['bandwidth'] == bandwidth)]
        
        if not subset.empty:
            # Группируем по диапазону значений
            grouped = subset.groupby('max_range')[metric].mean()
            ax.plot(grouped.index, grouped.values, 
                   marker='D', linewidth=2, markersize=8,
                   label=f'{matrix_type}')
    
    ax.set_xlabel('Max Range Value', fontsize=12)
    ax.set_ylabel(f'{metric} (log scale)', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.tick_params(axis='both', labelsize=10)

plt.tight_layout()
plt.savefig('plots2/3_range_vs_error.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. График: Сравнение ошибок по типам матриц (боксплот)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Распределение ошибок по типам матриц', fontsize=16, fontweight='bold')

for idx, metric in enumerate(['absolute error', 'relative error']):
    ax = axes[idx]
    
    # Подготовка данных для боксплота
    data_random = df[df['matrix_type'] == 'random'][metric]
    data_dominant = df[df['matrix_type'] == 'diagonally_dominant'][metric]
    
    # Боксплот
    bp = ax.boxplot([data_random, data_dominant], 
                    labels=['Random', 'Diagonally Dominant'],
                    patch_artist=True)
    
    # Цвета
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel(f'{metric} (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', labelsize=10)

plt.tight_layout()
plt.savefig('plots2/4_error_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. График: 3D визуализация (для случайных матриц)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Берем только случайные матрицы для 3D
df_rand_3d = df[df['matrix_type'] == 'random'].copy()
df_rand_3d['log_abs_error'] = np.log10(df_rand_3d['absolute error'] + 1e-20)

scatter = ax.scatter(df_rand_3d['system size'], 
                    df_rand_3d['bandwidth'], 
                    df_rand_3d['log_abs_error'],
                    c=df_rand_3d['max_range'],
                    cmap='viridis',
                    s=50,
                    alpha=0.7)

ax.set_xlabel('System Size', fontsize=12)
ax.set_ylabel('Bandwidth', fontsize=12)
ax.set_zlabel('log10(Absolute Error)', fontsize=12)
ax.set_title('3D: Ошибка случайных матриц\n(цвет = диапазон значений)', fontsize=14)

# Цветовая шкала
cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Max Range', fontsize=12)

plt.tight_layout()
plt.savefig('plots2/5_3d_random_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Все графики сохранены в папке 'plots':")
for i in range(1, 6):
    print(f"  plots/{i}_*.png")