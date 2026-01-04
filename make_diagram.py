# make_diagram.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
from matplotlib.ticker import ScalarFormatter

# Убедитесь, что папки существуют
RESULTS_DIR = 'data'
OUTPUT_DIR = 'plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_prepare_data():
    """Загрузка и подготовка данных из CSV файлов"""
    files_info = {
        'results_rand.csv': 'random',
        'results_dominant.csv': 'dominant',
    }
    
    all_data = []
    for filename, cond_type in files_info.items():
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            print(f"Загрузка файла: {filename}")
            df = pd.read_csv(path)
            
            # Извлекаем low/high из строки вида "[1.0, 10.0]"
            df['filling_range'] = df['filling range'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            df['low'] = df['filling_range'].apply(lambda x: x[0])
            df['high'] = df['filling_range'].apply(lambda x: x[1])
            df['range_width'] = df['high'] - df['low']
            
            # Добавляем тип обусловленности
            df['condition_type'] = cond_type
            
            # Вычисляем масштаб (для анализа)
            df['scale'] = df[['low', 'high']].abs().max(axis=1)
            
            # Вычисляем отношение ширины ленты к размеру системы
            df['bandwidth_ratio'] = df['bandwidth'] / df['system size']
            
            all_data.append(df)
        else:
            print(f"Файл {path} не найден. Пропускаем.")
    
    if not all_data:
        raise FileNotFoundError("Ни один CSV-файл не найден в папке 'data'")
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Преобразуем system size к целым числам
    df_all['system size'] = df_all['system size'].astype(int)
    df_all['bandwidth'] = df_all['bandwidth'].astype(int)
    
    return df_all

def setup_xaxis_sizes(ax, x_data=None, xlabel='Размер системы (n)'):
    """Настраивает ось X с логарифмическими отметками для размеров систем"""
    ax.set_xlabel(xlabel)
    ax.set_xscale('log')
    
    if x_data is not None and len(x_data) > 0:
        x_min, x_max = min(x_data), max(x_data)
        
        # Создаем логарифмически распределенные отметки
        log_min = np.log10(x_min)
        log_max = np.log10(x_max)
        log_ticks = np.logspace(log_min, log_max, num=6)
        
        # Округляем до красивых значений
        def round_to_nice(x):
            if x < 10:
                return int(round(x))
            elif x < 100:
                return int(round(x/10)*10)
            elif x < 1000:
                return int(round(x/100)*100)
            else:
                return int(round(x/1000)*1000)
        
        nice_ticks = [round_to_nice(x) for x in log_ticks]
        nice_ticks = sorted(set(nice_ticks))
        
        ax.set_xticks(nice_ticks)
        ax.set_xticklabels([str(x) for x in nice_ticks])

def create_size_vs_error_plots(df_all):
    """1. Влияние размера системы на точность метода Холецкого"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Влияние размера системы на точность метода Холецкого\nдля ленточных матриц', 
                 fontsize=14, fontweight='bold')
    
    error_types = ['absolute error', 'relative error']
    condition_types = df_all['condition_type'].unique()
    
    for i, err_type in enumerate(error_types):
        for j, cond_type in enumerate(condition_types):
            ax = axes[i, j]
            
            subset = df_all[df_all['condition_type'] == cond_type]
            
            # Группируем по размеру системы и усредняем
            grouped = subset.groupby('system size')[err_type].agg(['mean', 'std']).reset_index()
            
            if len(grouped) > 0:
                ax.loglog(grouped['system size'], grouped['mean'], 'o-', 
                         linewidth=2, markersize=6, label='Средняя погрешность')
                
                ax.fill_between(grouped['system size'],
                               grouped['mean'] - grouped['std'],
                               grouped['mean'] + grouped['std'],
                               alpha=0.2, label='±1 ст. откл.')
            
            setup_xaxis_sizes(ax, grouped['system size'] if len(grouped) > 0 else None)
            ax.set_ylabel('Погрешность' if j == 0 else '')
            ax.set_yscale('log')
            ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
            ax.legend(loc='best')
            
            cond_title = 'Случайная' if cond_type == 'random' else 'Диаг. доминирующая'
            err_title = 'Абсолютная' if err_type == 'absolute error' else 'Относительная'
            ax.set_title(f'{err_title} погрешность\n({cond_title} матрица)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, '1_size_vs_error_cholesky.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 1: Влияние размера системы сохранен")

def create_bandwidth_vs_error_plots(df_all):
    """2. Влияние ширины ленты на точность"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Влияние ширины ленты на точность метода Холецкого', 
                 fontsize=14, fontweight='bold')
    
    error_types = ['absolute error', 'relative error']
    condition_types = df_all['condition_type'].unique()
    
    # Выбираем несколько характерных размеров систем
    sizes_to_plot = sorted(df_all['system size'].unique())
    if len(sizes_to_plot) > 4:
        sizes_to_plot = sizes_to_plot[::len(sizes_to_plot)//4]  # 4 равномерно распределенных размера
    
    for i, err_type in enumerate(error_types):
        for j, cond_type in enumerate(condition_types):
            ax = axes[i, j]
            subset = df_all[df_all['condition_type'] == cond_type]
            
            # Для каждого размера системы строим зависимость от ширины ленты
            for size in sizes_to_plot[:2]:  # Берем 2 размера для наглядности
                size_subset = subset[subset['system size'] == size]
                if len(size_subset) > 0:
                    # Группируем по ширине ленты
                    grouped = size_subset.groupby('bandwidth')[err_type].mean().reset_index()
                    ax.loglog(grouped['bandwidth'], grouped[err_type], 'o-', 
                             linewidth=2, markersize=6, label=f'n={size}')
            
            ax.set_xlabel('Ширина ленты (bandwidth)')
            ax.set_ylabel('Погрешность')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
            ax.legend(loc='best')
            
            cond_title = 'Случайная' if cond_type == 'random' else 'Диаг. доминирующая'
            err_title = 'Абсолютная' if err_type == 'absolute error' else 'Относительная'
            ax.set_title(f'{err_title} погрешность\n({cond_title} матрица)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, '2_bandwidth_vs_error.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 2: Влияние ширины ленты сохранен")

def create_bandwidth_ratio_vs_error_plots(df_all):
    """3. Влияние отношения ширины ленты к размеру системы"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Влияние отношения ширины ленты к размеру системы\nна точность метода Холецкого', 
                 fontsize=14, fontweight='bold')
    
    # Определяем конкретные отношения ширины ленты
    target_ratios = [0.1, 0.4, 0.8]  # 1/10, 4/10, 8/10
    ratio_labels = ['1/10', '4/10', '8/10']
    
    error_types = ['absolute error', 'relative error']
    condition_types = df_all['condition_type'].unique()
    
    for i, err_type in enumerate(error_types):
        for j, cond_type in enumerate(condition_types):
            ax = axes[i, j]
            subset = df_all[df_all['condition_type'] == cond_type]
            
            if len(subset) > 0:
                # Собираем данные для каждого целевого отношения
                means = []
                stds = []
                x_labels = []
                
                for ratio in target_ratios:
                    # Фильтруем данные с близким отношением ширины ленты
                    ratio_subset = subset[np.isclose(subset['bandwidth_ratio'], ratio, atol=0.05)]
                    
                    if len(ratio_subset) > 0:
                        means.append(ratio_subset[err_type].mean())
                        stds.append(ratio_subset[err_type].std())
                        x_labels.append(f'{ratio_labels[target_ratios.index(ratio)]}\n(≈{ratio*100:.0f}%)')
                    else:
                        # Если нет данных для точного соотношения, используем ближайшее
                        closest_idx = (subset['bandwidth_ratio'] - ratio).abs().idxmin()
                        means.append(subset.loc[closest_idx, err_type])
                        stds.append(0)  # Не можем оценить std для одного значения
                        x_labels.append(f'{ratio_labels[target_ratios.index(ratio)]}\n(≈{ratio*100:.0f}%)')
                
                if means:
                    x_pos = np.arange(len(means))
                    bars = ax.bar(x_pos, means, yerr=stds if any(stds) else None,
                                 capsize=10, color=['lightblue', 'skyblue', 'steelblue'],
                                 alpha=0.8, edgecolor='black')
                    
                    # Настройки осей
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(x_labels, fontsize=11)
                    ax.set_ylabel('Погрешность')
                    ax.set_yscale('log')
                    ax.grid(True, axis='y', ls="--", linewidth=0.5, alpha=0.7)
                    
                    # Добавляем значения над столбцами
                    for idx, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
                        height = bar.get_height()
                        # Форматируем значение в зависимости от порядка величины
                        if mean_val < 1e-10:
                            label = f'{mean_val:.1e}'
                        elif mean_val < 1e-5:
                            label = f'{mean_val:.2e}'
                        else:
                            label = f'{mean_val:.1e}'
                        
                        ax.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                               label, ha='center', va='bottom', fontsize=10, fontweight='bold')
                        
                        # Показываем количество экспериментов под столбцом
                        ratio_subset = subset[np.isclose(subset['bandwidth_ratio'], 
                                                        target_ratios[idx], atol=0.05)]
                        count = len(ratio_subset)
                        ax.text(bar.get_x() + bar.get_width()/2., -0.05 * height,
                               f'n={count}', ha='center', va='top', fontsize=9)
            
            cond_title = 'Случайная' if cond_type == 'random' else 'Диаг. доминирующая'
            err_title = 'Абсолютная' if err_type == 'absolute error' else 'Относительная'
            ax.set_title(f'{err_title} погрешность\n({cond_title} матрица)', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, '3_bandwidth_ratio_vs_error.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 3: Влияние отношения ширины ленты к размеру сохранен")

def create_comparative_analysis(df_all):
    """4. Сравнительный анализ по диапазонам значений"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Сравнительный анализ точности метода Холецкого\nпо диапазонам значений матрицы', 
                 fontsize=14, fontweight='bold')
    
    # Группируем по диапазонам
    df_all['range_group'] = pd.cut(df_all['range_width'], 
                                   bins=[0, 10, 100, 1000, 10000],
                                   labels=['[1-10]', '[10-100]', '[100-1000]', '[1000-10000]'])
    
    for i, cond_type in enumerate(df_all['condition_type'].unique()):
        for j, err_type in enumerate(['absolute error', 'relative error']):
            ax = axes[i, j]
            
            subset = df_all[df_all['condition_type'] == cond_type]
            
            if len(subset) > 0:
                grouped = subset.groupby('range_group')[err_type].agg(['mean', 'std']).reset_index()
                
                x_pos = np.arange(len(grouped))
                bars = ax.bar(x_pos, grouped['mean'], yerr=grouped['std'], 
                             capsize=5, color='skyblue', alpha=0.8)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(grouped['range_group'])
                ax.set_ylabel('Погрешность')
                ax.set_yscale('log')
                ax.grid(True, axis='y', ls="--", linewidth=0.5, alpha=0.7)
                
                cond_title = 'Случайная' if cond_type == 'random' else 'Диаг. доминирующая'
                err_title = 'Абсолютная' if err_type == 'absolute error' else 'Относительная'
                ax.set_title(f'{cond_title} матрица\n{err_title} погрешность')
                
                for bar, mean_val in zip(bars, grouped['mean']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                           f'{mean_val:.1e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, '4_comparative_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 4: Сравнительный анализ по диапазонам сохранен")

def create_heatmap_analysis(df_all):
    """5. Тепловая карта: размер системы vs ширина ленты"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Тепловая карта: влияние размера системы и ширины ленты\nна относительную погрешность', 
                 fontsize=14, fontweight='bold')
    
    for idx, cond_type in enumerate(df_all['condition_type'].unique()):
        ax = axes[idx]
        subset = df_all[df_all['condition_type'] == cond_type]
        
        if len(subset) > 0:
            # Создаем сетку: размер системы vs ширина ленты
            pivot_table = subset.pivot_table(
                values='relative error',
                index='system size',
                columns='bandwidth',
                aggfunc='mean'
            )
            
            # Заполняем пропущенные значения
            pivot_table = pivot_table.fillna(pivot_table.mean())
            
            # Строим тепловую карту
            im = ax.imshow(np.log10(pivot_table.values), cmap='viridis', aspect='auto')
            
            # Настройка осей
            ax.set_xlabel('Ширина ленты')
            ax.set_ylabel('Размер системы')
            ax.set_title(f'{cond_type} матрица')
            
            # Добавляем цветовую шкалу
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('log10(относительная погрешность)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, '5_heatmap_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 5: Тепловая карта сохранена")

def main():
    """Основная функция анализа"""
    print("="*60)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ МЕТОДА ХОЛЕЦКОГО")
    print("ДЛЯ ЛЕНТОЧНЫХ МАТРИЦ")
    print("="*60)
    
    try:
        # 1. Загрузка данных
        print("\n1. Загрузка данных из CSV файлов...")
        df_all = load_and_prepare_data()
        print(f"   Загружено {len(df_all)} записей")
        print(f"   Диапазон размеров: {df_all['system size'].min()} - {df_all['system size'].max()}")
        print(f"   Диапазон ширины ленты: {df_all['bandwidth'].min()} - {df_all['bandwidth'].max()}")
        
        # 2. Создание графиков
        print("\n2. Создание графиков анализа...")
        
        print("   • График 1: Влияние размера системы на точность...")
        create_size_vs_error_plots(df_all)
        
        print("   • График 2: Влияние ширины ленты на точность...")
        create_bandwidth_vs_error_plots(df_all)
        
        print("   • График 3: Влияние относительной ширины ленты...")
        create_bandwidth_ratio_vs_error_plots(df_all)
        
        print("   • График 4: Сравнительный анализ по диапазонам значений...")
        create_comparative_analysis(df_all)
        
        print("   • График 5: Тепловая карта (размер vs ширина ленты)...")
        create_heatmap_analysis(df_all)
        
        print("\n" + "="*60)
        print("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        print(f"Все графики сохранены в папке: '{OUTPUT_DIR}'")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nОШИБКА: {e}")
        print("Убедитесь, что CSV файлы находятся в папке 'data/'")
        print("Сначала запустите main.py для генерации данных")
    except Exception as e:
        print(f"\nНЕОЖИДАННАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()