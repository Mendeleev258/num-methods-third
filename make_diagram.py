# make_diagram.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import ast

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
            
            all_data.append(df)
        else:
            print(f"Файл {path} не найден. Пропускаем.")
    
    if not all_data:
        raise FileNotFoundError("Ни один CSV-файл не найден в папке 'results'")
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Преобразуем system size к целым числам (если нужно)
    df_all['system size'] = df_all['system size'].astype(int)
    
    return df_all

def setup_xaxis_sizes(ax, x_data=None, xlabel='Размер системы (n)'):
    """Настраивает ось X с логарифмическими отметками для размеров систем"""
    ax.set_xlabel(xlabel)
    ax.set_xscale('log')
    
    # Автоматически определяем подходящие отметки
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
        nice_ticks = sorted(set(nice_ticks))  # Убираем дубликаты
        
        ax.set_xticks(nice_ticks)
        ax.set_xticklabels([str(x) for x in nice_ticks])

def create_size_vs_error_plots(df_all):
    """1. Влияние размера системы на точность метода Холецкого"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Влияние размера системы на точность метода Холецкого\nдля ленточных матриц', 
                 fontsize=14, fontweight='bold')
    
    # Группируем данные для анализа
    error_types = ['absolute error', 'relative error']
    condition_types = df_all['condition_type'].unique()
    
    for i, err_type in enumerate(error_types):
        for j, cond_type in enumerate(condition_types):
            ax = axes[i, j]
            
            # Фильтруем данные
            subset = df_all[df_all['condition_type'] == cond_type]
            
            # Усредняем по экспериментам одного размера
            grouped = subset.groupby('system size')[err_type].agg(['mean', 'std']).reset_index()
            
            if len(grouped) > 0:
                # Основной график
                ax.loglog(grouped['system size'], grouped['mean'], 'o-', 
                         linewidth=2, markersize=6, label='Средняя погрешность')
                
                # Область стандартного отклонения
                ax.fill_between(grouped['system size'],
                               grouped['mean'] - grouped['std'],
                               grouped['mean'] + grouped['std'],
                               alpha=0.2, label='±1 ст. откл.')
            
            # Настройка осей
            setup_xaxis_sizes(ax, grouped['system size'] if len(grouped) > 0 else None)
            ax.set_ylabel('Погрешность' if j == 0 else '')
            ax.set_yscale('log')
            ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
            ax.legend(loc='best')
            
            # Заголовок
            cond_title = 'Случайная' if cond_type == 'random' else 'Диаг. доминирующая'
            err_title = 'Абсолютная' if err_type == 'absolute error' else 'Относительная'
            ax.set_title(f'{err_title} погрешность\n({cond_title} матрица)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, '1_size_vs_error_cholesky.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 1: Влияние размера системы сохранен")

def create_condition_vs_error_plots(df_all):
    """2. Влияние обусловленности системы на точность"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Сравнение точности метода Холецкого\nдля разных типов матриц', 
                 fontsize=14, fontweight='bold')
    
    # Выбираем фиксированный размер системы (ближайший к 64)
    sizes = df_all['system size'].unique()
    if len(sizes) > 0:
        n_target = 64
        n_closest = min(sizes, key=lambda x: abs(x - n_target))
        
        # Берем данные с близким размером
        df_size = df_all[np.isclose(df_all['system size'], n_closest, rtol=0.2)]
        
        for ax_idx, err_type in enumerate(['absolute error', 'relative error']):
            ax = axes[ax_idx]
            
            # Группируем по типу обусловленности
            condition_data = []
            for cond_type in df_all['condition_type'].unique():
                subset = df_size[df_size['condition_type'] == cond_type]
                if len(subset) > 0:
                    mean_err = subset[err_type].mean()
                    std_err = subset[err_type].std()
                    condition_data.append({
                        'type': cond_type,
                        'mean': mean_err,
                        'std': std_err
                    })
            
            if condition_data:
                # Создаем столбчатую диаграмму
                types = [d['type'] for d in condition_data]
                means = [d['mean'] for d in condition_data]
                stds = [d['std'] for d in condition_data]
                
                x_pos = np.arange(len(types))
                bars = ax.bar(x_pos, means, yerr=stds, capsize=10, 
                             color=['skyblue', 'lightcoral'], alpha=0.8)
                
                # Подписи
                type_labels = ['Случайная', 'Диаг. доминирующая']
                ax.set_xticks(x_pos)
                ax.set_xticklabels(type_labels)
                
                # Добавляем значения над столбцами
                for i, (bar, mean_val) in enumerate(zip(bars, means)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1*height,
                           f'{mean_val:.2e}', ha='center', va='bottom')
            
            # Настройка осей
            err_title = 'Абсолютная' if err_type == 'absolute error' else 'Относительная'
            ax.set_ylabel(f'{err_title} погрешность')
            ax.set_yscale('log')
            ax.grid(True, axis='y', ls="--", linewidth=0.5, alpha=0.7)
            ax.set_title(f'{err_title} погрешность\n(n ≈ {n_closest})')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, '2_condition_vs_error.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 2: Влияние обусловленности сохранен")

def create_comparative_analysis(df_all):
    """3. Сравнительный анализ по разным диапазонам значений"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Сравнительный анализ точности метода Холецкого\nпо диапазонам значений матрицы', 
                 fontsize=14, fontweight='bold')
    
    # Группируем по диапазонам
    df_all['range_group'] = pd.cut(df_all['range_width'], 
                                   bins=[0, 1, 10, 100, 1000],
                                   labels=['[0-1]', '[1-10]', '[10-100]', '[100-1000]'])
    
    for i, cond_type in enumerate(df_all['condition_type'].unique()):
        for j, err_type in enumerate(['absolute error', 'relative error']):
            ax = axes[i, j]
            
            subset = df_all[df_all['condition_type'] == cond_type]
            
            if len(subset) > 0:
                # Группируем по диапазонам
                grouped = subset.groupby('range_group')[err_type].agg(['mean', 'std']).reset_index()
                
                # Столбчатая диаграмма
                x_pos = np.arange(len(grouped))
                bars = ax.bar(x_pos, grouped['mean'], yerr=grouped['std'], 
                             capsize=5, color='skyblue', alpha=0.8)
                
                # Настройки
                ax.set_xticks(x_pos)
                ax.set_xticklabels(grouped['range_group'])
                ax.set_ylabel('Погрешность')
                ax.set_yscale('log')
                ax.grid(True, axis='y', ls="--", linewidth=0.5, alpha=0.7)
                
                # Заголовок
                cond_title = 'Случайная' if cond_type == 'random' else 'Диаг. доминирующая'
                err_title = 'Абсолютная' if err_type == 'absolute error' else 'Относительная'
                ax.set_title(f'{cond_title} матрица\n{err_title} погрешность')
                
                # Добавить значения
                for bar, mean_val in zip(bars, grouped['mean']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                           f'{mean_val:.1e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, '3_comparative_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 3: Сравнительный анализ сохранен")

# 1. Загрузка данных
print("\n1. Загрузка данных из CSV файлов...")
df_all = load_and_prepare_data()
print(f"   Загружено {len(df_all)} записей")

# 2. Создание графиков
print("\n2. Создание графиков анализа...")

# 2.1 Влияние размера системы
print("   • График 1: Влияние размера системы на точность...")
create_size_vs_error_plots(df_all)

# 2.2 Влияние обусловленности
print("   • График 2: Влияние обусловленности системы...")
create_condition_vs_error_plots(df_all)

# 2.3 Сравнительный анализ по диапазонам
print("   • График 3: Сравнительный анализ по диапазонам значений...")
create_comparative_analysis(df_all)