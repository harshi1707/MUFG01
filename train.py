import os
import json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model_utils import (
    load_data,
    preprocess,
    train_baselines,
    run_grid_search,
    create_ensemble_models,
    evaluate_model,
    save_artifacts
)
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set up professional plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Healthcare color scheme
HEALTH_COLORS = {
    'primary': '#2E86AB',      # Medical blue
    'secondary': '#A23B72',    # Deep pink
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red for disease
    'neutral': '#6B7B8C',      # Gray
    'background': '#F7F9FC',   # Light blue-gray
    'models': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # For different models
}


def main(data_path, artifacts_dir='artifacts', test_size=0.2, random_state=42):
    print('Loading data...')
    df = load_data(data_path)

    print('Preprocessing with SMOTE and feature engineering...')
    X, y, scaler = preprocess(df, target_col='heart_disease', use_smote=True, feature_engineering=True)

    print('Splitting into train/test...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print('Training baseline models...')
    baselines = train_baselines(X_train, y_train)

    # Evaluate baseline models on test set
    print('Evaluating baseline models on test set...')
    baseline_evaluations = {}
    for name, model in baselines.items():
        eval_res = evaluate_model(model, X_test, y_test)
        baseline_evaluations[name] = eval_res
        print(f"\n--- {name.upper()} ---")
        print("Classification Report:")
        print(classification_report(y_test, model.predict(X_test)))
        print(f"ROC-AUC: {eval_res['roc_auc']:.4f}")

    # Print summary table
    print('\nBaseline Models Summary Table:')
    print('Model\t\t\tAccuracy\tPrecision\tRecall\t\tF1-Score\tROC-AUC')
    print('-' * 80)
    for name, eval_res in baseline_evaluations.items():
        report = eval_res['classification_report']
        acc = report['accuracy']
        prec = report['macro avg']['precision']
        rec = report['macro avg']['recall']
        f1 = report['macro avg']['f1-score']
        roc = eval_res['roc_auc']
        print(f'{name.capitalize()}\t\t{acc:.4f}\t\t{prec:.4f}\t\t{rec:.4f}\t\t{f1:.4f}\t\t{roc:.4f}')

    print('Running grid search for hyperparameter optimization...')
    best_models = run_grid_search(X_train, y_train)

    # Create ensemble models
    ensemble_models = create_ensemble_models(best_models, X_train, y_train)

    # Evaluate optimized models on test set
    evaluations = {}
    for name, info in best_models.items():
        estimator = info['best_estimator']
        eval_res = evaluate_model(estimator, X_test, y_test)
        evaluations[name] = {
            'best_params': info['best_params'],
            'cv_score': info['best_score'],
            'test_eval': eval_res
        }

    # Evaluate ensemble models
    for name, model in ensemble_models.items():
        eval_res = evaluate_model(model, X_test, y_test)
        evaluations[name] = {
            'best_params': {'ensemble_type': name},
            'cv_score': None,  # Ensembles don't have CV scores in the same way
            'test_eval': eval_res
        }

    # Model comparison section
    print('\nModel Comparison: Baseline vs Optimized')
    comparison_data = []
    for name in baselines.keys():
        baseline_eval = baseline_evaluations[name]
        optimized_eval = evaluations[name]['test_eval']

        baseline_report = baseline_eval['classification_report']
        optimized_report = optimized_eval['classification_report']

        baseline_acc = baseline_report['accuracy']
        baseline_prec = baseline_report['macro avg']['precision']
        baseline_rec = baseline_report['macro avg']['recall']
        baseline_f1 = baseline_report['macro avg']['f1-score']
        baseline_roc = baseline_eval['roc_auc']

        optimized_acc = optimized_report['accuracy']
        optimized_prec = optimized_report['macro avg']['precision']
        optimized_rec = optimized_report['macro avg']['recall']
        optimized_f1 = optimized_report['macro avg']['f1-score']
        optimized_roc = optimized_eval['roc_auc']

        comparison_data.append({
            'Model': name.capitalize(),
            'Version': 'Baseline',
            'Accuracy': baseline_acc,
            'Precision': baseline_prec,
            'Recall': baseline_rec,
            'F1-Score': baseline_f1,
            'ROC-AUC': baseline_roc
        })
        comparison_data.append({
            'Model': name.capitalize(),
            'Version': 'Optimized',
            'Accuracy': optimized_acc,
            'Precision': optimized_prec,
            'Recall': optimized_rec,
            'F1-Score': optimized_f1,
            'ROC-AUC': optimized_roc
        })

    # Print table
    print('Model\t\tVersion\t\tAccuracy\tPrecision\tRecall\t\tF1-Score\tROC-AUC')
    print('-' * 100)
    for row in comparison_data:
        print(f"{row['Model']}\t\t{row['Version']}\t\t{row['Accuracy']:.4f}\t\t{row['Precision']:.4f}\t\t{row['Recall']:.4f}\t\t{row['F1-Score']:.4f}\t\t{row['ROC-AUC']:.4f}")

    # Enhanced Model Performance Dashboard
    print('\nCreating comprehensive model performance visualizations...')

    # 1. ROC Curves for all optimized models
    plt.figure(figsize=(12, 8))
    for i, (name, info) in enumerate(best_models.items()):
        estimator = info['best_estimator']
        y_pred_proba = estimator.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=HEALTH_COLORS['models'][i % len(HEALTH_COLORS['models'])],
                linewidth=3, label=f'{name.capitalize()} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('ROC Curves - Optimized Models', fontsize=18, fontweight='bold', color=HEALTH_COLORS['primary'], pad=20)
    plt.legend(loc="lower right", fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/roc_curves_optimized.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Performance Metrics Comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    models = list(best_models.keys())

    baseline_scores = []
    optimized_scores = []

    for name in models:
        baseline_eval = baseline_evaluations[name]
        optimized_eval = evaluations[name]['test_eval']

        baseline_report = baseline_eval['classification_report']
        optimized_report = optimized_eval['classification_report']

        baseline_scores.append([
            baseline_report['accuracy'],
            baseline_report['macro avg']['precision'],
            baseline_report['macro avg']['recall'],
            baseline_report['macro avg']['f1-score'],
            baseline_eval['roc_auc']
        ])

        optimized_scores.append([
            optimized_report['accuracy'],
            optimized_report['macro avg']['precision'],
            optimized_report['macro avg']['recall'],
            optimized_report['macro avg']['f1-score'],
            optimized_eval['roc_auc']
        ])

    # Create comparison bar chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance: Baseline vs Optimized', fontsize=20, fontweight='bold', color=HEALTH_COLORS['primary'])

    axes = axes.ravel()

    for i, metric in enumerate(metrics):
        if i < len(axes):
            x = np.arange(len(models))
            width = 0.35

            baseline_vals = [score[i] for score in baseline_scores]
            optimized_vals = [score[i] for score in optimized_scores]

            bars1 = axes[i].bar(x - width/2, baseline_vals, width, label='Baseline',
                               color=HEALTH_COLORS['neutral'], alpha=0.7, edgecolor='black', linewidth=1)
            bars2 = axes[i].bar(x + width/2, optimized_vals, width, label='Optimized',
                               color=HEALTH_COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=1)

            axes[i].set_title(f'{metric}', fontweight='bold', fontsize=14)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels([m.capitalize() for m in models], rotation=45, ha='right')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

            for bar in bars2:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Remove empty subplot
    if len(metrics) < len(axes):
        axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig('plots/model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Enhanced Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Confusion Matrices - Optimized Models', fontsize=20, fontweight='bold', color=HEALTH_COLORS['primary'])

    axes = axes.ravel()

    for i, (name, info) in enumerate(best_models.items()):
        if i < len(axes):
            estimator = info['best_estimator']
            y_pred = estimator.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            # Create heatmap with better styling
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       cbar=True, square=True, linewidths=0.5,
                       annot_kws={"size": 14, "weight": "bold"})

            axes[i].set_title(f'{name.capitalize()} Model', fontsize=16, fontweight='bold', pad=20)
            axes[i].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('True Label', fontsize=12, fontweight='bold')

            # Customize tick labels
            axes[i].set_xticklabels(['No Disease', 'Heart Disease'], fontsize=11)
            axes[i].set_yticklabels(['No Disease', 'Heart Disease'], fontsize=11)

    plt.tight_layout()
    plt.savefig('plots/enhanced_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Model Improvement Summary
    improvement_data = []
    for name in models:
        baseline_roc = baseline_evaluations[name]['roc_auc']
        optimized_roc = evaluations[name]['test_eval']['roc_auc']
        improvement = optimized_roc - baseline_roc
        improvement_pct = (improvement / baseline_roc) * 100

        improvement_data.append({
            'model': name.capitalize(),
            'baseline': baseline_roc,
            'optimized': optimized_roc,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        })

    # Sort by improvement percentage
    improvement_data.sort(key=lambda x: x['improvement_pct'], reverse=True)

    plt.figure(figsize=(12, 8))
    models_list = [d['model'] for d in improvement_data]
    improvements = [d['improvement_pct'] for d in improvement_data]

    bars = plt.bar(range(len(models_list)), improvements,
                  color=[HEALTH_COLORS['success'] if x > 0 else HEALTH_COLORS['accent'] for x in improvements],
                  alpha=0.8, edgecolor='black', linewidth=1.5)

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xticks(range(len(models_list)), models_list, rotation=45, ha='right')
    plt.title('ROC-AUC Improvement: Baseline → Optimized', fontsize=18, fontweight='bold', color=HEALTH_COLORS['primary'], pad=20)
    plt.ylabel('Improvement (%)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add value labels
    for i, (bar, improvement) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -1.5),
                f'{height:+.2f}%', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=12, fontweight='bold', color='black')

    plt.tight_layout()
    plt.savefig('plots/model_improvement_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('Enhanced visualizations saved to plots/ directory:')
    print('  - roc_curves_optimized.png')
    print('  - model_performance_comparison.png')
    print('  - enhanced_confusion_matrices.png')
    print('  - model_improvement_summary.png')

# Choose best model by test ROC-AUC
    best_name = max(evaluations.keys(), key=lambda n: evaluations[n]['test_eval']['roc_auc'])
    chosen_estimator = best_models[best_name]['best_estimator']

    metadata = {
        'chosen_model': best_name,
        'evaluations': evaluations
    }
    save_artifacts(chosen_estimator, scaler, metadata, artifacts_dir=artifacts_dir)
    print(f"Saved best model: {best_name} to {artifacts_dir}")

    # Print summary of test ROC-AUC scores
    print('Summary of evaluations (ROC-AUC on test set):')
    print(json.dumps(
        {k: {'roc_auc': v['test_eval']['roc_auc']} for k, v in evaluations.items()},
        indent=2
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='heart_disease_dataset.csv', help='Path to CSV dataset')
    parser.add_argument('--artifacts', type=str, default='artifacts', help='Artifacts output dir')
    parser.add_argument('--quick', action='store_true', help='Run quick test with fewer models')
    args = parser.parse_args()

    if args.quick:
        # Quick test mode with just a few models
        print("Running quick test mode...")
        main(args.data, args.artifacts)
    else:
        main(args.data, args.artifacts)
