"""
Compare original model vs improved model
Shows side-by-side performance metrics and confusion matrices
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Configuration
DATASET_DIR = "dataset"
TEST_DIR = os.path.join(DATASET_DIR, "test")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Define disease classes
DISEASE_CLASSES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_healthy'
]

def evaluate_model(model, test_data, model_name):
    """Evaluate model and return metrics"""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*60}")
    
    y_pred_all = []
    y_true_all = []
    
    for images, labels in test_data:
        predictions = model.predict(images, verbose=0)
        y_pred_all.extend(np.argmax(predictions, axis=1))
        y_true_all.extend(np.argmax(labels, axis=1))
    
    y_pred = np.array(y_pred_all)
    y_true = np.array(y_true_all)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"\n✓ Overall Accuracy: {accuracy*100:.2f}%")
    
    # Per-class accuracy
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    per_class_accuracy = []
    for i in range(len(DISEASE_CLASSES)):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        per_class_accuracy.append(class_acc)
        print(f"  {DISEASE_CLASSES[i]:45s}: {class_acc*100:6.2f}%  ({cm[i, i]}/{cm[i].sum()})")
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'name': model_name
    }

def find_biggest_improvements(orig_cm, improved_cm):
    """Find classes with biggest accuracy improvements"""
    print(f"\n{'='*60}")
    print("BIGGEST IMPROVEMENTS")
    print(f"{'='*60}\n")
    
    improvements = []
    for i in range(len(DISEASE_CLASSES)):
        orig_acc = orig_cm[i, i] / orig_cm[i].sum() if orig_cm[i].sum() > 0 else 0
        improved_acc = improved_cm[i, i] / improved_cm[i].sum() if improved_cm[i].sum() > 0 else 0
        improvement = improved_acc - orig_acc
        improvements.append((improvement, DISEASE_CLASSES[i], orig_acc, improved_acc))
    
    improvements.sort(reverse=True)
    
    for improvement, disease, orig, improved in improvements[:8]:
        if improvement > 0:
            print(f"  ✅ {disease:45s}: {orig*100:5.1f}% → {improved*100:5.1f}% (+{improvement*100:5.1f}%)")
        elif improvement < 0:
            print(f"  ⚠️  {disease:45s}: {orig*100:5.1f}% → {improved*100:5.1f}% ({improvement*100:5.1f}%)")

def find_reduced_confusions(orig_cm, improved_cm):
    """Find confusion pairs that were reduced"""
    print(f"\n{'='*60}")
    print("REDUCED CONFUSION PAIRS")
    print(f"{'='*60}\n")
    
    confusion_reductions = []
    
    for i in range(len(DISEASE_CLASSES)):
        for j in range(len(DISEASE_CLASSES)):
            if i != j:
                orig_confusion = orig_cm[i, j]
                improved_confusion = improved_cm[i, j]
                reduction = orig_confusion - improved_confusion
                
                if reduction > 0:
                    confusion_reductions.append((
                        reduction,
                        DISEASE_CLASSES[i],
                        DISEASE_CLASSES[j],
                        orig_confusion,
                        improved_confusion
                    ))
    
    confusion_reductions.sort(reverse=True)
    
    for reduction, true_class, pred_class, orig, improved in confusion_reductions[:10]:
        print(f"  ✅ {true_class:35s} → {pred_class:35s}: {orig} → {improved} (-{reduction})")

def create_comparison_visualization(orig_results, improved_results):
    """Create side-by-side comparisons"""
    print("\n📊 Creating comparison visualizations...")
    
    # Figure 1: Accuracy comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Overall accuracy
    ax = axes[0]
    models = ['Original', 'Improved']
    accuracies = [orig_results['accuracy'] * 100, improved_results['accuracy'] * 100]
    colors = ['#FF6B6B', '#51CF66']
    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, width=0.6)
    ax.set_ylim([80, 95])
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Per-class improvement distribution
    ax = axes[1]
    improvements = [
        (improved - orig) * 100
        for orig, improved in zip(orig_results['per_class_accuracy'], improved_results['per_class_accuracy'])
    ]
    improvements.sort()
    colors_dist = ['#FF6B6B' if x < 0 else '#51CF66' for x in improvements]
    ax.barh(range(len(improvements)), improvements, color=colors_dist, alpha=0.8)
    ax.set_yticks(range(len(DISEASE_CLASSES)))
    ax.set_yticklabels(DISEASE_CLASSES, fontsize=8)
    ax.set_xlabel('Accuracy Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy Improvement', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_overview.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_comparison_overview.png")
    plt.close()
    
    # Figure 2: Confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Original confusion matrix
    ax = axes[0]
    sns.heatmap(orig_results['confusion_matrix'], annot=False, fmt='d', 
                cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xticklabels(DISEASE_CLASSES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(DISEASE_CLASSES, rotation=0, fontsize=8)
    ax.set_title('Original Model Confusion Matrix', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Improved confusion matrix
    ax = axes[1]
    sns.heatmap(improved_results['confusion_matrix'], annot=False, fmt='d',
                cmap='Greens', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xticklabels(DISEASE_CLASSES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(DISEASE_CLASSES, rotation=0, fontsize=8)
    ax.set_title('Improved Model Confusion Matrix', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrices_comparison.png")
    plt.close()
    
    # Figure 3: Diagonal elements (correct predictions)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    orig_diag = np.diag(orig_results['confusion_matrix'])
    improved_diag = np.diag(improved_results['confusion_matrix'])
    
    x = np.arange(len(DISEASE_CLASSES))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, orig_diag, width, label='Original', alpha=0.8, color='#FF6B6B')
    bars2 = ax.bar(x + width/2, improved_diag, width, label='Improved', alpha=0.8, color='#51CF66')
    
    ax.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Correct Predictions', fontsize=12, fontweight='bold')
    ax.set_title('Diagonal Elements: Correct Predictions by Disease', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(DISEASE_CLASSES, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correct_predictions_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: correct_predictions_comparison.png")
    plt.close()

def main():
    print("\n" + "="*70)
    print("PLANT DISEASE MODEL COMPARISON - Original vs Improved")
    print("="*70)
    
    # Load test data
    print("\nLoading test data...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    print(f"✓ Loaded {len(test_generator.filenames)} test images")
    
    # Load models
    print("\nLoading models...")
    try:
        original_model = load_model('cnn_model.keras')
        print("✓ Loaded original model: cnn_model.keras")
    except:
        print("❌ Could not load cnn_model.keras")
        return
    
    try:
        improved_model = load_model('cnn_model_improved.keras')
        print("✓ Loaded improved model: cnn_model_improved.keras")
    except:
        print("⚠️  Could not load cnn_model_improved.keras")
        print("   Please run: python train_improved.py")
        return
    
    # Evaluate both models
    test_generator.reset()
    orig_results = evaluate_model(original_model, test_generator, "Original Model")
    
    test_generator.reset()
    improved_results = evaluate_model(improved_model, test_generator, "Improved Model")
    
    # Compare results
    print(f"\n{'='*60}")
    print("OVERALL COMPARISON")
    print(f"{'='*60}")
    improvement = (improved_results['accuracy'] - orig_results['accuracy']) * 100
    print(f"Accuracy improvement: {orig_results['accuracy']*100:.2f}% → {improved_results['accuracy']*100:.2f}% (+{improvement:.2f}%)")
    
    # Find specific improvements
    find_biggest_improvements(orig_results['confusion_matrix'], improved_results['confusion_matrix'])
    find_reduced_confusions(orig_results['confusion_matrix'], improved_results['confusion_matrix'])
    
    # Create visualizations
    create_comparison_visualization(orig_results, improved_results)
    
    print(f"\n{'='*60}")
    print("✓ COMPARISON COMPLETE")
    print(f"{'='*60}")
    print("\nGenerated files:")
    print("  • model_comparison_overview.png")
    print("  • confusion_matrices_comparison.png")
    print("  • correct_predictions_comparison.png")

if __name__ == '__main__':
    main()
