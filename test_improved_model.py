"""
Test improved model on sample images
Shows predictions with confidence scores and flags confusing pairs
"""

import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import glob

# ============== CONFIGURATION ==============

MODEL_PATH = 'cnn_model_improved.keras'
LABEL_MAP_PATH = 'label_mapping.json'

# Confusing pairs that model struggles with
CONFUSING_PAIRS = {
    ('Potato___Late_blight', 'Tomato_Late_blight'): 'Cross-plant confusion',
    ('Tomato_Target_Spot', 'Tomato_Septoria_leaf_spot'): 'Similar fungal spots',
    ('Tomato_Leaf_Mold', 'Tomato_Early_blight'): 'Similar appearance'
}

def load_model_and_labels():
    """Load model and label mapping"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("\nRun training first:")
        print("  python train_improved.py")
        return None, None
    
    try:
        model = load_model(MODEL_PATH)
        print(f"✓ Loaded model: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None
    
    if not os.path.exists(LABEL_MAP_PATH):
        print(f"❌ Label mapping not found: {LABEL_MAP_PATH}")
        return None, None
    
    with open(LABEL_MAP_PATH, 'r') as f:
        class_indices = json.load(f)
        class_labels = {v: k for k, v in class_indices.items()}
    
    return model, class_labels

def extract_plant_type(class_name):
    """Extract plant type from class name"""
    if class_name.startswith('Pepper'):
        return 'Pepper'
    elif class_name.startswith('Potato'):
        return 'Potato'
    else:
        return 'Tomato'

def format_disease_name(class_name):
    """Format disease name for display"""
    if '__' in class_name:
        disease = class_name.replace('__', ' ').split('_', 1)[1]
    else:
        disease = class_name.split('_', 1)[1] if '_' in class_name else class_name
    
    disease = disease.replace('_', ' ')
    return disease.title()

def check_confusing_pair(pred_class, top_predictions):
    """Check if prediction is in a confusing pair"""
    for other_class, _ in top_predictions[1:]:
        for (class1, class2), reason in CONFUSING_PAIRS.items():
            if (pred_class == class1 and other_class == class2) or \
               (pred_class == class2 and other_class == class1):
                return reason
    return None

def predict_image(model, class_labels, image_path):
    """Predict disease class for a single image"""
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Get predictions
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get top 5 predictions
        top_indices = np.argsort(predictions)[::-1][:5]
        top_predictions = [
            (class_labels[idx], float(predictions[idx]))
            for idx in top_indices
        ]
        
        return top_predictions, None
    except Exception as e:
        return None, str(e)

def print_prediction_result(image_path, top_predictions):
    """Pretty print prediction results"""
    if top_predictions is None:
        return
    
    pred_class, confidence = top_predictions[0]
    plant = extract_plant_type(pred_class)
    disease = format_disease_name(pred_class)
    
    # Colors for output
    if confidence > 0.85:
        status_icon = "✅ HIGH"
        status_color = "\033[92m"  # Green
    elif confidence > 0.70:
        status_icon = "⚠️  MED"
        status_color = "\033[93m"  # Yellow
    else:
        status_icon = "❌ LOW"
        status_color = "\033[91m"  # Red
    
    reset_color = "\033[0m"
    
    print(f"\n{'='*70}")
    print(f"📷 {os.path.basename(image_path)}")
    print(f"{'='*70}")
    print(f"\n{status_color}{status_icon} CONFIDENCE: {confidence*100:.2f}%{reset_color}")
    print(f"   Plant: {plant}")
    print(f"   Disease: {disease}")
    
    # Check for confusing pairs
    confusing = check_confusing_pair(pred_class, top_predictions)
    if confusing and confidence < 0.80:
        print(f"\n   ⚠️  Warning: {confusing}")
    
    # Show top predictions
    print(f"\n   Top Predictions:")
    for i, (class_name, conf) in enumerate(top_predictions, 1):
        disease_name = format_disease_name(class_name)
        bar_width = int(conf * 20)
        bar = '█' * bar_width + '░' * (20 - bar_width)
        print(f"   {i}. {disease_name:35s} {bar} {conf*100:5.1f}%")
    
    # Check for cross-plant confusion
    plants = [extract_plant_type(c) for c, _ in top_predictions[:3]]
    if len(set(plants)) > 1:
        print(f"\n   ⚠️  Multiple plants in top 3 - verify plant type!")

def main():
    print("\n" + "="*70)
    print("PLANT DISEASE PREDICTION TEST")
    print("="*70)
    
    # Load model
    model, class_labels = load_model_and_labels()
    if model is None:
        return
    
    print(f"✓ Loaded {len(class_labels)} disease classes\n")
    
    # Find test images
    test_dirs = [
        'dataset/test',
        'archive/dataset/test',
        'sample_images',
        'test_images'
    ]
    
    image_files = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            image_files.extend(glob.glob(os.path.join(test_dir, '**/*.jpg'), recursive=True))
            image_files.extend(glob.glob(os.path.join(test_dir, '**/*.png'), recursive=True))
    
    if not image_files:
        print("❌ No test images found!")
        print("\nTip: Place images in one of these locations:")
        for d in test_dirs:
            print(f"  • {d}/")
        print("\nOr run manual testing by providing image path:")
        print("  python test_improved_model.py path/to/image.jpg")
        return
    
    # Limit to first 10 for demonstration
    print(f"Found {len(image_files)} images. Testing first 10...\n")
    
    for image_path in image_files[:10]:
        print(f"\nProcessing: {image_path}")
        top_predictions, error = predict_image(model, class_labels, image_path)
        
        if error:
            print(f"❌ Error: {error}")
        else:
            print_prediction_result(image_path, top_predictions)
    
    print(f"\n{'='*70}")
    print("✓ Testing complete!")
    print(f"{'='*70}\n")

def test_single_image(image_path):
    """Test a single image provided as argument"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print("\n" + "="*70)
    print("PLANT DISEASE PREDICTION - SINGLE IMAGE")
    print("="*70)
    
    # Load model
    model, class_labels = load_model_and_labels()
    if model is None:
        return
    
    top_predictions, error = predict_image(model, class_labels, image_path)
    
    if error:
        print(f"❌ Error: {error}")
    else:
        print_prediction_result(image_path, top_predictions)
    
    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Single image testing
        test_single_image(sys.argv[1])
    else:
        # Batch testing
        main()
