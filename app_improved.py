"""
Flask backend for improved plant disease detection
Features:
- Confidence scoring and thresholding
- Detection of confusing disease pairs
- Plant-specific validation
- Enhanced prediction display
"""

from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import json
import os
from PIL import Image
import io

app = Flask(__name__)

# Load the improved model
MODEL_PATH = 'cnn_model_improved.keras'
LABEL_MAP_PATH = 'label_mapping.json'

# Load model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print(f"✓ Loaded improved model: {MODEL_PATH}")
else:
    print(f"⚠ {MODEL_PATH} not found. Using fallback...")
    MODEL_PATH = 'cnn_model.keras'
    model = load_model(MODEL_PATH)

# Load label mapping
with open(LABEL_MAP_PATH, 'r') as f:
    class_indices = json.load(f)
    class_labels = {v: k for k, v in class_indices.items()}

# ============== CONFUSION PAIR DEFINITIONS ==============
# These pairs are known to be confused by the model
CONFUSING_PAIRS = {
    ('Potato___Late_blight', 'Tomato_Late_blight'): {
        'confidence_threshold': 0.75,
        'advice': 'High confusion between Potato and Tomato Late Blight. Check leaf morphology carefully.',
        'distinguishing_features': [
            'Potato leaves are elongated, Tomato leaves are more serrated',
            'Potato blight: concentric rings with brown center',
            'Tomato blight: more scattered lesions'
        ]
    },
    
    ('Tomato_Target_Spot', 'Tomato_Septoria_leaf_spot'): {
        'confidence_threshold': 0.80,
        'advice': 'Model has difficulty distinguishing Target Spot from Septoria. Look for concentric rings.',
        'distinguishing_features': [
            'Target Spot: prominent concentric rings, yellow halo',
            'Septoria: smaller scattered spots, darker centers',
            'Target Spot lesions are larger and more circular'
        ]
    },
    
    ('Tomato_Leaf_Mold', 'Tomato_Early_blight'): {
        'confidence_threshold': 0.75,
        'advice': 'Similar appearance. Leaf Mold has fuzzy appearance on leaf underside.',
        'distinguishing_features': [
            'Leaf Mold: yellow chlorosis on top, grayish mold on bottom',
            'Early Blight: concentric rings (target-like) on both surfaces',
            'Check leaf underside for fuzzy fungal growth'
        ]
    }
}

# ============== HELPER FUNCTIONS ==============

def extract_plant_type(class_name):
    """Extract plant type from class name"""
    if class_name.startswith('Pepper'):
        return 'Pepper'
    elif class_name.startswith('Potato'):
        return 'Potato'
    else:
        return 'Tomato'

def is_confusing_pair(true_class, pred_class):
    """Check if this is a known confusing pair"""
    for (class1, class2), info in CONFUSING_PAIRS.items():
        if (true_class == class1 and pred_class == class2) or \
           (true_class == class2 and pred_class == class1):
            return True, info
    return False, None

def get_confidence_message(confidence, class_name):
    """Get message based on confidence level"""
    if confidence >= 0.90:
        return f"Very confident ({confidence*100:.1f}%)"
    elif confidence >= 0.75:
        return f"Confident ({confidence*100:.1f}%)"
    elif confidence >= 0.60:
        return f"Somewhat confident ({confidence*100:.1f}%)"
    else:
        return f"Low confidence ({confidence*100:.1f}%) - Verify manually!"

def validate_cross_plant_confusion(top_predictions):
    """Check if there's dangerous cross-plant confusion"""
    plants = [extract_plant_type(label) for label, _ in top_predictions]
    
    if len(set(plants)) > 1:
        # Multiple plants in top predictions - warning!
        return True, plants[0]
    return False, None

def format_disease_name(class_name):
    """Format disease name for display"""
    # Remove plant prefix
    if '__' in class_name:
        disease = class_name.replace('__', ' ').split('_', 1)[1]
    else:
        disease = class_name.split('_', 1)[1] if '_' in class_name else class_name
    
    # Clean up underscores
    disease = disease.replace('_', ' ')
    return disease.title()

# ============== MAIN PREDICTION ENDPOINT ==============


@app.route('/', methods=['GET'])
def index():
    return send_from_directory('web', 'index.html')


@app.route('/app.js', methods=['GET'])
def frontend_js():
    return send_from_directory('web', 'app.js')


@app.route('/styles.css', methods=['GET'])
def frontend_css():
    return send_from_directory('web', 'styles.css')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided', 'status': 'ERROR'}), 400
        
        file = request.files['image']
        
        # Read and preprocess image
        img_stream = io.BytesIO(file.read())
        img = Image.open(img_stream).convert('RGB')
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
        
        # Primary prediction
        predicted_class = top_predictions[0][0]
        confidence = top_predictions[0][1]
        
        # ============== ADVANCED CHECKS ==============
        
        # 1. Check for low confidence
        low_confidence = confidence < 0.60
        
        # 2. Check for dangerous cross-plant confusion
        has_cross_plant, primary_plant = validate_cross_plant_confusion(top_predictions)
        
        # 3. Check if predicted class is in a confusing pair
        is_confusing = False
        confusion_info = None
        if confidence < 0.85:  # Only flag if confidence is moderate
            for other_class, _ in top_predictions[1:3]:
                is_confusing, confusion_info = is_confusing_pair(predicted_class, other_class)
                if is_confusing:
                    break
        
        # Extract plant and disease
        plant = extract_plant_type(predicted_class)
        disease = format_disease_name(predicted_class)
        
        # ============== BUILD RESPONSE ==============
        
        response = {
            'status': 'HIGH_CONFIDENCE' if confidence > 0.85 else \
                      'MEDIUM_CONFIDENCE' if confidence > 0.70 else \
                      'LOW_CONFIDENCE',
            'predicted_class': predicted_class,
            'plant': plant,
            'disease': disease,
            'confidence': confidence,
            'confidence_message': get_confidence_message(confidence, predicted_class),
            'top_predictions': [
                {
                    'class': label,
                    'confidence': conf,
                    'display': f"{format_disease_name(label)}: {conf*100:.2f}%"
                }
                for label, conf in top_predictions
            ],
            'message': '',
            'warnings': [],
            'distinguishing_features': []
        }
        
        # ============== ADD WARNINGS && ADVICE ==============
        
        if low_confidence:
            response['warnings'].append('⚠ Low confidence - image quality or similar disease pattern')
            response['message'] = 'Low confidence prediction. Please verify with expert or retake photo.'
        
        if has_cross_plant and confidence < 0.75:
            response['warnings'].append(f'⚠ Multiple plants in top predictions - verify it\'s a {primary_plant} plant!')
            response['message'] = f'Model uncertain about plant type. This appears to be {primary_plant}, but verify manually.'
        
        if is_confusing and confusion_info:
            response['warnings'].append(f"⚠ This prediction is in a commonly confused pair")
            response['message'] = confusion_info['advice']
            response['distinguishing_features'] = confusion_info['distinguishing_features']
        
        # If no message set, create default
        if not response['message']:
            if confidence > 0.90:
                response['message'] = f"High confidence identification: {disease} on {plant}"
            elif confidence > 0.75:
                response['message'] = f"Identified as {disease} on {plant} (confidence: {confidence*100:.1f}%)"
            else:
                response['message'] = f"Possible {disease} on {plant} - low confidence, verify with expert"
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'status': 'ERROR'
        }), 500

# ============== HEALTH CHECK ==============

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': 'improved' if 'improved' in MODEL_PATH else 'standard',
        'classes': len(class_labels),
        'confusing_pairs': len(CONFUSING_PAIRS)
    })

# ============== MODEL INFO ==============

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'model_path': MODEL_PATH,
        'classes': list(sorted(set([extract_plant_type(c) for c in class_labels.values()]))),
        'total_classes': len(class_labels),
        'confusing_pairs_detected': len(CONFUSING_PAIRS),
        'improvements': [
            'Class weights for imbalanced data',
            'Enhanced data augmentation',
            'Confidence thresholding',
            'Confusing pair detection',
            'Cross-plant validation'
        ]
    })

if __name__ == '__main__':
    print("🌱 Plant Disease Detection API (Improved Model)")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Classes: {len(class_labels)}")
    print(f"   Confusing pairs tracked: {len(CONFUSING_PAIRS)}")
    print("\nStarting Flask server...")
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
