from flask import Flask, render_template, request, jsonify
import json
import os
from datetime import datetime
import threading
import traceback

app = Flask(__name__)

# =============================================================================
# PATH CONFIG
# =============================================================================
RAW_DATA_PATH = "data/rawdata/raw.txt"
REVIEWS_FILE = "data/reviews.json"
FAISS_INDEX_PATH = "vstore/faiss_index"


# =============================================================================
# SAFE IMPORT HELPERS
# =============================================================================
def try_import(module_name, alias=None):
    """Safely import a module and print the result."""
    try:
        mod = __import__(module_name, fromlist=[''])
        print(f"✅ Successfully loaded module: {module_name}")
        if alias:
            globals()[alias] = mod
        return mod
    except Exception as e:
        print(f"❌ Failed to import {module_name}: {e}")
        print(traceback.format_exc())
        return None


# Try loading optional modules
config = try_import("config")
rag_utils = try_import("rag_utils")
analyze_module = try_import("analyze")
data_processor = try_import("data_processor")
model_evaluator = try_import("model_evaluator")

print("\n--- Initialization Summary ---")
print(f"config loaded: {config is not None}")
print(f"rag_utils loaded: {rag_utils is not None}")
print(f"analyze.py loaded: {analyze_module is not None}")
print(f"data_processor loaded: {data_processor is not None}")
print(f"model_evaluator loaded: {model_evaluator is not None}")
print("--------------------------------\n")

# =============================================================================
# REGISTER BLUEPRINTS
# =============================================================================
if analyze_module and hasattr(analyze_module, 'bp'):
    app.register_blueprint(analyze_module.bp)
    print("✅ Analyze blueprint registered successfully")
else:
    print("❌ Analyze blueprint not available")

# =============================================================================
# FRONTEND ROUTES
# =============================================================================
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/metrics')
def analysis():
    return render_template('metrics.html')

@app.route('/advice')
def advice():
    return render_template('advice.html')

@app.route('/reviews')
def reviews():
    return render_template('reviews.html')


# =============================================================================
# REVIEW SYSTEM
# =============================================================================
@app.route('/api/submit-review', methods=['POST'])
def submit_review():
    try:
        data = request.get_json(force=True)
        review_text = data.get('review', '').strip()
        if not review_text:
            return jsonify({'success': False, 'error': 'Review text required'})

        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        with open(RAW_DATA_PATH, 'a', encoding='utf-8') as f:
            f.write(f"\n{review_text}")

        review_data = {
            'content': review_text,
            'timestamp': datetime.now().isoformat(),
            'username': 'Anonymous User'
        }

        reviews = []
        if os.path.exists(REVIEWS_FILE):
            with open(REVIEWS_FILE, 'r', encoding='utf-8') as f:
                reviews = json.load(f)
        reviews.append(review_data)

        os.makedirs(os.path.dirname(REVIEWS_FILE), exist_ok=True)
        with open(REVIEWS_FILE, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, indent=2)

        if data_processor and hasattr(data_processor, "process_and_embed_data_faiss"):
            try:
                data_processor.process_and_embed_data_faiss()
                print("✅ Review successfully processed into FAISS embeddings.")
            except Exception as e:
                print(f"⚠️ Embedding update failed: {e}")
        else:
            print("⚠️ Skipping embedding — data_processor not available")

        return jsonify({'success': True, 'message': 'Review saved successfully'})

    except Exception as e:
        print(f"❌ Error in /api/submit-review: {e}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/reviews')
def get_reviews():
    try:
        if os.path.exists(REVIEWS_FILE):
            with open(REVIEWS_FILE, 'r', encoding='utf-8') as f:
                reviews = json.load(f)
            return jsonify(sorted(reviews, key=lambda x: x['timestamp'], reverse=True))
        else:
            return jsonify([])
    except Exception as e:
        print(f"❌ Error in /api/reviews: {e}")
        return jsonify({'error': str(e)})


# =============================================================================
# ML EVALUATION
# =============================================================================
@app.route('/api/evaluate-models')
def evaluate_models():
    try:
        if not model_evaluator:
            raise ImportError("model_evaluator not available")
        evaluator = model_evaluator.MLEvaluator(FAISS_INDEX_PATH, RAW_DATA_PATH)
        frontend_metrics = evaluator.run_all_evaluations()
        return jsonify({'success': True, 'metrics': frontend_metrics})
    except Exception as e:
        print(f"❌ Error in /api/evaluate-models: {e}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


# =============================================================================
# RAG / ANALYSIS ENDPOINTS
# =============================================================================
# REMOVED the /api/analyze route - now handled by blueprint

@app.route('/api/rag-query', methods=['POST'])
def rag_query():
    try:
        if not rag_utils or not hasattr(rag_utils, "search_reviews"):
            raise ImportError("rag_utils or search_reviews() not found")

        data = request.get_json(force=True)
        question = data.get('query', '').strip()
        if not question:
            return jsonify({'success': False, 'error': 'Query text required'})

        results = rag_utils.search_reviews(question)
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        print(f"❌ Error in /api/rag-query: {e}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


# =============================================================================
# APP INITIALIZATION
# =============================================================================
@app.before_request
def initialize_data():
    try:
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(REVIEWS_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        if not os.path.exists(REVIEWS_FILE):
            with open(REVIEWS_FILE, 'w') as f:
                json.dump([], f)
    except Exception as e:
        print(f"⚠️ Error initializing data files: {e}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("\n🚀 Starting Flask app in TEST MODE")
    print("This will show which components are missing or failing.")
    print("Visit http://127.0.0.1:5000 to test your app.\n")
    app.run(debug=True)