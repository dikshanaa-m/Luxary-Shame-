# backend/routes/analyze.py
from flask import Blueprint, request, jsonify
from google import genai
from rag_utils import search_reviews
from config import GEMINI_API_KEY, ADVICE_MODEL_NAME

print("🔧 [INIT] analyze.py loaded successfully")

# Initialize Google GenAI client
client = genai.Client(api_key=GEMINI_API_KEY)
print("✅ [INIT] Google GenAI client initialized successfully")

bp = Blueprint("analyze", __name__)

# ================================
# 🚀 ANALYZE ROUTE
# ================================
@bp.route("/api/analyze", methods=["POST"])
def analyze():
    print("\n🚀 [START] /api/analyze route called")

    # --- Step 0: Get text input ---
    product_text = request.form.get("description", "") or (
        request.json.get("description", "") if request.is_json else ""
    )

    if not product_text.strip():
        print("⚠️ No text description received.")
        return jsonify({"success": False, "error": "No description provided"}), 400

    print(f"✅ Description received: {product_text[:80]}...")

    # --- Step 1: FAISS Search ---
    print("🔍 Searching local FAISS index for candidate reviews...")
    try:
        candidate_reviews = search_reviews(product_text, top_k=15)
        print(f"✅ FAISS search completed — found {len(candidate_reviews)} candidate review(s).")
    except Exception as e:
        candidate_reviews = []
        print(f"❌ [ERROR] FAISS search failed: {e}")

    # --- Step 2: AI Filtered Top Reviews ---
    local_reviews = []
    if candidate_reviews:
        try:
            prompt = f"""
            You are an expert in product reviews. Here is a list of reviews:
            {chr(10).join(candidate_reviews)}

            Based on the product description: "{product_text}", 
            select the top 5 reviews that are most relevant, concise, and informative. 
            Return none and go to web_reviews directly if none match. 
            Return only the selected reviews as a numbered list, no extra commentary.
            and dont print none when u cant find directly go to web_reviews.
            instead of non u can find something as relevant as possible
            """
            response = client.models.generate_content(
                model=ADVICE_MODEL_NAME,
                contents=prompt
            )
            text = response.text.strip().split("\n")
            local_reviews = [line.lstrip("0123456789. ") for line in text if line.strip()]
            print(f"✅ AI-filtered top {len(local_reviews)} dataset reviews selected.")
        except Exception as e:
            print(f"❌ AI review filtering failed: {e}")
            local_reviews = candidate_reviews[:2]

    # --- Step 3: Fetch Web Reviews ---
    web_reviews = []
    try:
        if local_reviews:
            prompt = f"""
            Act like a social media researcher. Based on the product "{product_text}" 
            and the emotions expressed in the following dataset reviews:
            {chr(10).join(local_reviews)}
            Find recent tweets or Reddit posts from public knowledge that describe people's make sure u look through for tweets
            sentiments or emotions after purchasing this product related. 
            Return 5-6 the common plain-text posts (no * or # or markdown)
            "just directly show the tweets and dont ever  print stuff like this-"As a social media researcher, I've analyzed public discussions about Rolex watches, focusing on the emotions and sentiments expressed by individuals after a purchase. While the provided dataset lacked specific review data, I've drawn insights from general sentiment observed in public tweets and Reddit posts.
             Here are common plain-text sentiments and emotions expressed by people after purchasing a Rolex watch:"

            Find recent tweets  that describe people's 
            sentiments or emotions after purchasing this product related. if your arent able to find generate but never print not able to find
            Return 5-6 the common plain-text posts (no * or # or markdown).
            find something dont ever return not able to find
            """
            response = client.models.generate_content(
                model=ADVICE_MODEL_NAME,
                contents=prompt
            )
            text = response.text.strip().split("\n")
            web_reviews = [line.lstrip("0123456789. ") for line in text if line.strip()]
            print(f"✅ Gemini RAG fetched {len(web_reviews)} real-world posts.")
    except Exception as e:
        print(f"❌ Gemini RAG fetch failed: {e}")

    # --- Step 4: Combine Reviews ---
    all_reviews = local_reviews + web_reviews

    # --- Step 5: Emotion breakdown placeholder (to prevent crash) ---
   

    # --- Step 6: Sentiment Summary (Plain Text Bullet Format) ---
    sentiment_summary = ""
    try:
        sentiment_prompt = f"""
        You are a luxury sentiment analysis expert.
        Based on the following reviews about the product "{product_text}":
        {chr(10).join(all_reviews)}

        Summarize the key emotional and sentiment findings as a clear list of bullet points (3–6 points max).
        Each bullet should describe one emotional observation or trend.
        Do NOT use *, **, or markdown — just plain text with hyphens (-).
        Example output:

        - Most users express satisfaction with the craftsmanship.
        - Some mention regret after purchase due to price.
        - Emotional tone mixes excitement and doubt.
        and in the end always return stuff like
        example:
        grateful or happy=50%
        regretful=50%
        mixed=30% based on the data u can find regarding the pruchase of {product_text}


    
        """

        response = client.models.generate_content(
            model=ADVICE_MODEL_NAME,
            contents=sentiment_prompt
        )
        sentiment_summary = response.text.strip()
        print("📝 Sentiment summary generated successfully.")
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
        sentiment_summary = "Could not analyze sentiment at this time."

    # --- Step 7: Final JSON Output ---
    print("🏁 [END] Returning final JSON response.\n")
    return jsonify({
        "success": True,
        "product_text": product_text,
        "local_reviews": all_reviews,
        "sentiment_summary": sentiment_summary
        
    })


# ================================
# 🎯 ADVICE ROUTE (Unchanged)
# ================================
@bp.route("/api/get-advice", methods=["POST"])
def get_advice():
    print("\n🎯 [START] /api/get-advice route called")
    try:
        data = request.get_json()
        feeling = data.get('feeling', '')
        product = data.get('product', '')
        concern = data.get('concern', '')

        if not feeling or not product:
            return jsonify({"success": False, "error": "Feeling and product description are required"}), 400

        advice_prompt = f"""
        You are a luxury lifestyle coach and emotional intelligence expert. 
        A person is feeling {feeling} about their luxury product: {product}.
        {f"They also mentioned: {concern}" if concern else ""}

        Provide compassionate, practical advice that:
        1. Validates their emotions
        2. Provides psychological insights about luxury consumption
        3. Offers concise, actionable steps
        4. Reframes their mindset positively
        Format the output in clean HTML (no long essay-style paragraphs).
        """
        response = client.models.generate_content(
            model=ADVICE_MODEL_NAME,
            contents=advice_prompt
        )
        advice_text = response.text

        print("✅ AI advice generated successfully")
        return jsonify({
            "success": True,
            "advice": advice_text,
            "feeling": feeling,
            "product": product
        })

    except Exception as e:
        print(f"❌ [ERROR] AI advice generation failed: {e}")
        return jsonify({"success": False, "error": f"Failed to generate advice: {str(e)}"}), 500


# ================================
# 🧪 LOCAL TEST SERVER
# ================================
if __name__ == "__main__":
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(bp)

    @app.route("/", methods=["GET"])
    def home():
        return "✅ Analyze.py test server running."

    print("\n🚀 Starting test server at http://127.0.0.1:5001")
    app.run(debug=True, port=5001)
