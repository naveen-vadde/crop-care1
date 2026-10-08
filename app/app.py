
# Importing essential libraries and modules

from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import base64
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from groq import Groq
import markdown
import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import FuncFormatter
from tavily import TavilyClient
from database import (
    init_db, ping, DuplicateKeyError, PyMongoError,
    create_user, get_user_by_id, get_user_by_mobile,
    create_plant, get_plant, list_plants, delete_plant,
    add_record, list_records, get_record, delete_record,
    record_to_dict, plant_to_dict,
)
from utils.severity_analyzer import (
    calculate_severity_score, 
    estimate_affected_area, 
    analyze_progression,
    get_severity_color
)
from datetime import datetime
from urllib.parse import urlparse, urljoin
import html as html_lib
import logging
import re
import uuid
import os
from matplotlib.figure import Figure
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = os.path.join(config.BASE_DIR, 'models', 'plant_disease_model.pth')
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = os.path.join(config.BASE_DIR, 'models', 'RandomForest.pkl')
with open(crop_recommendation_model_path, 'rb') as _f:
    crop_recommendation_model = pickle.load(_f)

# Fertilizer reference table (ideal N/P/K per crop) - loaded once at start-up
fertilizer_df = pd.read_csv(os.path.join(config.BASE_DIR, 'Data', 'fertilizer.csv'))
fertilizer_df['Crop'] = fertilizer_df['Crop'].astype(str).str.strip().str.lower()

CROP_FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch the temperature (deg C) and humidity (%) of a city from OpenWeatherMap.
    :return: (temperature, humidity) or None if it could not be fetched
    """
    if not config.weather_api_key:
        app.logger.error("WEATHER_API_KEY is not set")
        return None
    if not city_name:
        return None
    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city_name, "appid": config.weather_api_key, "units": "metric"},
            timeout=8,
        )
        x = response.json()
        if "main" in x:
            return round(float(x["main"]["temp"]), 2), x["main"]["humidity"]
        app.logger.warning(f"Weather API error for city '{city_name}': {x}")
        return None
    except Exception as e:
        app.logger.error(f"Weather fetch failed: {e}")
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image bytes to a tensor and predicts the disease label
    :return: prediction (string), confidence (float, percent)
    """
    # The network was trained on 256x256 RGB images. Forcing RGB (PNG/alpha/
    # greyscale uploads would crash) and an exact 256x256 size (other aspect
    # ratios break the final Linear layer) keeps every upload working.
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img)).convert('RGB')
    img_u = torch.unsqueeze(transform(image), 0)

    with torch.no_grad():
        yb = model(img_u)
        probs = torch.nn.functional.softmax(yb, dim=1)
        confidence, preds = torch.max(probs, dim=1)
    prediction = disease_classes[preds[0].item()]
    confidence_pct = round(confidence[0].item() * 100, 2)
    return prediction, confidence_pct


# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------

app = Flask(__name__)

# Secret key for sessions
if not config.SECRET_KEY:
    app.logger.warning("SECRET_KEY is not set - using an insecure development key. "
                       "Set SECRET_KEY in .env before deploying!")
app.secret_key = config.SECRET_KEY or 'dev-only-insecure-key'

app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'   # blocks cross-site form posts

logging.basicConfig(level=logging.INFO)

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Connect to MongoDB Atlas (URI comes from MONGODB_URI in .env) and make sure
# the indexes exist. Fails fast with a clear message if the connection is wrong.
try:
    init_db(app)
except RuntimeError as exc:
    app.logger.error("DATABASE ERROR: %s", exc)
    raise

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please login to access this page.'
login_manager.login_message_category = 'warning'



@login_manager.user_loader
def load_user(user_id):
    try:
        return get_user_by_id(user_id)
    except PyMongoError as e:
        app.logger.error(f"User lookup failed: {e}")
        return None


def is_safe_redirect(target):
    """Only allow redirects back to this same site (prevents open redirect)."""
    if not target:
        return False
    ref = urlparse(request.host_url)
    test = urlparse(urljoin(request.host_url, target))
    return test.scheme in ('http', 'https') and ref.netloc == test.netloc


@app.route('/health')
def health():
    """Used by Docker / hosting health checks. Also verifies the database."""
    try:
        ping()
        return jsonify(status='ok', database='up')
    except Exception as e:
        app.logger.error(f"Health check failed: {e}")
        return jsonify(status='error', database='down'), 503

# ===============================================================================================
# ------------------------------------ AUTHENTICATION -------------------------------------------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Farmer registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        mobile = request.form.get('mobile', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        if not name or not mobile or not password:
            flash('All fields are required!', 'danger')
            return render_template('register.html')
        
        if len(mobile) != 10 or not mobile.isdigit():
            flash('Mobile number must be 10 digits!', 'danger')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters!', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return render_template('register.html')
        
        # Check if mobile already exists
        try:
            if get_user_by_mobile(mobile):
                flash('Mobile number already registered! Please login.', 'warning')
                return redirect(url_for('login'))

            # Create new user (the unique index on `mobile` also guards against races)
            user = create_user(name[:100], mobile, password)

            flash(f'Welcome {name}! Your account has been created successfully.', 'success')
            login_user(user)
            return redirect(url_for('home'))
        except DuplicateKeyError:
            flash('Mobile number already registered! Please login.', 'warning')
            return redirect(url_for('login'))
        except Exception as e:
            app.logger.error(f"Registration error: {e}")
            flash('Registration failed. Please try again.', 'danger')
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Farmer login page"""
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        mobile = request.form.get('mobile', '').strip()
        password = request.form.get('password', '')
        
        if not mobile or not password:
            flash('Mobile number and password are required!', 'danger')
            return render_template('login.html')
        
        try:
            user = get_user_by_mobile(mobile)
        except PyMongoError as e:
            app.logger.error(f"Login lookup failed: {e}")
            flash('Database is temporarily unavailable. Please try again.', 'danger')
            return render_template('login.html')
        
        if user and user.check_password(password):
            login_user(user)
            flash(f'Welcome back, {user.name}!', 'success')
            
            # Redirect to next page or home
            next_page = request.args.get('next')
            return redirect(next_page) if is_safe_redirect(next_page) else redirect(url_for('home'))
        else:
            flash('Invalid mobile number or password!', 'danger')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))


# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


def get_crop_tips(crop_name):
    """Use Groq LLM to get best practices for growing the recommended crop."""
    try:
        client = Groq(api_key=config.groq_api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Give me 5 concise best practices for growing {crop_name} successfully. "
                        "Format as a numbered list. Keep each point to 1-2 sentences. "
                        "Focus on soil preparation, watering, fertilization, pest control, and harvesting tips."
                    ),
                }
            ],
            model=config.GROQ_TIPS_MODEL,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        app.logger.error(f"External API call failed: {e}")
        return None


def translate_to_hindi(text):
    """Translate given plain text to Hindi using Groq llama-3.3-70b-versatile."""
    try:
        client = Groq(api_key=config.groq_api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Translate the following text to Hindi. "
                        "Keep the same formatting (numbered lists, bold etc). "
                        "Return only the translated text, nothing else.\n\n"
                        + text
                    ),
                }
            ],
            model=config.GROQ_CHAT_MODEL,
        )
        result = chat_completion.choices[0].message.content
        return Markup(markdown.markdown(result))
    except Exception as e:
        app.logger.error(f"External API call failed: {e}")
        return None


def tavily_search(query, topic="general", max_results=5, include_images=True):
    """Run a Tavily search and return results + image URLs."""
    try:
        client = TavilyClient(api_key=config.tavily_api_key)
        response = client.search(
            query=query,
            topic=topic,
            max_results=max_results,
            include_images=include_images,
            include_answer=True,
            search_depth="advanced",
        )
        return response
    except Exception as e:
        app.logger.error(f"External API call failed: {e}")
        return None


def groq_vision_describe(image_url, prompt):
    """Use Groq vision model to describe/analyze an image from URL."""
    try:
        client = Groq(api_key=config.groq_api_key)
        completion = client.chat.completions.create(
            model=config.GROQ_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_completion_tokens=512,
        )
        return completion.choices[0].message.content
    except Exception as e:
        app.logger.error(f"External API call failed: {e}")
        return None


def make_cost_revenue_chart(cost, revenue, crop_name):
    """Generate a bar chart comparing estimated cost vs revenue. Returns base64 PNG."""
    fig = Figure(figsize=(5, 3.5))
    ax = fig.subplots()
    bars = ax.bar(['Input Cost', 'Expected Revenue'], [cost, revenue],
                  color=['#f4a261', '#52b788'], width=0.45, edgecolor='white', linewidth=1.2)
    ax.set_title(f'Cost vs Revenue — {crop_name}', fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Amount (₹)', fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'₹{int(x):,}'))
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(cost, revenue) * 0.02,
                f'₹{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_facecolor('#f8fdf9')
    fig.patch.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def make_monthly_calendar_chart(season):
    """Generate a Gantt-style monthly activity chart. Returns base64 PNG."""
    season_map = {
        'Kharif': {
            'Land Prep': (6, 1), 'Sowing': (7, 1), 'Fertilizing': (7, 2),
            'Irrigation': (8, 2), 'Pest Control': (8, 2), 'Harvesting': (10, 1),
        },
        'Rabi': {
            'Land Prep': (10, 1), 'Sowing': (11, 1), 'Fertilizing': (11, 2),
            'Irrigation': (12, 2), 'Pest Control': (1, 2), 'Harvesting': (3, 1),
        },
        'Zaid': {
            'Land Prep': (3, 1), 'Sowing': (3, 1), 'Fertilizing': (4, 1),
            'Irrigation': (4, 2), 'Pest Control': (5, 1), 'Harvesting': (6, 1),
        },
    }
    key = next((k for k in season_map if k in season), 'Kharif')
    activities = season_map[key]
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    colors = ['#52b788','#f4a261','#2d6a4f','#74c69d','#d8f3dc','#b7e4c7']

    fig = Figure(figsize=(8, 3.5))
    ax = fig.subplots()
    for i, (activity, (start_month, duration)) in enumerate(activities.items()):
        # convert month to 0-indexed offset
        start = (start_month - 1) % 12
        ax.barh(i, duration, left=start, color=colors[i % len(colors)],
                edgecolor='white', linewidth=1, height=0.6)
        ax.text(start + duration / 2, i, activity, ha='center', va='center',
                fontsize=8, fontweight='bold', color='#1b1b2f')

    ax.set_yticks(range(len(activities)))
    ax.set_yticklabels(list(activities.keys()), fontsize=8)
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels, fontsize=8)
    ax.set_title(f'Monthly Activity Calendar — {key} Season', fontsize=10, fontweight='bold', pad=8)
    ax.set_facecolor('#f8fdf9')
    fig.patch.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def parse_number(value, default=None):
    """Parse '50,000', '₹ 50000', '2.5' -> float. Returns default if not a number."""
    try:
        cleaned = re.sub(r'[^\d.]', '', str(value or ''))
        return float(cleaned) if cleaned else default
    except ValueError:
        return default


def extract_primary_crop(plan_text):
    """Best-effort extraction of the recommended crop name from the LLM plan."""
    lines = [l.strip() for l in plan_text.splitlines()]
    for i, line in enumerate(lines):
        if re.search(r'(primary|recommended)\s+crop', line, re.IGNORECASE):
            candidates = [line.split(':', 1)[1]] if ':' in line else []
            candidates += lines[i + 1:i + 4]
            for cand in candidates:
                bold = re.search(r'\*\*([^*]{2,40})\*\*', cand)
                name = bold.group(1) if bold else cand
                name = re.split(r'[(\u2013\u2014\-.,]| - ', name)[0]
                name = re.sub(r'[*#_`>]', '', name).strip()
                if 2 <= len(name) <= 30 and not re.search(r'primary|recommended|crop', name, re.IGNORECASE):
                    return name
    return "Crop"


_SAFE_B64 = re.compile(r'^[A-Za-z0-9+/=\s]*$')


def sanitize_plan_html(fragment):
    """Strip anything from user-posted HTML that xhtml2pdf could use to fetch
    remote resources or run active content before it is turned into a PDF."""
    # 1) paired tags that carry active/remote content: remove including contents
    fragment = re.sub(r'(?is)<(script|style|iframe|object|svg|form|textarea)\b.*?</\1\s*>', '', fragment)
    # 2) any remaining/unpaired risky tags
    fragment = re.sub(r'(?is)</?(script|style|iframe|object|embed|svg|form|textarea|link|meta|img|input|base)\b[^>]*>', '', fragment)
    # 3) event handlers and URL-carrying attributes
    fragment = re.sub(r'(?i)\s(on\w+|src|href|style|srcset|background)\s*=\s*("[^"]*"|\'[^\']*\'|[^\s>]+)', '', fragment)
    return fragment


@app.route('/translate-hindi', methods=['POST'])
@login_required
def translate_hindi():
    payload = request.get_json(silent=True) or {}
    blocks = payload.get('blocks', [])
    if (not isinstance(blocks, list) or not blocks or len(blocks) > 10
            or not all(isinstance(b, str) and b.strip() and len(b) <= 8000 for b in blocks)):
        return jsonify({'error': 'No valid text provided'}), 400
    if not config.groq_api_key:
        return jsonify({'error': 'Translation is not configured'}), 503
    try:
        client = Groq(api_key=config.groq_api_key)
        translated = []
        for text_block in blocks:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Translate the following text to Hindi. "
                            "Keep the same formatting (numbered lists, bold etc). "
                            "Return only the translated text, nothing else.\n\n"
                            + text_block
                        ),
                    }
                ],
                model=config.GROQ_CHAT_MODEL,
            )
            result = chat_completion.choices[0].message.content
            translated.append(str(Markup(markdown.markdown(result))))
        return jsonify({'blocks': translated})
    except Exception as e:
        app.logger.error(f"Translation error: {e}")
        return jsonify({'error': 'Translation failed. Please try again.'}), 500


# render home page


@ app.route('/')
@login_required
def home():
    title = 'CropCare - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
@login_required
def crop_recommend():
    title = 'CropCare - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
@login_required
def fertilizer_recommendation():
    title = 'CropCare - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@app.route('/crop-predict', methods=['POST'])
@login_required
def crop_prediction():
    title = 'CropCare - Crop Recommendation'

    try:
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city", "").strip()

        weather = weather_fetch(city)
        if weather is None:
            return render_template('try_again.html', title=title)

        temperature, humidity = weather
        # The model was trained on a DataFrame, so pass named columns.
        data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=CROP_FEATURES)
        final_prediction = str(crop_recommendation_model.predict(data)[0])
        tips = get_crop_tips(final_prediction)
        if tips:
            tips = Markup(markdown.markdown(tips))
        return render_template('crop-result.html', prediction=final_prediction, tips=tips, title=title)
    except Exception as e:
        app.logger.error(f"Crop prediction error: {e}")
        return render_template('try_again.html', title=title)

# render fertilizer recommendation result page


@app.route('/fertilizer-predict', methods=['POST'])
@login_required
def fert_recommend():
    title = 'CropCare - Fertilizer Suggestion'

    try:
        crop_name = request.form['cropname'].strip().lower()
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['pottasium'])

        row = fertilizer_df[fertilizer_df['Crop'] == crop_name]
        if row.empty:
            app.logger.warning(f"Unknown crop in fertilizer form: {crop_name!r}")
            return render_template('try_again.html', title=title)

        ideal = row.iloc[0]
        # Difference between ideal and actual level for each nutrient. The
        # nutrient that is furthest off decides the advice (ties are handled
        # in N, P, K order).
        diffs = [('N', float(ideal['N']) - N),
                 ('P', float(ideal['P']) - P),
                 ('K', float(ideal['K']) - K)]
        nutrient, diff = max(diffs, key=lambda d: abs(d[1]))

        if diff == 0:
            response = Markup("Your soil's N, P and K levels already match the ideal levels for "
                              f"{html_lib.escape(crop_name)}. No fertilizer correction is needed.")
        else:
            key = f"{nutrient}High" if diff < 0 else f"{nutrient}low"
            response = Markup(str(fertilizer_dic[key]))

        return render_template('fertilizer-result.html', recommendation=response, title=title)
    except Exception as e:
        app.logger.error(f"Fertilizer recommendation error: {e}")
        return render_template('try_again.html', title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
@login_required
def disease_prediction():
    title = 'CropCare - Disease Detection'

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename:
            flash('Please choose a leaf image to upload.', 'warning')
            return render_template('disease.html', title=title)
        try:
            img = file.read()
            prediction, confidence = predict_image(img)
            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, confidence=confidence, title=title)
        except Exception as e:
            app.logger.error(f"Disease prediction error: {e}")
            flash('Could not analyse that image. Please upload a clear JPG or PNG photo of a leaf.', 'danger')
    return render_template('disease.html', title=title)


# ===============================================================================================
# MARKET PRICE INSIGHT

@app.route('/market-price')
@login_required
def market_price():
    title = 'CropCare - Market Price Insight'
    return render_template('market-price.html', title=title)


@app.route('/market-price-result', methods=['POST'])
@login_required
def market_price_result():
    title = 'CropCare - Market Price Insight'
    crop = request.form.get('crop', '').strip()
    state = request.form.get('state', '').strip()
    season = request.form.get('season', '').strip()

    if not crop:
        return render_template('try_again.html', title=title)

    try:
        # 1. Tavily search for real market data
        search_query = f"{crop} market price India {state} {season} mandi {datetime.now().year}"
        tavily_data = tavily_search(search_query, topic="general", max_results=4, include_images=True)
        search_context = ""
        search_sources = []
        search_images = []
        if tavily_data:
            if tavily_data.get('answer'):
                search_context = f"Recent web data: {tavily_data['answer']}\n\n"
            for r in tavily_data.get('results', []):
                search_context += f"- {r.get('title','')}: {r.get('content','')[:300]}\n"
                search_sources.append({'title': r.get('title',''), 'url': r.get('url','')})
            search_images = tavily_data.get('images', [])[:3]

        # 2. Groq LLM for structured insight (grounded with Tavily context)
        client = Groq(api_key=config.groq_api_key)
        prompt = (
            f"You are an agricultural market expert for India. "
            f"Using the following real-time web data as context:\n{search_context}\n"
            f"Provide a market price insight report for {crop} in {state if state else 'India'} "
            f"during {season if season else 'the current season'}.\n\n"
            "Include:\n"
            "1. Typical price range (₹ per quintal or kg)\n"
            "2. Best months to sell for maximum profit\n"
            "3. Key market factors affecting price\n"
            "4. Top 3 mandis/markets to sell in this region\n"
            "5. Price trend outlook (rising/stable/falling) with reason\n\n"
            "Format as a numbered list with clear headings. Be specific with numbers."
        )
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=config.GROQ_CHAT_MODEL,
        )
        result = chat_completion.choices[0].message.content
        insight = Markup(markdown.markdown(result))

        # 3. Groq vision: describe first relevant image if available
        vision_caption = None
        vision_image_url = None
        if search_images:
            img_url = search_images[0] if isinstance(search_images[0], str) else search_images[0].get('url','')
            if img_url:
                vision_image_url = img_url
                vision_caption = groq_vision_describe(
                    img_url,
                    f"This image is related to {crop} farming or market in India. "
                    "Briefly describe what you see in 1-2 sentences, focusing on agricultural relevance."
                )

        return render_template('market-price-result.html', crop=crop, state=state,
                               season=season, insight=insight, sources=search_sources,
                               vision_image_url=vision_image_url, vision_caption=vision_caption,
                               title=title)
    except Exception as e:
        app.logger.error(f"Market price error: {e}")
        return render_template('try_again.html', title=title)


# ===============================================================================================
# SMART CROP PLANNING

@app.route('/crop-planning')
@login_required
def crop_planning():
    title = 'CropCare - Smart Crop Planning'
    return render_template('crop-planning.html', title=title)


@app.route('/crop-planning-result', methods=['POST'])
@login_required
def crop_planning_result():
    title = 'CropCare - Smart Crop Planning'
    land_size = request.form.get('land_size', '').strip()
    land_unit = request.form.get('land_unit', 'acres')
    budget = request.form.get('budget', '').strip()
    state = request.form.get('state', '').strip()
    season = request.form.get('season', '').strip()
    soil_type = request.form.get('soil_type', '').strip()
    water_source = request.form.get('water_source', '').strip()

    land_value = parse_number(land_size)
    if not land_value or land_value <= 0 or not season:
        return render_template('try_again.html', title=title)
    budget_value = parse_number(budget)

    try:
        # 1. Tavily search for regional crop data
        search_query = f"best crops to grow {state if state else 'India'} {season} {soil_type} soil farming {datetime.now().year}"
        tavily_data = tavily_search(search_query, topic="general", max_results=4, include_images=True)
        search_context = ""
        search_sources = []
        search_images = []
        if tavily_data:
            if tavily_data.get('answer'):
                search_context = f"Recent web data: {tavily_data['answer']}\n\n"
            for r in tavily_data.get('results', []):
                search_context += f"- {r.get('title','')}: {r.get('content','')[:300]}\n"
                search_sources.append({'title': r.get('title',''), 'url': r.get('url','')})
            search_images = tavily_data.get('images', [])[:3]

        # 2. Groq LLM for the full plan
        client = Groq(api_key=config.groq_api_key)
        prompt = (
            f"You are an expert agricultural planner for India. "
            f"Using the following real-time web data as context:\n{search_context}\n"
            f"Create a smart crop plan for a farmer with the following details:\n"
            f"- Land: {land_size} {land_unit}\n"
            f"- Budget: ₹{budget if budget else 'moderate'}\n"
            f"- Location: {state if state else 'India'}\n"
            f"- Season: {season}\n"
            f"- Soil type: {soil_type if soil_type else 'not specified'}\n"
            f"- Water source: {water_source if water_source else 'not specified'}\n\n"
            "Provide:\n"
            "1. Recommended primary crop with justification\n"
            "2. Intercrop or secondary crop suggestion\n"
            "3. Month-by-month activity calendar (sowing to harvest)\n"
            "4. Estimated input cost (₹) and expected revenue (₹) — give specific numbers\n"
            "5. Crop rotation advice for next season\n"
            "6. Key risks and mitigation tips\n\n"
            "IMPORTANT: In section 4, always include two lines formatted exactly like:\n"
            "ESTIMATED_COST: <number>\n"
            "ESTIMATED_REVENUE: <number>\n"
            "Format the rest clearly with numbered sections and sub-points."
        )
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=config.GROQ_CHAT_MODEL,
        )
        raw_plan = chat_completion.choices[0].message.content

        # 3. Extract cost/revenue for chart
        cost_match = re.search(r'ESTIMATED_COST:\W*([\d][\d,]*)', raw_plan)
        revenue_match = re.search(r'ESTIMATED_REVENUE:\W*([\d][\d,]*)', raw_plan)
        cost_val = int(parse_number(cost_match.group(1), 0)) if cost_match else 0
        cost_val = cost_val or int(budget_value or 50000)
        revenue_val = int(parse_number(revenue_match.group(1), 0)) if revenue_match else 0
        revenue_val = revenue_val or int(cost_val * 1.6)
        # Strip the machine-readable lines from display
        display_plan = re.sub(r'(?m)^.*ESTIMATED_(COST|REVENUE).*\n?', '', raw_plan)
        plan = Markup(markdown.markdown(display_plan))

        # 4. Generate charts
        crop_name = extract_primary_crop(raw_plan)
        chart_cost_revenue = make_cost_revenue_chart(cost_val, revenue_val, crop_name)
        chart_calendar = make_monthly_calendar_chart(season)

        # 5. Groq vision on first search image
        vision_caption = None
        vision_image_url = None
        if search_images:
            img_url = search_images[0] if isinstance(search_images[0], str) else search_images[0].get('url','')
            if img_url:
                vision_image_url = img_url
                vision_caption = groq_vision_describe(
                    img_url,
                    f"This image is related to farming in {state if state else 'India'} during {season}. "
                    "Briefly describe what you see in 1-2 sentences, focusing on agricultural relevance."
                )

        return render_template('crop-planning-result.html', plan=plan, land_size=land_size,
                               land_unit=land_unit, state=state, season=season,
                               chart_cost_revenue=chart_cost_revenue,
                               chart_calendar=chart_calendar,
                               sources=search_sources,
                               vision_image_url=vision_image_url,
                               vision_caption=vision_caption,
                               title=title)
    except Exception as e:
        app.logger.error(f"Crop planning error: {e}")
        return render_template('try_again.html', title=title)


@app.route('/crop-planning-pdf', methods=['POST'])
@login_required
def crop_planning_pdf():
    """Render the crop plan as a downloadable PDF using xhtml2pdf."""
    try:
        from xhtml2pdf import pisa
        plan_html = request.form.get('plan_html', '')
        plan_html = sanitize_plan_html(plan_html)
        chart_cost_revenue = request.form.get('chart_cost_revenue', '')
        chart_calendar = request.form.get('chart_calendar', '')
        if not _SAFE_B64.match(chart_cost_revenue):
            chart_cost_revenue = ''
        if not _SAFE_B64.match(chart_calendar):
            chart_calendar = ''
        land_size = html_lib.escape(request.form.get('land_size', ''))
        land_unit = html_lib.escape(request.form.get('land_unit', ''))
        state = html_lib.escape(request.form.get('state', ''))
        season = html_lib.escape(request.form.get('season', ''))

        chart_cal_tag = f'<img src="data:image/png;base64,{chart_calendar}" style="width:100%;"/>' if chart_calendar else ''
        chart_cr_tag = f'<img src="data:image/png;base64,{chart_cost_revenue}" style="width:60%;"/>' if chart_cost_revenue else ''

        pdf_html = f"""<!DOCTYPE html><html><head>
        <meta charset="utf-8">
        <style>
          @page {{ margin: 2cm; }}
          body {{ font-family: Helvetica, Arial, sans-serif; color: #2d3436; line-height: 1.6; font-size: 11pt; }}
          h1 {{ color: #2d6a4f; font-size: 16pt; border-bottom: 1px solid #52b788; padding-bottom: 4pt; }}
          h2, h3, h4 {{ color: #2d6a4f; }}
          .meta {{ color: #636e72; font-size: 9pt; margin-bottom: 12pt; }}
          img {{ max-width: 100%; margin: 8pt 0; }}
          ul, ol {{ padding-left: 16pt; }}
          li {{ margin-bottom: 3pt; }}
          .footer {{ margin-top: 16pt; font-size: 8pt; color: #636e72; border-top: 1px solid #e0e0e0; padding-top: 4pt; }}
        </style></head><body>
        <h1>Smart Crop Plan - CropCare</h1>
        <div class="meta">{land_size} {land_unit} | {state} | {season}</div>
        {chart_cal_tag}
        {chart_cr_tag}
        {plan_html}
        <div class="footer">Generated by CropCare | AI-powered agricultural planning</div>
        </body></html>"""

        pdf_buf = io.BytesIO()
        pisa_status = pisa.CreatePDF(pdf_html, dest=pdf_buf)
        if pisa_status.err:
            raise Exception(f"pisa error: {pisa_status.err}")
        pdf_buf.seek(0)
        return Response(
            pdf_buf.read(),
            mimetype='application/pdf',
            headers={'Content-Disposition': 'attachment; filename=crop_plan.pdf'}
        )
    except Exception as e:
        app.logger.error(f"PDF generation error: {e}")
        return "PDF generation failed. Please go back and try again.", 500


# ===============================================================================================
# DISEASE TIMELINE TRACKING

@app.route('/disease-tracking')
@login_required
def disease_tracking():
    """Render disease tracking page with list of tracked plants"""
    title = 'CropCare - Disease Timeline Tracking'
    plants_data = list_plants(current_user.oid)
    today = datetime.now().strftime('%Y-%m-%d')
    return render_template('disease-tracking.html', title=title, plants=plants_data, today=today)


ALLOWED_IMAGE_FORMATS = {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp', 'BMP': 'bmp'}


def remove_upload(filename):
    """Delete an uploaded image from disk (ignores missing files)."""
    if not filename:
        return
    path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(filename))
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        app.logger.warning(f"Could not delete {path}: {e}")


@app.route('/disease-tracking/upload', methods=['POST'])
@login_required
def disease_tracking_upload():
    """Handle disease tracking image upload and analysis"""
    title = 'CropCare - Disease Timeline Tracking'
    saved_filename = None

    try:
        plant_selection = request.form.get('plant_selection')
        file = request.files.get('file')
        notes = request.form.get('notes', '')[:1000]
        observation_date_str = request.form.get('observation_date')

        if not file or not file.filename or not observation_date_str or not plant_selection:
            return render_template('try_again.html', title=title)

        try:
            observation_date = datetime.strptime(observation_date_str, '%Y-%m-%d')
        except ValueError:
            return render_template('try_again.html', title=title)

        img_bytes = file.read()

        # Make sure it really is a supported image (not just a renamed file)
        with Image.open(io.BytesIO(img_bytes)) as probe:
            extension = ALLOWED_IMAGE_FORMATS.get(probe.format)
        if not extension:
            flash('Unsupported image type. Please upload a JPG, PNG, WEBP or BMP image.', 'danger')
            return redirect(url_for('disease_tracking'))

        prediction, confidence = predict_image(img_bytes)
        severity_score = calculate_severity_score(prediction, confidence)
        affected_area = estimate_affected_area(img_bytes)

        new_plant_created = False
        if plant_selection == 'new':
            plant_name = request.form.get('plant_name', '').strip()
            crop_type = request.form.get('crop_type', '').strip()
            location = request.form.get('location', '').strip()

            if not plant_name or not crop_type:
                return render_template('try_again.html', title=title)

            plant = create_plant(current_user.oid, plant_name, crop_type, location)
            new_plant_created = True
        else:
            plant = get_plant(plant_selection, current_user.oid)
            if not plant:
                return render_template('try_again.html', title=title)

        # Unique, safe file name (never trust the uploaded name)
        saved_filename = f"{plant['plant_id']}_{observation_date.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}.{extension}"
        with open(os.path.join(app.config['UPLOAD_FOLDER'], saved_filename), 'wb') as f:
            f.write(img_bytes)

        try:
            add_record(
                plant, saved_filename, prediction, confidence,
                severity_score, affected_area, observation_date, notes,
            )
        except Exception:
            # Don't leave an empty plant behind if saving the record failed
            if new_plant_created:
                delete_plant(plant)
            raise

        return redirect(url_for('disease_tracking_view', plant_id=plant['plant_id']))

    except Exception as e:
        app.logger.error(f"Disease tracking upload error: {e}")
        remove_upload(saved_filename)   # don't leave orphan files behind
        return render_template('try_again.html', title=title)


@app.route('/disease-tracking/view/<plant_id>')
@login_required
def disease_tracking_view(plant_id):
    """View disease progression timeline for a specific plant"""
    title = 'CropCare - Disease Timeline'
    
    plant = get_plant(plant_id, current_user.oid)
    if not plant:
        return render_template('try_again.html', title=title)
    
    # All disease records for this plant (newest first)
    records = list_records(plant)
    
    # Format records for display
    records_data = []
    for record in records:
        record_dict = record_to_dict(record)
        record_dict['formatted_date'] = record['timestamp'].strftime('%b %d, %Y')
        record_dict['severity_color'] = get_severity_color(record['severity_score'])
        records_data.append(record_dict)
    
    # Analyze progression
    analysis = None
    if len(records) >= 2:
        analysis = analyze_progression(records_data)
    
    # Generate charts
    chart_severity = None
    chart_confidence = None
    
    if len(records) >= 2:
        chart_severity = generate_severity_chart(records)
        chart_confidence = generate_confidence_chart(records)
    
    return render_template(
        'disease-tracking-view.html',
        title=title,
        plant=plant_to_dict(plant, len(records)),
        records=records_data,
        analysis=analysis,
        chart_severity=chart_severity,
        chart_confidence=chart_confidence
    )


def generate_severity_chart(records):
    """Generate severity progression chart"""
    try:
        sorted_records = sorted(records, key=lambda x: x['timestamp'])
        dates = [r['timestamp'].strftime('%m/%d') for r in sorted_records]
        severities = [r['severity_score'] for r in sorted_records]
        colors = [get_severity_color(s) for s in severities]
        
        fig = Figure(figsize=(8, 4))
        ax = fig.subplots()
        ax.plot(range(len(dates)), severities, marker='o', linewidth=2.5, markersize=8, color='#2d6a4f')
        
        # Color the area under the curve
        ax.fill_between(range(len(dates)), severities, alpha=0.3, color='#52b788')
        
        # Add severity threshold lines
        ax.axhline(y=20, color='#95d5b2', linestyle='--', linewidth=1, alpha=0.5, label='Mild')
        ax.axhline(y=60, color='#f4a261', linestyle='--', linewidth=1, alpha=0.5, label='Moderate')
        ax.axhline(y=80, color='#e63946', linestyle='--', linewidth=1, alpha=0.5, label='Severe')
        
        ax.set_xlabel('Date', fontsize=10, fontweight='bold')
        ax.set_ylabel('Severity Score', fontsize=10, fontweight='bold')
        ax.set_title('Disease Severity Progression', fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 100)
        
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45, ha='right')
        fig.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        app.logger.error(f"Chart generation error: {e}")
        return None


def generate_confidence_chart(records):
    """Generate confidence score chart"""
    try:
        sorted_records = sorted(records, key=lambda x: x['timestamp'])
        dates = [r['timestamp'].strftime('%m/%d') for r in sorted_records]
        confidences = [r['confidence'] for r in sorted_records]
        
        fig = Figure(figsize=(8, 4))
        ax = fig.subplots()
        ax.plot(range(len(dates)), confidences, marker='s', linewidth=2.5, markersize=8, color='#f4a261')
        ax.fill_between(range(len(dates)), confidences, alpha=0.3, color='#f4a261')
        
        ax.set_xlabel('Date', fontsize=10, fontweight='bold')
        ax.set_ylabel('Confidence (%)', fontsize=10, fontweight='bold')
        ax.set_title('Detection Confidence Over Time', fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 100)
        
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels(dates, rotation=45, ha='right')
        fig.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        app.logger.error(f"Chart generation error: {e}")
        return None


@app.route('/disease-tracking/delete-record/<record_id>', methods=['POST'])
@login_required
def delete_disease_record(record_id):
    """Delete a specific disease record"""
    try:
        record = get_record(record_id)
        # Same answer for "not found" and "not yours" so ids can't be probed
        if not record or record['user_id'] != current_user.oid:
            return jsonify({'success': False, 'message': 'Record not found'}), 404

        delete_record(record['_id'])
        remove_upload(record['image_filename'])   # only after the DB change succeeded

        return jsonify({'success': True, 'message': 'Record deleted successfully'})
    except Exception as e:
        app.logger.error(f"Delete record error: {e}")
        return jsonify({'success': False, 'message': 'Could not delete the record'}), 500


@app.route('/disease-tracking/delete-plant/<plant_id>', methods=['POST'])
@login_required
def delete_plant_route(plant_id):
    """Delete an entire plant and all its records"""
    try:
        plant = get_plant(plant_id, current_user.oid)
        if not plant:
            return jsonify({'success': False, 'message': 'Plant not found'}), 404

        filenames = delete_plant(plant)   # removes the plant and all its records
        for name in filenames:
            remove_upload(name)

        return jsonify({'success': True, 'message': 'Plant deleted successfully'})
    except Exception as e:
        app.logger.error(f"Delete plant error: {e}")
        return jsonify({'success': False, 'message': 'Could not delete the plant'}), 500


# ===============================================================================================
if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=int(os.getenv('PORT', 8000)))
