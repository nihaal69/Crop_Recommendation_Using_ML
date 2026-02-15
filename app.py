from flask import Flask, request, render_template, url_for
import numpy as np
import pickle
import os
from typing import Optional, Dict, Any

app = Flask(__name__)

# Constants
MODEL_PATH = 'model.pkl'
SCALER_STAND_PATH = 'standscaler.pkl'
SCALER_MINMAX_PATH = 'minmaxscaler.pkl'
STATIC_IMAGE_FOLDER = os.path.join('static', 'images')

# Crop dictionary with numerical image names
CROP_DICT = {
    1: {"name": "Rice", "image": "1.jpg"},
    2: {"name": "Maize", "image": "2.jpg"},
    3: {"name": "Jute", "image": "3.jpg"},
    4: {"name": "Cotton", "image": "4.jpg"},
    5: {"name": "Coconut", "image": "5.jpg"},
    6: {"name": "Papaya", "image": "6.jpg"},
    7: {"name": "Orange", "image": "7.jpg"},
    8: {"name": "Apple", "image": "8.jpg"},
    9: {"name": "Muskmelon", "image": "9.jpg"},
    10: {"name": "Watermelon", "image": "10.jpg"},
    11: {"name": "Grapes", "image": "11.jpg"},
    12: {"name": "Mango", "image": "12.jpg"},
    13: {"name": "Banana", "image": "13.jpg"},
    14: {"name": "Pomegranate", "image": "14.jpg"},
    15: {"name": "Lentil", "image": "15.jpg"},
    16: {"name": "Blackgram", "image": "16.jpg"},
    17: {"name": "Mungbean", "image": "17.jpg"},
    18: {"name": "Mothbeans", "image": "18.jpg"},
    19: {"name": "Pigeonpeas", "image": "19.jpg"},
    20: {"name": "Kidneybeans", "image": "20.jpg"},
    21: {"name": "Chickpea", "image": "21.jpg"},
    22: {"name": "Coffee", "image": "22.jpg"}
}

# Complete growing guide resources for all 22 crops
GROWING_GUIDES = {
    "rice": {
        "url": "https://www.knowledgebank.irri.org/rice-production-management",
        "description": "Complete guide to rice cultivation from planting to harvest"
    },
    "maize": {
        "url": "https://www.pioneer.com/us/agronomy/growing_maize.html",
        "description": "Maize growing techniques and best practices"
    },
    "jute": {
        "url": "https://www.cabi.org/isc/datasheet/29772",
        "description": "Jute cultivation and fiber production guide"
    },
    "cotton": {
        "url": "https://www.cotton.org/tech/ace/growing/index.cfm",
        "description": "Cotton growing and production techniques"
    },
    "coconut": {
        "url": "https://www.fao.org/3/y4622e/y4622e04.htm",
        "description": "Coconut palm management and cultivation"
    },
    "papaya": {
        "url": "https://www.ctahr.hawaii.edu/oc/freepubs/pdf/F_N-3.pdf",
        "description": "Papaya production guide"
    },
    "orange": {
        "url": "https://citrusindustry.net/category/production/",
        "description": "Orange and citrus production resources"
    },
    "apple": {
        "url": "https://extension.umn.edu/fruit/growing-apples",
        "description": "Apple orchard management guide"
    },
    "muskmelon": {
        "url": "https://extension.umn.edu/vegetables/growing-melons",
        "description": "Muskmelon and cantaloupe growing guide"
    },
    "watermelon": {
        "url": "https://www.almanac.com/plant/watermelons",
        "description": "Watermelon cultivation tips"
    },
    "grapes": {
        "url": "https://extension.umn.edu/fruit/growing-grapes-home-garden",
        "description": "Grape vine management and production"
    },
    "mango": {
        "url": "https://www.agric.wa.gov.au/mangoes/mango-growing-guide",
        "description": "Mango cultivation best practices"
    },
    "banana": {
        "url": "https://www.agric.wa.gov.au/bananas/banana-growing-western-australia",
        "description": "Banana plantation management"
    },
    "pomegranate": {
        "url": "https://www.fao.org/3/ca3556en/ca3556en.pdf",
        "description": "Pomegranate production manual"
    },
    "lentil": {
        "url": "https://www.pulses.org/resources/growing-guides/lentil",
        "description": "Lentil cultivation guide"
    },
    "blackgram": {
        "url": "https://www.ikisan.com/tn-blackgram-crop-production.html",
        "description": "Black gram cultivation practices"
    },
    "mungbean": {
        "url": "https://www.ikisan.com/tn-green-gram-crop-production.html",
        "description": "Mung bean production techniques"
    },
    "mothbeans": {
        "url": "https://www.ikisan.com/ap-moth-bean-crop-production.html",
        "description": "Moth bean cultivation guide"
    },
    "pigeonpeas": {
        "url": "https://www.ikisan.com/tn-red-gram-crop-production.html",
        "description": "Pigeon pea farming methods"
    },
    "kidneybeans": {
        "url": "https://www.almanac.com/plant/kidney-beans",
        "description": "Kidney bean growing guide"
    },
    "chickpea": {
        "url": "https://www.pulses.org/resources/growing-guides/chickpea",
        "description": "Chickpea cultivation techniques"
    },
    "coffee": {
        "url": "https://www.ncausa.org/about-coffee/coffee-growing",
        "description": "Coffee cultivation and processing"
    },
    "default": {
        "url": "https://www.fao.org/agriculture/crops/core-themes/theme/spi/plant-production-standards/en/",
        "description": "General agricultural best practices"
    }
}

# Complete seed suppliers for all 22 crops
SEED_SUPPLIERS = {
    "rice": [
        {"name": "Rice Seed Co", "url": "https://www.riceseed.com"},
        {"name": "AgriSeeds", "url": "https://www.agriseeds.com/rice"}
    ],
    "maize": [
        {"name": "Maize Seed Specialists", "url": "https://www.maizeseeds.com"},
        {"name": "CropKing", "url": "https://www.cropking.com/maize"}
    ],
    "jute": [
        {"name": "FiberCrop Seeds", "url": "https://www.fibercrop.com/jute"},
        {"name": "AgroFiber", "url": "https://www.agrofiber.com"}
    ],
    "cotton": [
        {"name": "Cotton Seed Distributors", "url": "https://www.cottonseed.com"},
        {"name": "DeltaPine", "url": "https://www.deltapine.com"}
    ],
    "coconut": [
        {"name": "Tropical Seeds", "url": "https://www.tropicalseeds.com/coconut"},
        {"name": "PalmTree Nursery", "url": "https://www.palmtree.com"}
    ],
    "papaya": [
        {"name": "Tropical Fruit Seeds", "url": "https://www.tropicalfruitseeds.com/papaya"},
        {"name": "Rainforest Seeds", "url": "https://www.rainforestseeds.com"}
    ],
    "orange": [
        {"name": "Citrus Seed Co", "url": "https://www.citrusseed.com"},
        {"name": "FruitTree Nursery", "url": "https://www.fruittree.com/orange"}
    ],
    "apple": [
        {"name": "AppleSeeds", "url": "https://www.appleseeds.com"},
        {"name": "Orchard Supply", "url": "https://www.orchard.com/apple"}
    ],
    "muskmelon": [
        {"name": "Melon Seed Co", "url": "https://www.melonseed.com"},
        {"name": "Garden Seeds", "url": "https://www.gardenseeds.com/muskmelon"}
    ],
    "watermelon": [
        {"name": "Watermelon Seeds Inc", "url": "https://www.watermelonseeds.com"},
        {"name": "Summer Fruits Seeds", "url": "https://www.summerfruits.com"}
    ],
    "grapes": [
        {"name": "GrapeVine Seeds", "url": "https://www.grapeseeds.com"},
        {"name": "Vineyard Supply", "url": "https://www.vineyard.com/seeds"}
    ],
    "mango": [
        {"name": "Tropical Fruit Seeds", "url": "https://www.tropicalfruitseeds.com/mango"},
        {"name": "Mango Seed Co", "url": "https://www.mangoseed.com"}
    ],
    "banana": [
        {"name": "Banana Plant Co", "url": "https://www.bananaplant.com"},
        {"name": "Tropical Plants", "url": "https://www.tropicalplants.com/banana"}
    ],
    "pomegranate": [
        {"name": "FruitTree Nursery", "url": "https://www.fruittree.com/pomegranate"},
        {"name": "Mediterranean Seeds", "url": "https://www.medseeds.com/pomegranate"}
    ],
    "lentil": [
        {"name": "Pulse Seeds", "url": "https://www.pulseseeds.com/lentil"},
        {"name": "Legume Seed Co", "url": "https://www.legumeseed.com"}
    ],
    "blackgram": [
        {"name": "Indian Pulse Seeds", "url": "https://www.indianpulse.com/blackgram"},
        {"name": "Asia Seeds", "url": "https://www.asiaseeds.com"}
    ],
    "mungbean": [
        {"name": "Asian Pulse Co", "url": "https://www.asianpulse.com/mungbean"},
        {"name": "Green Seed Co", "url": "https://www.greenseed.com"}
    ],
    "mothbeans": [
        {"name": "Desert Seeds", "url": "https://www.desertseeds.com/mothbeans"},
        {"name": "Arid Crop Seeds", "url": "https://www.aridcrop.com"}
    ],
    "pigeonpeas": [
        {"name": "Tropical Pulse", "url": "https://www.tropicalpulse.com/pigeonpea"},
        {"name": "Legume Specialists", "url": "https://www.legumespecial.com"}
    ],
    "kidneybeans": [
        {"name": "Bean Seed Co", "url": "https://www.beanseed.com/kidney"},
        {"name": "Pulse Seeds", "url": "https://www.pulseseeds.com/kidneybeans"}
    ],
    "chickpea": [
        {"name": "Mediterranean Seeds", "url": "https://www.medseeds.com/chickpea"},
        {"name": "Pulse Seeds", "url": "https://www.pulseseeds.com/chickpea"}
    ],
    "coffee": [
        {"name": "Coffee Plant Co", "url": "https://www.coffeeplant.com"},
        {"name": "Tropical Seeds", "url": "https://www.tropicalseeds.com/coffee"}
    ],
    "default": [
        {"name": "Global Seed Suppliers", "url": "https://www.worldseed.org"},
        {"name": "Agri Marketplace", "url": "https://www.agrimarketplace.com"}
    ]
}

# Load model and scalers with error handling
def load_models():
    """Load ML models and scalers with proper error handling."""
    try:
        model = pickle.load(open(MODEL_PATH, 'rb'))
        sc = pickle.load(open(SCALER_STAND_PATH, 'rb'))
        ms = pickle.load(open(SCALER_MINMAX_PATH, 'rb'))
        return model, sc, ms
    except FileNotFoundError as e:
        raise Exception(f"Model file not found: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading models: {str(e)}")


try:
    model, sc, ms = load_models()
except Exception as e:
    print(f"Critical error during startup: {str(e)}")
    # In a production environment, you might want to exit here
    # import sys; sys.exit(1)


# Helper functions
def validate_input(data: Dict[str, str]) -> Optional[np.ndarray]:
    """Validate and convert form input to numpy array."""
    try:
        feature_list = [
            float(data.get('Nitrogen', 0)),
            float(data.get('Phosporus', 0)),
            float(data.get('Potassium', 0)),
            float(data.get('Temperature', 0)),
            float(data.get('Humidity', 0)),
            float(data.get('Ph', 0)),
            float(data.get('Rainfall', 0))
        ]
        return np.array(feature_list).reshape(1, -1)
    except (ValueError, TypeError):
        return None


def get_crop_info(prediction: int) -> Dict[str, Any]:
    """Get crop information based on prediction."""
    crop_info = CROP_DICT.get(prediction, {"name": "Unknown", "image": None})

    # Check if image actually exists
    if crop_info["image"]:
        image_path = os.path.join(STATIC_IMAGE_FOLDER, crop_info["image"])
        if not os.path.exists(image_path):
            app.logger.warning(f"Image not found: {image_path}")
            crop_info["image"] = None
        else:
            # Generate URL for the image
            crop_info["image_url"] = url_for('static', filename=f'images/{crop_info["image"]}')

    return crop_info


# Define routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template("index.html", result=None, crop_info=None)


@app.route("/predict", methods=['POST'])
def predict():
    """Handle prediction requests."""
    # Validate input
    features = validate_input(request.form)
    if features is None:
        return render_template('index.html',
                               error="Invalid input. Please enter numeric values for all fields.",
                               result=None,
                               crop_info=None)


    try:
        # Scale features
        scaled_features = ms.transform(features)
        final_features = sc.transform(scaled_features)

        # Make prediction
        prediction = model.predict(final_features)[0]
        crop_info = get_crop_info(prediction)

        if crop_info["name"] == "Unknown":
            result = "Sorry, we could not determine the best crop for the provided data."
        else:
            result = f"{crop_info['name']} is the best crop for your conditions"

        return render_template('index.html',
                               result=result,
                               crop_info=crop_info,
                               error=None)

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html',
                               error="An error occurred during prediction. Please try again.",
                               result=None,
                               crop_info=None)
@app.route('/growing-guide/<crop_name>')
def growing_guide(crop_name):
    guide = GROWING_GUIDES.get(crop_name.lower(), GROWING_GUIDES["default"])
    return render_template('guide.html', crop_name=crop_name, guide_url=guide["url"], description=guide["description"])

@app.route('/buy-seeds/<crop_name>')
def buy_seeds(crop_name):
    suppliers = SEED_SUPPLIERS.get(crop_name.lower(), SEED_SUPPLIERS["default"])
    return render_template('seeds.html', crop_name=crop_name, suppliers=suppliers)
@app.route('/guide_us')
def guide_us():
    return render_template('guide_us.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)