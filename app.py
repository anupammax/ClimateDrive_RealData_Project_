import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="ClimateDrive | Vehicle Intelligence", page_icon="🚘", layout="wide", initial_sidebar_state="expanded")

# Premium dashboard styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#edf4ff; --muted:#9baec8; --panel:#101d31; --line:rgba(161,188,224,.14); --cyan:#67e8f9; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 78% 0%, #183653 0%, #0b1423 40%, #080f1b 100%); color:var(--ink); }
[data-testid="stHeader"] {background:rgba(8,15,27,.78)}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#0d1b2d,#0a1321); border-right:1px solid var(--line)}
[data-testid="stSidebar"] * {color:#eaf2ff}
.block-container {padding-top:1.7rem; padding-bottom:3rem; max-width:1450px}
h1,h2,h3 {font-family:'Manrope',sans-serif; letter-spacing:-.035em; color:#f3f7ff}
.hero {padding:28px 30px; border:1px solid rgba(135,190,232,.22); border-radius:24px; background:linear-gradient(115deg,rgba(25,57,84,.94),rgba(16,29,49,.84) 58%,rgba(21,48,68,.75)); position:relative; overflow:hidden; margin-bottom:20px;}
.hero:after {content:'◉'; position:absolute; right:38px; top:-70px; font-size:270px; color:rgba(103,232,249,.055);}
.eyebrow {color:#76e7f5; text-transform:uppercase; letter-spacing:.18em; font-size:11px; font-weight:800;}
.hero h1 {font-size:clamp(30px,4vw,48px); line-height:1.08; margin:10px 0 12px;}
.hero p {color:#b4c7dd; max-width:690px; font-size:15px; line-height:1.65; margin:0;}
.pill {display:inline-block;border:1px solid rgba(103,232,249,.28); background:rgba(103,232,249,.08); color:#9ff3fb; border-radius:100px;padding:6px 10px;font-size:11px;font-weight:700;margin-right:6px;}
.metric {background:linear-gradient(145deg,rgba(22,39,62,.95),rgba(13,25,42,.95)); border:1px solid var(--line); border-radius:17px; padding:17px 18px; min-height:112px;}
.metric-label {color:#9db1ca;font-size:12px;font-weight:600;letter-spacing:.04em;text-transform:uppercase}
.metric-value {font-family:'Manrope';font-size:28px;font-weight:800;color:#f4f8ff;margin:7px 0 2px;}
.metric-sub {font-size:11px;color:#829bb9;}
.section-kicker {color:#78e7f3;font-size:11px;letter-spacing:.16em;text-transform:uppercase;font-weight:800;margin-bottom:4px;}
.vehicle-card {border:1px solid rgba(155,188,222,.17);border-radius:19px;overflow:hidden;background:linear-gradient(155deg,rgba(21,39,61,.96),rgba(12,23,39,.97));height:100%;}
.vehicle-art {height:130px;display:flex;align-items:center;justify-content:center;background:radial-gradient(ellipse at 50% 85%,rgba(103,232,249,.18),transparent 52%),linear-gradient(135deg,#1a3853,#102033);font-size:72px;position:relative;}
.vehicle-art small {position:absolute;top:13px;left:14px;font-size:10px;letter-spacing:.1em;text-transform:uppercase;background:rgba(5,13,23,.5);border:1px solid rgba(255,255,255,.12);padding:5px 8px;border-radius:20px;color:#cce5f7;}
.vehicle-body {padding:17px;}
.vehicle-name {font-family:'Manrope';font-size:19px;font-weight:800;color:#f3f7ff;margin:2px 0 7px;}
.vehicle-price {font-size:21px;font-weight:800;color:#75e6f3;}
.vehicle-meta {color:#9db1ca;font-size:12px;margin:3px 0 12px;}
.vehicle-reason {color:#c1d0e2;font-size:13px;line-height:1.55;min-height:60px;}
.tag {display:inline-block;font-size:10px;color:#bceef3;border:1px solid rgba(103,232,249,.2);background:rgba(103,232,249,.06);border-radius:8px;padding:5px 8px;margin:8px 5px 0 0;}
.note {border:1px solid rgba(103,232,249,.19);background:rgba(29,76,92,.19);border-radius:13px;padding:13px 15px;color:#b9d7e5;font-size:13px;line-height:1.55;}
hr {border-color:var(--line)!important;}
.stButton>button {border-radius:12px;border:1px solid rgba(103,232,249,.4);background:linear-gradient(100deg,#67e8f9,#8eeaf2);color:#082033;font-weight:800;padding:.65rem 1.2rem;}
.stButton>button:hover {border-color:#fff;color:#061421;}
[data-testid="stMetric"] {background:transparent}
label {font-weight:600!important}
footer {visibility:hidden}
</style>
""", unsafe_allow_html=True)

DATA_PATH = "climate_vehicle_dataset.csv"
FEATURES = ["temperature_c", "rainfall_mm", "humidity_pct", "wind_kmh", "road_condition", "terrain", "passengers", "priority"]

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model():
    df = load_data()
    X = df[FEATURES].copy()
    encoders = {}
    for col in ["road_condition", "terrain", "priority"]:
        encoders[col] = LabelEncoder()
        X[col] = encoders[col].fit_transform(X[col].astype(str))
    target = LabelEncoder()
    y = target.fit_transform(df["recommended_vehicle"].astype(str))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced", min_samples_leaf=2)
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    return model, encoders, target, score

def safe_encode(value, encoder):
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return 0

def recommend_exact_models(rainfall, road, terrain, passengers, priority, budget_lakh, ev_charging):
    # This transparent shortlist is rule-based, not generated by the ML classifier.
    if road == "icy" or (road == "wet" and rainfall >= 100):
        return [{"name":"Pause & reassess the trip", "price":None, "fuel":"Safety first", "type":"Travel advisory", "why":"Icy, flooded or extremely wet roads can be dangerous in any vehicle. Check local advisories and postpone if conditions are unsafe.", "icon":"⚠️"}], "Safety advisory — a vehicle recommendation cannot make unsafe road conditions safe."
    if passengers >= 6:
        options = [
            ("Maruti Suzuki Ertiga", 9.0, "Petrol / CNG", "7-seat MPV", "A practical family option; check luggage capacity when all seats are occupied.", "🚐"),
            ("Toyota Rumion", 10.0, "Petrol / CNG", "7-seat MPV", "Another 7-seat option; compare variant features and service access nearby.", "🚐"),
            ("Kia Carens Clavis", 12.0, "Petrol / Diesel", "Family MPV", "Consider for family trips; confirm exact variant, seating and current price.", "🚙")]
    elif road == "rough" or terrain in ("hills", "rural"):
        options = [
            ("Maruti Suzuki Brezza", 9.0, "Petrol / CNG", "Compact SUV", "Higher seating and useful ground clearance for moderate uneven roads; not an off-road guarantee.", "🚙"),
            ("Hyundai Venue", 8.0, "Petrol / Diesel", "Compact SUV", "A compact option for mixed city and rural driving; compare tyre and safety specs by variant.", "🚙"),
            ("Mahindra Thar 4WD", 15.0, "Petrol / Diesel", "4WD SUV", "Only worth considering if terrain genuinely requires 4WD; less convenient for some city use.", "🚙")]
    elif priority == "safety" or road == "wet" or rainfall >= 30:
        options = [
            ("Tata Nexon", 9.0, "Petrol / Diesel / EV", "Compact SUV", "A compact SUV to compare for wet-weather use; tyres, speed and exact safety equipment matter most.", "🚙"),
            ("Hyundai Venue", 8.0, "Petrol / Diesel", "Compact SUV", "A city-friendly SUV option; verify safety equipment on the selected variant.", "🚙"),
            ("Honda Amaze", 8.0, "Petrol", "Compact sedan", "A sedan alternative for everyday roads; check current variant equipment and ground clearance.", "🚗")]
    elif priority == "economy":
        options = [
            ("Maruti Suzuki Swift", 7.0, "Petrol", "Hatchback", "Compact for city use; compare real-world mileage, insurance and service costs.", "🚗"),
            ("Maruti Suzuki Wagon R", 6.0, "Petrol / CNG", "Tall hatchback", "Practical city-focused option; compare variant safety features and running costs.", "🚗"),
            ("Toyota Urban Cruiser Taisor", 8.0, "Petrol / CNG", "Compact crossover", "A crossover alternative; check variant price and ownership costs.", "🚙")]
    else:
        options = [
            ("Maruti Suzuki Swift", 7.0, "Petrol", "Hatchback", "Compact city hatchback for everyday urban travel.", "🚗"),
            ("Hyundai i20", 8.0, "Petrol", "Premium hatchback", "A comfort-focused hatchback option; compare safety kit and variant pricing.", "🚗"),
            ("Tata Nexon", 9.0, "Petrol / Diesel / EV", "Compact SUV", "An SUV alternative if you prefer a higher seating position.", "🚙")]
    if ev_charging and terrain == "city" and passengers <= 4 and priority in ("economy", "balanced"):
        options.insert(0, ("Tata Punch.ev", 11.0, "Electric", "Electric city SUV", "Consider if daily range, home/work charging and budget fit your routine.", "⚡"))
    fitting = [o for o in options if o[1] <= budget_lakh]
    chosen = fitting[:3] if fitting else sorted(options, key=lambda o: abs(o[1]-budget_lakh))[:3]
    note = "Illustrative shortlist only. Prices are rough ex-showroom guide figures, not live quotes; verify current city, variant and on-road price."
    if not fitting:
        note = "No example in this shortlist clearly fits the selected budget. Closest examples are shown; prices vary by city, variant and date. " + note
    return [{"name":n,"price":p,"fuel":f,"type":t,"why":w,"icon":i} for n,p,f,t,w,i in chosen], note

with st.sidebar:
    st.markdown("<div class='eyebrow'>CLIMATEDRIVE / CONTROL PANEL</div>", unsafe_allow_html=True)
    st.markdown("## Your driving profile")
    st.caption("Adjust the conditions to explore a demo recommendation.")
    page = st.radio("Workspace", ["Vehicle dashboard", "Climate insights", "About the model"], label_visibility="collapsed")
    st.divider()
    st.markdown("### Weather conditions")
    temp = st.slider("Temperature (°C)", -5, 50, 29)
    rain = st.slider("Rainfall (mm)", 0, 150, 18)
    humidity = st.slider("Humidity (%)", 10, 100, 58)
    wind = st.slider("Wind speed (km/h)", 0, 120, 16)
    st.markdown("### Road & journey")
    road = st.selectbox("Road condition", ["dry", "wet", "rough", "icy"])
    terrain = st.selectbox("Terrain", ["city", "highway", "hills", "rural"])
    passengers = st.slider("Passengers", 1, 8, 4)
    priority = st.selectbox("Main priority", ["balanced", "economy", "safety", "cargo"])
    budget = st.slider("Budget (₹ lakh, ex-showroom guide)", 4, 25, 10)
    ev = st.toggle("I have convenient EV charging", value=False)
    st.caption("Prices are approximate examples. Verify current listings before purchase.")

try:
    df = load_data()
    model, encoders, target_encoder, score = train_model()
except Exception as e:
    st.error(f"Could not load project data. Keep app.py and climate_vehicle_dataset.csv in the same folder. Details: {e}")
    st.stop()

st.markdown("""
<div class="hero">
 <div class="eyebrow">SMART MOBILITY • WEATHER-AWARE DECISIONS</div>
 <h1>Find the right drive<br>for the conditions.</h1>
 <p>A climate-aware vehicle decision-support demo for Indian roads. Explore a machine-learning category estimate, then compare practical model examples against your budget and journey profile.</p>
 <div style="margin-top:18px"><span class="pill">LIVE INPUT CONTROLS</span><span class="pill">INDIA MODEL EXAMPLES</span><span class="pill">ML + RULE-BASED SHORTLIST</span></div>
</div>
""", unsafe_allow_html=True)

if page == "Vehicle dashboard":
    # Build prediction features
    row = {"temperature_c":temp, "rainfall_mm":rain, "humidity_pct":humidity, "wind_kmh":wind,
           "road_condition":safe_encode(road, encoders["road_condition"]), "terrain":safe_encode(terrain, encoders["terrain"]),
           "passengers":passengers, "priority":safe_encode(priority, encoders["priority"])}
    pred_code = model.predict(pd.DataFrame([row], columns=FEATURES))[0]
    pred_label = target_encoder.inverse_transform([pred_code])[0]
    probs = model.predict_proba(pd.DataFrame([row], columns=FEATURES))[0]
    confidence = float(probs.max()) * 100
    recommendations, note = recommend_exact_models(rain, road, terrain, passengers, priority, budget, ev)
    c1,c2,c3,c4 = st.columns(4)
    metrics = [("TEMPERATURE",f"{temp}°C","Current scenario input"),("RAINFALL",f"{rain} mm","Scenario rainfall"),("ML CATEGORY",pred_label,"Model-estimated category"),("TOP-CLASS SCORE",f"{confidence:.0f}%","Demo model score, not certainty")]
    for col,(label,value,sub) in zip([c1,c2,c3,c4],metrics):
        with col: st.markdown(f"<div class='metric'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-sub'>{sub}</div></div>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("<div class='section-kicker'>01 / RECOMMENDATION ENGINE</div><h2>Vehicles to compare</h2>",unsafe_allow_html=True)
    st.markdown("<div class='note'>The ML model predicts a broad vehicle category from a synthetic training dataset. Exact Indian model names below are a separate, transparent rule-based shortlist—not ML predictions or personalized professional advice.</div>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    cols = st.columns(len(recommendations))
    for idx,(col,item) in enumerate(zip(cols,recommendations)):
        with col:
            price = f"₹{item['price']:.0f}L" if item['price'] is not None else "—"
            st.markdown(f"""<div class="vehicle-card"><div class="vehicle-art"><small>{item['type']}</small>{item['icon']}</div><div class="vehicle-body"><div class="vehicle-name">{item['name']}</div><div class="vehicle-price">{price}</div><div class="vehicle-meta">{item['fuel']} · Approx. ex-showroom guide</div><div class="vehicle-reason">{item['why']}</div><span class="tag">Climate-aware</span><span class="tag">Compare variants</span></div></div>""",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown(f"<div class='note'>ⓘ {note}</div>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    left,right = st.columns([1.15,1])
    with left:
        st.markdown("<div class='section-kicker'>02 / INPUT SNAPSHOT</div><h3>Your scenario at a glance</h3>",unsafe_allow_html=True)
        snapshot = pd.DataFrame({"Signal":["Humidity","Wind","Road","Terrain","Passengers","Priority","Budget","EV charging"],"Selected value":[f"{humidity}%",f"{wind} km/h",road.title(),terrain.title(),str(passengers),priority.title(),f"₹{budget} lakh", "Available" if ev else "Not specified"]})
        st.dataframe(snapshot, use_container_width=True, hide_index=True)
    with right:
        st.markdown("<div class='section-kicker'>03 / MODEL TRANSPARENCY</div><h3>What the score means</h3>",unsafe_allow_html=True)
        st.markdown(f"The model's highest class score is **{confidence:.1f}%** for **{pred_label}**. This is the model's score across its learned classes—not a probability that a real vehicle is safe or ideal. The test split accuracy on the synthetic dataset is **{score*100:.1f}%**.")
        st.progress(min(confidence/100,1.0))
        st.caption("Synthetic demonstration data • Not validated against real-world outcomes")

elif page == "Climate insights":
    st.markdown("<div class='section-kicker'>DATA EXPLORER</div><h2>Climate & mobility patterns</h2>",unsafe_allow_html=True)
    st.markdown("Explore the dataset used in this classroom prototype. The dataset is synthetic, so charts illustrate the workflow rather than real-world climate or vehicle performance.")
    a,b = st.columns(2)
    with a:
        st.markdown("### Road-condition mix")
        st.bar_chart(df["road_condition"].value_counts())
    with b:
        st.markdown("### Vehicle category labels")
        st.bar_chart(df["recommended_vehicle"].value_counts())
    c,d = st.columns(2)
    with c:
        st.markdown("### Temperature distribution")
        st.line_chart(df["temperature_c"].sort_values().reset_index(drop=True))
    with d:
        st.markdown("### Rainfall vs humidity")
        st.scatter_chart(df, x="rainfall_mm", y="humidity_pct", color="recommended_vehicle")
    st.markdown("### Preview dataset")
    st.dataframe(df.head(20),use_container_width=True,hide_index=True)
    st.download_button("Download demo dataset CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="climate_vehicle_dataset.csv", mime="text/csv")

else:
    st.markdown("<div class='section-kicker'>PROJECT NOTES</div><h2>About ClimateDrive</h2>",unsafe_allow_html=True)
    st.markdown("### How it works")
    st.markdown("1. **Input layer:** temperature, rainfall, humidity, wind, road, terrain, passengers and travel priority.\n2. **ML category estimate:** a Random Forest classifier trained on a synthetic educational dataset predicts a broad vehicle category.\n3. **Exact-model shortlist:** transparent Python rules choose example Indian-market models based on passengers, road/terrain, priority, budget and EV charging access.\n4. **Decision support:** the interface shows assumptions and caveats so users can inspect the result.")
    st.markdown("### Model details")
    m1,m2,m3 = st.columns(3)
    with m1: st.markdown(f"<div class='metric'><div class='metric-label'>ALGORITHM</div><div class='metric-value' style='font-size:22px'>Random Forest</div><div class='metric-sub'>Supervised classification</div></div>",unsafe_allow_html=True)
    with m2: st.markdown(f"<div class='metric'><div class='metric-label'>ESTIMATORS</div><div class='metric-value'>180</div><div class='metric-sub'>Trees in ensemble</div></div>",unsafe_allow_html=True)
    with m3: st.markdown(f"<div class='metric'><div class='metric-label'>TEST ACCURACY</div><div class='metric-value'>{score*100:.1f}%</div><div class='metric-sub'>Synthetic held-out split only</div></div>",unsafe_allow_html=True)
    st.markdown("### Limitations & responsible use")
    st.warning("This is a college project prototype, not a road-safety tool or commercial recommendation system. Training data is synthetic; measured accuracy does not establish real-world reliability. Exact model suggestions are rule-based examples, not generated by the classifier. Price guides are approximate and may be outdated. Always verify current prices, variants, local availability, road conditions and official safety information.")
    st.markdown("### Suggested future work")
    st.markdown("- Replace synthetic data with validated, documented datasets.\n- Evaluate against a separate real-world test set and report class-wise metrics.\n- Add official vehicle specifications and verified current pricing.\n- Include location-specific weather feeds only with clear source attribution.\n- Add tests for edge cases and user feedback.")

st.divider()
st.caption("CLIMATEDRIVE • Academic prototype • Built with Streamlit + scikit-learn • Use official sources for real purchase and travel decisions")
