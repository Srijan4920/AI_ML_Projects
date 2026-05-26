import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from fpdf import FPDF

# Set page config for premium look
st.set_page_config(
    page_title="Disease Predictor AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling for visual excellence
st.markdown("""
<style>
    /* Main container styling */
    .reportview-container {
        background: #0e1117;
    }
    /* Card design */
    .prediction-card {
        background-color: #1e293b;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        margin-bottom: 20px;
    }
    .prediction-header {
        font-size: 24px;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 12px;
    }
    .prediction-value {
        font-size: 32px;
        font-weight: 800;
        color: #f0fdf4;
        background: linear-gradient(135deg, #0ea5e9, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .model-badge {
        background-color: #334155;
        color: #cbd5e1;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 500;
        margin-right: 8px;
        display: inline-block;
        margin-top: 4px;
    }
    /* Alert styles */
    .alert-high {
        background-color: #7f1d1d;
        color: #fef2f2;
        border: 1px solid #b91c1c;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
    }
    .alert-medium {
        background-color: #7c2d12;
        color: #fff7ed;
        border: 1px solid #ea580c;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
    }
    .alert-low {
        background-color: #064e3b;
        color: #ecfdf5;
        border: 1px solid #059669;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Disease Info Database
DISEASE_INFO = {
    "Paralysis (brain hemorrhage)": {
        "description": "Loss of muscle function in part of your body, often caused by a brain hemorrhage or stroke.",
        "precautions": ["Seek emergency medical attention immediately", "Do not move the patient unnecessarily", "Keep airways clear"],
        "severity": "High"
    },
    "Hypertension": {
        "description": "A condition in which the force of the blood against the artery walls is too high (High Blood Pressure).",
        "precautions": ["Reduce salt intake", "Avoid stress", "Monitor blood pressure regularly", "Take prescribed medication"],
        "severity": "Medium"
    },
    "Hepatitis B": {
        "description": "A serious liver infection caused by the hepatitis B virus.",
        "precautions": ["Avoid alcohol", "Consult a hepatologist", "Eat a healthy balanced diet", "Get vaccinated"],
        "severity": "High"
    },
    "Impetigo": {
        "description": "A highly contagious skin infection that mainly affects infants and young children.",
        "precautions": ["Keep affected areas clean and dry", "Avoid touching or scratching sores", "Wash hands regularly", "Consult a dermatologist"],
        "severity": "Low"
    },
    "Chronic cholestasis": {
        "description": "A long-term condition where the flow of bile from your liver is reduced or blocked.",
        "precautions": ["Avoid fatty foods", "Limit alcohol intake", "Consult a gastroenterologist", "Take fat-soluble vitamins"],
        "severity": "Medium"
    },
    "Hepatitis C": {
        "description": "An infection caused by a virus that attacks the liver and leads to inflammation.",
        "precautions": ["Avoid alcohol", "Get tested regularly", "Practice safe hygiene", "Consult a specialist"],
        "severity": "High"
    },
    "Typhoid": {
        "description": "An infectious bacterial fever that causes severe intestinal irritation.",
        "precautions": ["Drink boiled or bottled water", "Eat thoroughly cooked food", "Practice strict hand hygiene", "Complete the full antibiotic course"],
        "severity": "High"
    },
    "Dimorphic hemorrhoids(piles)": {
        "description": "Swollen veins in your lower rectum and anus, similar to varicose veins.",
        "precautions": ["Eat high-fiber foods", "Drink plenty of water", "Avoid straining during bowel movements", "Use warm baths"],
        "severity": "Low"
    },
    "Vertigo (Benign paroxysmal Positional Vertigo)": {
        "description": "A sensation of spinning or dizziness, typically triggered by specific changes in head position.",
        "precautions": ["Avoid sudden head movements", "Sit or lie down immediately when feeling dizzy", "Perform Epley maneuver under guidance"],
        "severity": "Medium"
    },
    "Cervical spondylosis": {
        "description": "Age-related wear and tear affecting the spinal disks in your neck.",
        "precautions": ["Maintain good posture", "Perform gentle neck exercises", "Use a supportive neck pillow", "Avoid heavy lifting"],
        "severity": "Medium"
    },
    "Tuberculosis": {
        "description": "A potentially serious infectious disease that mainly affects your lungs.",
        "precautions": ["Stay isolated during initial treatment", "Take all prescribed medications fully", "Wear a mask around others", "Ensure good ventilation"],
        "severity": "High"
    },
    "Hyperthyroidism": {
        "description": "The overproduction of thyroid hormones by the thyroid gland, accelerating metabolism.",
        "precautions": ["Monitor heart rate", "Reduce caffeine intake", "Follow a low-iodine diet if advised", "Take antithyroid medication"],
        "severity": "Medium"
    },
    "Malaria": {
        "description": "A disease caused by a plasmodium parasite, transmitted by the bite of infected mosquitoes.",
        "precautions": ["Take anti-malarial drugs as prescribed", "Use mosquito nets and repellents", "Avoid stagnant water nearby", "Keep hydrated"],
        "severity": "High"
    },
    "Gastroenteritis": {
        "description": "An intestinal infection marked by diarrhea, cramps, nausea, vomiting, and fever.",
        "precautions": ["Stay hydrated with ORS (Oral Rehydration Salts)", "Follow the BRAT diet (Bananas, Rice, Applesauce, Toast)", "Avoid dairy and fatty foods", "Wash hands thoroughly"],
        "severity": "Medium"
    },
    "Osteoarthritis": {
        "description": "The most common form of arthritis, characterized by the wear and tear of joint cartilage.",
        "precautions": ["Maintain a healthy weight", "Do low-impact exercise (swimming, cycling)", "Use hot or cold therapy", "Consult a rheumatologist"],
        "severity": "Medium"
    },
    "Heart attack": {
        "description": "A medical emergency when blood flow to a part of the heart is blocked.",
        "precautions": ["Call emergency medical services immediately", "Take an aspirin if advised by emergency services", "Stay calm and lie down", "Perform CPR if trained"],
        "severity": "High"
    },
    "Dengue": {
        "description": "A mosquito-borne viral disease causing severe flu-like symptoms and potentially high fever.",
        "precautions": ["Monitor platelet counts regularly", "Stay well hydrated", "Avoid NSAIDs like ibuprofen/aspirin (use paracetamol instead)", "Get plenty of rest"],
        "severity": "High"
    },
    "Pneumonia": {
        "description": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid.",
        "precautions": ["Get plenty of rest", "Stay hydrated", "Take prescribed antibiotics or antivirals", "Avoid smoking and smoke exposure"],
        "severity": "High"
    },
    "Urinary tract infection": {
        "description": "An infection in any part of your urinary system, most commonly the bladder and kidneys.",
        "precautions": ["Drink plenty of water", "Urinate frequently", "Avoid bladder irritants (caffeine, alcohol)", "Complete the antibiotic course"],
        "severity": "Medium"
    },
    "Hypoglycemia": {
        "description": "A condition in which your blood sugar (glucose) level is lower than normal.",
        "precautions": ["Eat/drink 15 grams of fast-acting carbs (juice, candy)", "Recheck blood sugar in 15 minutes", "Carry glucose tablets", "Consult your physician"],
        "severity": "High"
    },
    "Bronchial Asthma": {
        "description": "A chronic condition that inflames and narrows the airways in the lungs.",
        "precautions": ["Avoid triggers (dust, pollen, smoke)", "Always keep a quick-relief rescue inhaler nearby", "Follow your asthma action plan", "Use a peak flow meter"],
        "severity": "High"
    },
    "Arthritis": {
        "description": "Inflammation of one or more joints, causing pain, stiffness, and reduced mobility.",
        "precautions": ["Engage in regular low-impact exercise", "Maintain a healthy body weight", "Use anti-inflammatory medication under guidance", "Apply warm compresses"],
        "severity": "Medium"
    },
    "Hepatitis D": {
        "description": "A serious liver disease caused by the hepatitis D virus, which only occurs in people with hepatitis B.",
        "precautions": ["Avoid alcohol and liver-toxic substances", "Monitor liver health closely", "Consult a hepatology specialist", "Take antiviral medications"],
        "severity": "High"
    },
    "Hypothyroidism": {
        "description": "A condition in which the thyroid gland doesn't produce enough thyroid hormone.",
        "precautions": ["Take thyroid hormone replacement therapy as prescribed", "Have regular thyroid blood tests", "Follow a balanced diet", "Limit soy product intake"],
        "severity": "Medium"
    },
    "Acne": {
        "description": "A common skin condition that occurs when hair follicles become plugged with oil and dead skin cells.",
        "precautions": ["Wash face gently twice a day", "Avoid touching or popping pimples", "Use non-comedogenic skin products", "Consult a dermatologist"],
        "severity": "Low"
    },
    "GERD": {
        "description": "Gastroesophageal reflux disease, a chronic digestive disease where stomach acid flows back into the food pipe.",
        "precautions": ["Avoid triggers (spicy, fatty, or acidic foods)", "Do not lie down immediately after eating", "Eat smaller, more frequent meals", "Elevate the head of your bed"],
        "severity": "Medium"
    },
    "Peptic ulcer disease": {
        "description": "Sores that develop on the inside lining of your stomach and the upper part of your small intestine.",
        "precautions": ["Avoid NSAID pain relievers (like ibuprofen)", "Avoid spicy, acidic, and fatty foods", "Limit alcohol and smoking", "Take antacid medications"],
        "severity": "Medium"
    },
    "Peptic ulcer disease": {
        "description": "Sores that develop on the inside lining of your stomach and the upper part of your small intestine.",
        "precautions": ["Avoid NSAID pain relievers (like ibuprofen)", "Avoid spicy, acidic, and fatty foods", "Limit alcohol and smoking", "Take antacid medications"],
        "severity": "Medium"
    },
    "Psoriasis": {
        "description": "A skin disease that causes red, itchy scaly patches, most commonly on the knees, elbows, trunk, and scalp.",
        "precautions": ["Keep skin moisturized", "Avoid skin injuries and cuts", "Identify and avoid triggers (like stress)", "Consult a dermatologist"],
        "severity": "Medium"
    },
    "Drug Reaction": {
        "description": "An unwanted or adverse reaction to a medication prescribed by a physician or taken over the counter.",
        "precautions": ["Stop the suspected medication immediately", "Consult your physician or pharmacist", "Seek emergency care if breathing difficulty occurs", "Keep a list of allergic medications"],
        "severity": "High"
    },
    "Diabetes": {
        "description": "A disease that occurs when your blood glucose, also called blood sugar, is too high.",
        "precautions": ["Monitor blood glucose levels regularly", "Follow a low-glycemic, balanced diet", "Exercise regularly", "Take prescribed insulin or medication"],
        "severity": "High"
    },
    "Varicose veins": {
        "description": "Gnarled, enlarged veins, most commonly appearing in the legs and feet.",
        "precautions": ["Avoid standing or sitting for long periods", "Elevate legs when resting", "Wear compression stockings", "Exercise to improve circulation"],
        "severity": "Low"
    },
    "Hepatitis A": {
        "description": "A highly contagious liver infection caused by the hepatitis A virus.",
        "precautions": ["Avoid alcohol", "Practice strict handwashing", "Eat simple, nutritious meals", "Rest and allow the liver to recover"],
        "severity": "Medium"
    },
    "Hepatitis E": {
        "description": "A liver disease caused by the hepatitis E virus, mainly transmitted through contaminated drinking water.",
        "precautions": ["Drink safe, boiled, or bottled water", "Practice good hygiene and handwashing", "Pregnant women should seek immediate medical care", "Avoid alcohol"],
        "severity": "High"
    },
    "Migraine": {
        "description": "A neurological condition that can cause multiple symptoms, most notably intense, throbbing headaches.",
        "precautions": ["Identify and avoid dietary and stress triggers", "Rest in a quiet, dark room during attacks", "Use cold compresses on the forehead", "Take prescribed abortive medications"],
        "severity": "Medium"
    },
    "Allergy": {
        "description": "A condition in which the immune system reacts abnormally to a foreign substance.",
        "precautions": ["Avoid known allergens (dust, pollen, specific foods)", "Keep antihistamines handy", "Seek emergency help if anaphylaxis occurs", "Use a dust-proof mattress cover"],
        "severity": "Medium"
    },
    "Jaundice": {
        "description": "A medical condition with yellowing of the skin or whites of the eyes, arising from excess of the pigment bilirubin.",
        "precautions": ["Avoid alcohol completely", "Drink plenty of water and clear fluids", "Eat a low-fat, high-carbohydrate diet", "Consult a hepatologist immediately"],
        "severity": "High"
    },
    "AIDS": {
        "description": "Acquired immunodeficiency syndrome, a chronic, potentially life-threatening condition caused by HIV.",
        "precautions": ["Adhere strictly to antiretroviral therapy (ART)", "Practice safe intimacy", "Avoid exposure to infections", "Eat a highly nutritious diet"],
        "severity": "High"
    },
    "Alcoholic hepatitis": {
        "description": "Inflammation of the liver caused by drinking alcohol.",
        "precautions": ["Stop drinking alcohol completely and permanently", "Follow a high-calorie, nutritious diet", "Monitor for liver failure symptoms", "Consult a hepatologist"],
        "severity": "High"
    }
}

# Categorized symptoms dictionary
CATEGORIZED_SYMPTOMS = {
    "🌡️ Systemic & General": {
        "Fever": "fever",
        "Fatigue": "fatigue",
        "Weight Loss": "weight_loss"
    },
    "🧠 Neurological & Sensory": {
        "Headache": "headache",
        "Yellow Eyes": "yellow_eyes"
    },
    "🤢 Gastrointestinal": {
        "Nausea": "nausea",
        "Vomiting": "vomiting"
    },
    "🦴 Musculoskeletal": {
        "Joint Pain": "joint_pain"
    },
    "🩹 Dermatological": {
        "Skin Rash": "skin_rash"
    },
    "🫁 Respiratory": {
        "Cough": "cough"
    }
}

# Cache model training resources to run instantly
@st.cache_resource
def load_and_train_models():
    data = pd.read_csv('improved_disease_dataset.csv')
    
    # Target encoding
    encoder = LabelEncoder()
    data["disease"] = encoder.fit_transform(data["disease"])
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].values.ravel()
    
    # Define pipelines (oversampling inside the pipeline to avoid data leakage)
    svm_pipeline = ImbPipeline([
        ('sampler', RandomOverSampler(random_state=42)),
        ('classifier', SVC(probability=True, random_state=42))
    ])
    
    nb_pipeline = ImbPipeline([
        ('sampler', RandomOverSampler(random_state=42)),
        ('classifier', GaussianNB())
    ])
    
    rf_pipeline = ImbPipeline([
        ('sampler', RandomOverSampler(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    gb_pipeline = ImbPipeline([
        ('sampler', RandomOverSampler(random_state=42)),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    
    # Ensemble voting classifier with 4 robust classifiers
    voting_clf = VotingClassifier(
        estimators=[
            ('svm', svm_pipeline),
            ('nb', nb_pipeline),
            ('rf', rf_pipeline),
            ('gb', gb_pipeline)
        ],
        voting='soft'
    )
    
    # Fit models
    voting_clf.fit(X, y)
    svm_pipeline.fit(X, y)
    nb_pipeline.fit(X, y)
    rf_pipeline.fit(X, y)
    gb_pipeline.fit(X, y)
    
    return voting_clf, svm_pipeline, nb_pipeline, rf_pipeline, gb_pipeline, encoder, list(X.columns)

# Load data and models
try:
    voting_clf, svm_pipeline, nb_pipeline, rf_pipeline, gb_pipeline, encoder, symptom_cols = load_and_train_models()
except Exception as e:
    st.error(f"Error loading models or dataset: {e}")
    st.stop()

# Helper function to generate PDF without emojis (emojis cause latin-1 font encoding crashes in PDF core fonts)
def generate_pdf(selected_display_names, ensemble_pred, severity, confidence, rf_pred, nb_pred, svm_pred, gb_pred, desc, precautions):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(0, 10, text="MEDICAL DIAGNOSTIC AI REPORT", ln=True, align="C")
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 5, text=f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(5)
    
    # Horizontal line
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    # Symptoms
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, text="Patient Reported Symptoms:", ln=True)
    pdf.set_font('helvetica', '', 11)
    symptoms_str = ", ".join(selected_display_names)
    pdf.multi_cell(0, 6, text=symptoms_str)
    pdf.ln(5)
    
    # Diagnostic Consensus
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, text="Diagnostic Consensus:", ln=True)
    pdf.set_font('helvetica', '', 11)
    pdf.cell(0, 6, text=f"Primary Predicted Disease: {ensemble_pred}", ln=True)
    pdf.cell(0, 6, text=f"Severity Alert Level: {severity}", ln=True)
    pdf.cell(0, 6, text=f"Ensemble Confidence Score: {confidence:.2f}%", ln=True)
    pdf.ln(5)
    
    # Individual Classifier Results
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, text="Individual Classifier Results:", ln=True)
    pdf.set_font('helvetica', '', 11)
    pdf.cell(0, 6, text=f"- Support Vector Machine (SVM): {svm_pred}", ln=True)
    pdf.cell(0, 6, text=f"- Random Forest Classifier: {rf_pred}", ln=True)
    pdf.cell(0, 6, text=f"- Gaussian Naive Bayes: {nb_pred}", ln=True)
    pdf.cell(0, 6, text=f"- Gradient Boosting Classifier: {gb_pred}", ln=True)
    pdf.ln(5)
    
    # Disease Overview
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, text=f"About {ensemble_pred}:", ln=True)
    pdf.set_font('helvetica', '', 11)
    pdf.multi_cell(0, 6, text=desc)
    pdf.ln(5)
    
    # Precautions
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, text="Recommended Precautions:", ln=True)
    pdf.set_font('helvetica', '', 11)
    for p in precautions:
        pdf.cell(0, 6, text=f"- {p}", ln=True)
    pdf.ln(10)
    
    # Disclaimer
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font('helvetica', 'I', 9)
    disclaimer = "Disclaimer: This report was generated by an AI assistant prototype for screening purposes. It does not replace professional medical evaluation. If severity is HIGH, seek emergency services immediately."
    pdf.multi_cell(0, 5, text=disclaimer)
    
    return pdf.output()

# Sidebar - Information
with st.sidebar:
    st.image("https://img.icons8.com/external-flatart-icons-flat-flatarticons/128/external-medical-science-flatart-icons-flat-flatarticons.png", width=80)
    st.title("Diagnostic Center")
    st.markdown("""
    This intelligent diagnostic assistant uses a machine learning ensemble model to analyze symptom inputs and predict the most probable disease.
    
    **Ensemble Models:**
    *   Support Vector Machine (SVM)
    *   Random Forest Classifier
    *   Gradient Boosting Classifier
    *   Gaussian Naive Bayes
    
    *Note: This is an AI prototype for demonstration purposes and is not a substitute for professional medical advice.*
    """)
    st.divider()
    st.caption("Developed with Streamlit & Scikit-Learn")

# Main Page Header
st.title("🩺 Medical Diagnosis Assistant AI")
st.markdown("Use the categorized selectors below to enter symptoms and generate a diagnostic consensus report.")

# Categorized Symptom Selection Section
st.header("1. Choose Symptoms")

selected_symptom_cols = []
selected_display_names = []

# Generate the symptom checklist dynamically in category grids
for category_name, symptoms_dict in CATEGORIZED_SYMPTOMS.items():
    with st.expander(category_name, expanded=True):
        cols = st.columns(3)
        for idx, (display_name, col_name) in enumerate(symptoms_dict.items()):
            col_target = cols[idx % 3]
            is_checked = col_target.checkbox(display_name, key=col_name)
            if is_checked:
                selected_symptom_cols.append(col_name)
                selected_display_names.append(display_name)

# Diagnosis Action
st.divider()

if st.button("Generate Diagnostic Consensus", type="primary", use_container_width=True):
    if len(selected_symptom_cols) == 0:
        st.warning("⚠️ Please select at least one symptom from the categories above to run the diagnostics.")
    else:
        # Create input feature vector
        input_data = [0] * len(symptom_cols)
        for sc in selected_symptom_cols:
            idx = symptom_cols.index(sc)
            input_data[idx] = 1
            
        input_vector = np.array(input_data).reshape(1, -1)
        
        # Predictions
        rf_pred = encoder.classes_[rf_pipeline.predict(input_vector)[0]]
        nb_pred = encoder.classes_[nb_pipeline.predict(input_vector)[0]]
        svm_pred = encoder.classes_[svm_pipeline.predict(input_vector)[0]]
        gb_pred = encoder.classes_[gb_pipeline.predict(input_vector)[0]]
        ensemble_pred = encoder.classes_[voting_clf.predict(input_vector)[0]]
        
        # Soft voting probabilities
        ensemble_probs = voting_clf.predict_proba(input_vector)[0]
        
        # Create a DataFrame for top 5 diseases
        prob_df = pd.DataFrame({
            "Disease": encoder.classes_,
            "Confidence (%)": ensemble_probs * 100
        }).sort_values(by="Confidence (%)", ascending=False).head(5)
        
        # Look up disease info
        disease_details = DISEASE_INFO.get(ensemble_pred, {
            "description": "No description is currently available for this disease.",
            "precautions": ["Consult a medical professional for advice."],
            "severity": "Medium"
        })
        
        severity = disease_details["severity"]
        desc = disease_details["description"]
        precautions = disease_details["precautions"]
        
        # Layout Results
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.subheader("Diagnostic Results")
            
            # Premium card design for the final prediction
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-header">Primary Ensemble Diagnosis</div>
                <div class="prediction-value">{ensemble_pred}</div>
                <div style="margin-top: 16px;">
                    <span class="model-badge">Random Forest: {rf_pred}</span>
                    <span class="model-badge">SVM: {svm_pred}</span>
                    <span class="model-badge">Naive Bayes: {nb_pred}</span>
                    <span class="model-badge">Gradient Boosting: {gb_pred}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Severity Alert Callouts
            if severity == "High":
                st.markdown(f"""
                <div class="alert-high">
                    <strong>🚨 Urgent Medical Warning (Severity: High)</strong><br>
                    {ensemble_pred} can be a severe condition. Immediate professional medical evaluation is recommended.
                </div>
                """, unsafe_allow_html=True)
            elif severity == "Medium":
                st.markdown(f"""
                <div class="alert-medium">
                    <strong>⚠️ Standard Alert (Severity: Medium)</strong><br>
                    Monitor your symptoms closely and schedule an appointment with a general physician soon.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-low">
                    <strong>✅ General Care (Severity: Low)</strong><br>
                    Recommended precautions and home-care should help resolve symptoms. Consult a doctor if they persist.
                </div>
                """, unsafe_allow_html=True)
                
            # Overview and Precautions
            st.markdown(f"### About {ensemble_pred}")
            st.write(desc)
            
            st.markdown("### Recommended Precautions")
            for p in precautions:
                st.write(f"- {p}")
            
        with col2:
            st.subheader("Probability Distributions")
            
            # Bar chart of probabilities
            st.bar_chart(
                prob_df,
                x="Disease",
                y="Confidence (%)",
                color="#0ea5e9",
                use_container_width=True
            )
            
            # Detailed list of top 5
            st.dataframe(
                prob_df.style.format({"Confidence (%)": "{:.2f}%"}),
                hide_index=True,
                use_container_width=True
            )
            
            st.divider()
            
            # Report Generator & Download
            st.subheader("Diagnostic Report Exporter")
            st.caption("Generate a formatted PDF diagnostic report containing selected symptoms, consensus predictions, and medical precautions.")
            
            # Compile PDF
            try:
                pdf_bytes = generate_pdf(
                    selected_display_names=selected_display_names,
                    ensemble_pred=ensemble_pred,
                    severity=severity,
                    confidence=prob_df.iloc[0]['Confidence (%)'],
                    rf_pred=rf_pred,
                    nb_pred=nb_pred,
                    svm_pred=svm_pred,
                    gb_pred=gb_pred,
                    desc=desc,
                    precautions=precautions
                )
                
                # Download Button for PDF
                st.download_button(
                    label="📥 Download Diagnostic PDF Report",
                    data=bytes(pdf_bytes),
                    file_name=f"diagnostic_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as pdf_error:
                st.error(f"Error compiling diagnostic PDF: {pdf_error}")
