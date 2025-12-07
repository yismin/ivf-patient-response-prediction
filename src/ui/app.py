import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
# Add model path
sys.path.append(str(Path(__file__).parent.parent / "model"))
from predict import PatientPredictor
# Page configuration
st.set_page_config(
    page_title="IVF Response Predictor",
    page_icon="🏥",
    layout="wide"
)

# Initialize predictor (cached to load only once)
@st.cache_resource
def load_predictor():
    return PatientPredictor()

predictor = load_predictor()

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-low {
        background-color: #f44336;
        border-left: 5px solid #f44336;
    }
    .prediction-optimal {
        background-color: #4caf50;
        border-left: 5px solid #4caf50;
    }
    .prediction-high {
        background-color: #ff9800;
        border-left: 5px solid #ff9800;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header"> IVF Patient Response Predictor</h1>', unsafe_allow_html=True)

st.markdown("""
This application predicts patient response to IVF treatment based on clinical parameters.
The model provides probability estimates for **low**, **optimal**, and **high** response categories.
""")

# Sidebar - Information
with st.sidebar:
    st.header(" About")
    st.info("""
    **Model Information:**
    - Type: Random Forest Classifier
    - Accuracy: 86.1%
    - F1-Score: 0.860
    
    **Response Categories:**
    - **Low**: Poor response to stimulation
    - **Optimal**: Expected normal response
    - **High**: Hyper-response to stimulation
    """)
    
    st.header("📊 Model Features")
    st.markdown("""
    Most important predictors:
    1. **AMH** (37.9%)
    2. **n_Follicles** (27.3%)
    3. **AFC** (22.6%)
    4. **E2_day5** (5.2%)
    5. **Age** (4.0%)
    """)

# Main content - Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Patient Information")
    
    # Input form
    with st.form("patient_form"):
        age = st.slider("Age (years)", 20, 45, 32, help="Patient's age in years")
        
        amh = st.number_input(
            "AMH (ng/mL)", 
            min_value=0.0, 
            max_value=20.0, 
            value=2.5, 
            step=0.1,
            help="Anti-Müllerian Hormone level"
        )
        
        afc = st.number_input(
            "AFC (Antral Follicle Count)", 
            min_value=0, 
            max_value=50, 
            value=15,
            help="Number of antral follicles at baseline"
        )
        
        n_follicles = st.number_input(
            "Number of Follicles Retrieved", 
            min_value=0, 
            max_value=50, 
            value=12,
            help="Total follicles retrieved during procedure"
        )
        
        e2_day5 = st.number_input(
            "E2 Day 5 (pg/mL)", 
            min_value=0.0, 
            max_value=5000.0, 
            value=450.0, 
            step=10.0,
            help="Estradiol level on day 5 of stimulation"
        )
        
        cycle_number = st.number_input(
            "Cycle Number", 
            min_value=1, 
            max_value=10, 
            value=1,
            help="IVF attempt number"
        )
        
        protocol = st.selectbox(
            "Stimulation Protocol",
            ["flexible antagonist", "fixed antagonist", "agonist"],
            help="Type of ovarian stimulation protocol used"
        )
        
        submit_button = st.form_submit_button("Predict Response", use_container_width=True)

with col2:
    st.header("📈 Prediction Results")
    
    if submit_button:
        # Create patient data
        patient_data = {
            'Age': age,
            'AMH': amh,
            'n_Follicles': n_follicles,
            'E2_day5': e2_day5,
            'AFC': afc,
            'cycle_number': cycle_number,
            'Protocol': protocol
        }
        
        # Make prediction
        with st.spinner("Analyzing patient data..."):
            result = predictor.predict(patient_data)
        
        # Display prediction
        prediction = result['prediction']
        confidence = result['confidence']
        probabilities = result['probabilities']
        
        # Color coding
        if prediction == 'low':
            box_class = 'prediction-low'
            emoji = '🔴'
        elif prediction == 'optimal':
            box_class = 'prediction-optimal'
            emoji = '🟢'
        else:
            box_class = 'prediction-high'
            emoji = '🟡'
        
        # Prediction box
        st.markdown(f"""
        <div class="{box_class} prediction-box">
            <h2>{emoji} Predicted Response: {prediction.upper()}</h2>
            <h3>Confidence: {confidence*100:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability bars
        st.subheader("Probability Distribution")
        
        fig = go.Figure()
        
        colors = {
            'low': '#f44336',
            'optimal': '#4caf50',
            'high': '#ff9800'
        }
        
        categories = ['low', 'optimal', 'high']
        values = [probabilities[cat] * 100 for cat in categories]
        bar_colors = [colors[cat] for cat in categories]
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in values],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Response Probability (%)",
            xaxis_title="Response Category",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.subheader("🩺 Clinical Interpretation")
        
        if confidence > 0.7:
            confidence_text = "**High confidence** - Model is confident in this prediction."
        elif confidence > 0.5:
            confidence_text = "**Moderate confidence** - Prediction is likely but monitor closely."
        else:
            confidence_text = "**Low confidence** - Prediction uncertain, clinical judgment recommended."
        
        st.info(confidence_text)
        
        # Key factors
        st.markdown("**Key Patient Factors:**")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("AMH", f"{amh:.2f} ng/mL")
        with col_b:
            st.metric("AFC", afc)
        with col_c:
            st.metric("Follicles", n_follicles)
        
        # Recommendations
        st.subheader(" Clinical Considerations")
        
        if prediction == 'low':
            st.warning("""
            **Low Response Indicated:**
            - Consider protocol adjustment for next cycle
            - Evaluate need for higher gonadotropin doses
            - Discuss prognosis and alternative options
            - Monitor closely for cycle cancellation criteria
            """)
        elif prediction == 'optimal':
            st.success("""
            **Optimal Response Expected:**
            - Continue with standard protocol
            - Good prognosis for cycle success
            - Normal monitoring schedule
            - Standard trigger timing
            """)
        else:  # high
            st.warning("""
            **High Response Indicated:**
            - Monitor for OHSS (Ovarian Hyperstimulation Syndrome) risk
            - Consider dose reduction if stimulation not started
            - Evaluate need for coasting or cycle cancellation
            - Consider antagonist protocol if not already used
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>IVF Patient Response Prediction System | Tanit AI Internship Project</p>
    <p><small>⚠️ This tool is for research and educational purposes. Always use clinical judgment.</small></p>
</div>
""", unsafe_allow_html=True)

#Run the app with: streamlit run ../src/ui/app.py