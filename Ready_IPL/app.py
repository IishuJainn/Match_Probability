import streamlit as st
import pickle
import pandas as pd

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load the trained model pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="IPL Win Predictor", layout="wide")
st.title('IPL Win Predictor')

# Sidebar
st.sidebar.header('Choose Options')
batting_team = st.sidebar.selectbox('Select the batting team', sorted(teams))
bowling_team = st.sidebar.selectbox('Select the bowling team', sorted(teams))
selected_city = st.sidebar.selectbox('Select host city', sorted(cities))
target = st.sidebar.number_input('Target Score')

# Main content area with three columns
col1, col2, col3 = st.columns(3)

with col1:
    score = st.number_input('Current Score', min_value=0, value=0)
with col2:
    overs = st.number_input('Overs Completed', min_value=0, max_value=20, value=0)
with col3:
    wickets_lost = st.number_input('Wickets Lost', min_value=0, max_value=10, value=0)

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets_lost

    if runs_left <= 0:
        st.error(f"{batting_team} - 100%")
        st.success(f"{bowling_team} - 0%")
    elif balls_left <= 0:
        st.error(f"{batting_team} - 0%")
        st.success(f"{bowling_team} - 100%")
    elif wickets_left <= 0:
        st.error(f"{batting_team} - 0%")
        st.success(f"{bowling_team} - 100%")
    else:
        crr = score / overs
        rrr = (runs_left * 6) / balls_left

        # Create input DataFrame for prediction
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'Runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Predict probabilities using the model pipeline
        result = pipe.predict_proba(input_df)
        loss_probability = result[0][0]
        win_probability = result[0][1]

        st.markdown(f"<h2 style='text-align:center; color:#ff5722;'>{batting_team} - {round(win_probability * 100)}%</h2>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align:center; color:#4caf50;'>{bowling_team} - {round(loss_probability * 100)}%</h2>", unsafe_allow_html=True)
