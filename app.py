import numpy as np
import pickle
import pandas as pd
import streamlit as st
import uvicorn
from numpy import asarray
from sklearn.preprocessing import StandardScaler
import base64

pickle_in=open("Randomforest.pkl","rb")
Randomforest=pickle.load(pickle_in)
scaler=pickle.load(open("scaler.pkl","rb"))
sexencoder=pickle.load(open("sexencoder.pkl","rb"))
classencoder=pickle.load(open("classencoder.pkl","rb"))
travelencoder=pickle.load(open("travelencoder.pkl","rb"))
custencoder=pickle.load(open("custencoder.pkl","rb"))

page_bg_img = '''
<style>
body {
background-image: url("https://wallpapercave.com/wp/wp961689.jpg");
background-size: 100%;

background-repeat: no-repeat;
background-attachment: fixed;

}

</style>

'''
st.markdown(
    """
<style>
.big-font {
    font-size:40px !important;
    color:blue;
    text-align:center;
}
.small-font {
    font-size:20px !important;
    color:white;
    text-align:center;
}
.error_class {
    font-size:25px !important;
    color:black;
    background:#e8dcdc;
    text-align:center;
    
}
.success_class {
    font-size:25px !important;
    color:black;
    background:#66ff66;
    text-align:center;
}
.reportview-container .markdown-text-container {
    font-family: monospace;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
.Widget>label {
    color: white;
    font-size:20px !important;
    font-family: monospace;
    
}
[class^="st-b"]  {
    color: white;
    font-family: monospace;
}
.slider{
    color:black
}
.st-bb {
    background-color: transparent;
}
.st-at {
    background-color: none;
}
footer {
    font-family: monospace;
}
.reportview-container .main footer, .reportview-container .main footer a {
    color: #0c0080;
}
header .decoration {
    background-image: none;
}
.val{
    font-size:10px !important;
    color:black;
    text-align:center;
}
</style>
""",
    unsafe_allow_html=True,
)

def predict_satisfaction(Gender, Customer_Type, Age, Type_of_Travel, Class,
       Flight_Distance, Inflight_wifi_service, Ease_of_Online_booking,
        Food_and_drink, Online_boarding, Seat_comfort,
       Inflight_entertainment, Onboard_service, Leg_room_service,
       Baggage_handling, Checkin_service, Inflight_service,
       Cleanliness, Departure_Delay_in_Minutes, Arrival_Delay_in_Minutes):
    Age=scaler.transform(asarray(Age).reshape(1,-1))
    Flight_Distance=scaler.transform(asarray(Flight_Distance).reshape(1,-1))
    Departure_Delay_in_Minutes=scaler.transform(asarray(Departure_Delay_in_Minutes).reshape(1,-1))
    Arrival_Delay_in_Minutes=scaler.transform(asarray(Arrival_Delay_in_Minutes).reshape(1,-1))
    
    Gender = sexencoder.transform(np.array(Gender).reshape(-1,1))[0]
    Type_of_Travel=travelencoder.transform(np.asarray(Type_of_Travel).reshape(1,-1))[0]
    
    Customer_Type = custencoder.transform(np.asarray(Customer_Type).reshape(1,-1))[0]
    
    Class=classencoder.transform(np.asarray(Class).reshape(1,-1))[0]
    

 
    
    
    predict_satisfaction=Randomforest.predict([[Gender, Customer_Type, Age, Type_of_Travel, Class,
       Flight_Distance, Inflight_wifi_service, Ease_of_Online_booking,
        Food_and_drink, Online_boarding, Seat_comfort,
       Inflight_entertainment, Onboard_service, Leg_room_service,
       Baggage_handling, Checkin_service, Inflight_service,
       Cleanliness, Departure_Delay_in_Minutes, Arrival_Delay_in_Minutes]])   
    if predict_satisfaction==0:
        pred='Passenger not satisfied'
    else:
        pred='Passenger Satisfied'
    return pred    
      
    
    

def main():
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    st.title("Passenger Satisfaction Prediction")
    
    st.subheader("""
             Here is a prediction model that predicts whether the passenger is satisfied or not.
          """) 
   
   
  
    
    
    Gender=st.radio('Gender', ['Male','Female'])
   
    Customer_Type=st.radio('Customer Type', ['Loyal Customer','disloyal Customer'])
   
    Age=st.slider("Select age:",min_value=1,max_value=100)
    
    
    Type_of_Travel=st.radio('Type of Travel', ['Personal Travel','Business travel'])
   
    
    Class=st.radio('Class', ['Eco Plus','Business','Eco'])
   
    
    Flight_Distance=st.text_input('Flight Distance','0')
    Inflight_wifi_service=st.slider('In-flight WiFi service',min_value=0,max_value=5)
    Ease_of_Online_booking=st.slider('Ease of Online Booking',min_value=0,max_value=5)
    Food_and_drink=st.slider('Food and Drink',min_value=0,max_value=5)
    Online_boarding=st.slider('Online boarding',min_value=0,max_value=5)
    Seat_comfort=st.slider('Seat Comfort',min_value=0,max_value=5)
    Inflight_entertainment=st.slider('In-flight Entertainment',min_value=0,max_value=5)
    Onboard_service=st.slider('On-board service',min_value=0,max_value=5)
    Leg_room_service=st.slider('Leg room service',min_value=0,max_value=5)
    Baggage_handling=st.slider('Baggage handling',min_value=0,max_value=5)
    Checkin_service=st.slider('Checkin Service',min_value=0,max_value=5)
    Inflight_service=st.slider('In-flight service',min_value=0,max_value=5)
    Cleanliness=st.slider('Cleanliness',min_value=0,max_value=5)
    Departure_Delay_in_Minutes=st.text_input('Departure Delay in minutes',0)
    Arrival_Delay_in_Minutes=st.text_input('Arrival Delay in minutes',0)
    

    result=""
    
    if st.button("predict"):
         result=predict_satisfaction(Gender, Customer_Type, Age, Type_of_Travel, Class,
       Flight_Distance, Inflight_wifi_service, Ease_of_Online_booking,
        Food_and_drink, Online_boarding, Seat_comfort,
       Inflight_entertainment, Onboard_service, Leg_room_service,
       Baggage_handling, Checkin_service, Inflight_service,
       Cleanliness, Departure_Delay_in_Minutes, Arrival_Delay_in_Minutes)
    st.success(result)
   
        
        

    
if __name__=='__main__':
    main()