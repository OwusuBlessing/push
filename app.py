# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 21:32:41 2022

@author: ME

"""
import numpy as np
import streamlit as st
import pickle

feature_names = ['Location_Easting_OSGR', 'Location_Northing_OSGR', 'Longitude',
       'Latitude', 'Police_Force', 'Number_of_Vehicles',
       'Number_of_Casualties', 'Day_of_Week', 'Local_Authority_(District)',
       'First_Road_Class', '1st_Road_Number', 'Road_Type', 'Speed_limit',
       'Junction_Detail', 'Junction_Control', '2nd_Road_Class',
       '2nd_Road_Number', 'Light_Conditions', 'Weather_Conditions',
       'Road_Surface_Conditions', 'Carriageway_Hazards', 'Urban_or_Rural_Area',
       'Did_Police_Officer_Attend_Scene_of_Accident', 'Hour', 'Month']
with open("Json_files/all_encoded_features.json", 'rb') as fp:
    all_dic = pickle.load(fp)
    
with open('Json_files/min_max_dic.json', 'rb') as fp:
    minmax_dic = pickle.load(fp)
    
path = "Model/xgb_top_25_features.pkl"
def load_model(path):
    with open(path,"rb") as f:
        model = pickle.load(f)
    return model

def get_value(feature_name,val,my_dict=all_dic):
    feature_dic = my_dict[feature_name]
    for key,value in feature_dic.items():
        if val == key:
           return value
def scale_value(feature_name,value,dic =minmax_dic ):
    maximum = dic[feature_name][0]
    minimum = dic[feature_name][1]
    scaled_value = (value - minimum)/(maximum - minimum)
    return scaled_value
def get_feature_dic(feature_name,dic = all_dic):
    return dic[feature_name]

def main():
    # Face Analysis Application #
    st.title("Traffic Accident Severity prediction in the UK")
    activities = ["Home", "Predict Traffic Accident Severity", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    st.sidebar.markdown(
        """ Developed by Team kubeflow for hamoye premier project   
            Email us @ : owususammy509@gmail.com
            """)
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Definition : The degree of under-reporting for traffic accidents is quite high in the United kingdom  which results into incorrect injury degree.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The main functionality of the application to predict the injury on 3-step scale.
                 1. Fatal
                 2. Serious.
                 3. Slight
                 """)
    elif choice == "Predict Traffic Accident Severity":
            st.subheader("Accident Severity Prediction")
            Location_Easting_OSGR = st.number_input("Enter the geographical location  Easting of accident location",min_value=None,max_value=None)
            log_value = np.log(Location_Easting_OSGR)
            east_scaled = scale_value("Location_Easting_OSGR",log_value)
            
         
            Location_Northing_OSGR= st.number_input("Enter the geographical location  Northing of accident location")
            log_value_2 = np.log(Location_Northing_OSGR)
            north_scaled = scale_value("Location_Easting_OSGR",log_value_2)
            
            Longitude = st.number_input("Enter the longitude(GWS84 format) of the accident location")
            long_scaled = scale_value("Location_Easting_OSGR",Longitude)
            
            Latitude= st.number_input("Enter the latitude(GWS84 format)  of accident location")
            log_value_3 = np.log(Latitude)
            lat_scaled = scale_value("Location_Easting_OSGR",log_value_3)
            
            police_force = st.selectbox("which Police is assigned or responsible in area of accident",tuple(get_feature_dic("Police_Force").keys()))
            police_force_code = get_value("Police_Force",police_force)
            pf_scld = scale_value("Police_Force",police_force_code )
            
            number_of_vehicle = st.number_input("Enter number of vehicles involved",min_value=1,max_value=100,step=1)
            nfv_scld = scale_value("Number_of_Vehicles",number_of_vehicle)
            
            Number_of_Casualties= st.number_input("How many Casualities are involved",min_value=1,max_value=100,step=1)
            ncs_scld = scale_value("Number_of_Casualties", Number_of_Casualties)
            
            Day_of_week = st.radio("Select the day of the incidence",tuple(get_feature_dic("Day_of_week").keys()))
            day_code = get_value("Day_of_week",Day_of_week)
            df_week_scld = scale_value("Day_of_Week",day_code)
            
            Local_Authority_District = st.selectbox("Select the local authority district",tuple(get_feature_dic("Local_Authority_District").keys()))
            lc_code = get_value("Local_Authority_District",Local_Authority_District)
            lc_scld = scale_value("Local_Authority_(District)",lc_code)
            
            First_Road_Class = st.radio("What is the classification of the road where the accident happened",tuple(get_feature_dic("first_road_class").keys()))
            road_code = get_value("first_road_class",First_Road_Class)
            rc_scld = scale_value("1st_Road_Class",road_code)
            
            first_road_number = st.number_input("What is the number of the road selected above?",min_value=1,max_value=500,step=1)
            frn_scld = scale_value("1st_Road_Number",first_road_number)
            
            Road_Type = st.radio("What type of road did the accident occur",tuple(get_feature_dic("Road_type").keys()))
            rt_code = get_value("Road_type",Road_Type)
            rt_scld = scale_value("Road_Type",rt_code)
            
            limit=[30,50,40,20,10,70,60 ,0]
            speed_limit = st.radio("What is the speed limit at the region of accident?",tuple(limit))
            sl_scld = scale_value("Speed_limit",speed_limit)
            
            Junction_Detail = st.radio("Select the correct information about the junction",tuple(get_feature_dic("junction_detail").keys()))
            jd_code = get_value("junction_detail",Junction_Detail)
            jd_scld = scale_value("Junction_Detail",jd_code)
            
            Junction_Control = st.radio("What type of traffic control is available at the junction",tuple(get_feature_dic("junction_control").keys()))
            jc_code = get_value("junction_control",Junction_Control)
            jc_scaled = scale_value("Junction_Control", jc_code)
            
            second_road_class = st.radio("Did the accident happend in a crossing ,if yes:select it's class.",tuple(get_feature_dic("second_road_class").keys()))
            src_code = get_value("second_road_class",second_road_class)
            sr_scaled = scale_value("2nd_Road_Class",src_code)
            
            road_num = st.number_input("What is the number of road selected in the previous question",min_value=1,max_value=500,step=1)
            scd_number_scld = scale_value("2nd_Road_Number",road_num)
            
            Light_Conditions = st.radio("What type of lighting conditions was available at the site of accident",tuple(get_feature_dic("Light_Conditions").keys()))
            lcon_code = get_value("Light_Conditions",Light_Conditions)
            lcon_scld = scale_value("Light_Conditions",lcon_code)
            
            Weather_Conditions = st.radio("What is the weather condition at the time of accident",tuple(get_feature_dic("Weather_Conditions").keys()))
            wc_code = get_value("Weather_Conditions",Weather_Conditions)
            wc_scaled = scale_value("Weather_Conditions", wc_code)
            
            Road_Surface_Conditions = st.radio("What is the nature of the road surface?",tuple(get_feature_dic("Road_Surface_Conditions").keys()))
            rsc_code = get_value("Road_Surface_Conditions",Road_Surface_Conditions)
            rsc_scld = scale_value("Road_Surface_Conditions", rsc_code)
            
            Carriageway_Hazards = st.radio("Is there any event or object happening at the context of the accident?",tuple(get_feature_dic("Carriageway_Hazards").keys()))
            ch_code = get_value("Carriageway_Hazards",Carriageway_Hazards)
            ch_scld = scale_value("Carriageway_Hazards",ch_code)
            
            Urban_or_Rural_Area = st.radio("Is the area of accident rural or urban?",tuple(get_feature_dic("Urban_or_Rural_Area").keys()))
            ur_code =get_value("Urban_or_Rural_Area",Urban_or_Rural_Area)
            ur_scld = scale_value("Urban_or_Rural_Area", ur_code)
            
            plc_attend = st.radio("Did the police attend the scene of accident to make report ?",tuple(get_feature_dic("Did_Police_Officer_Attend_Scene_of_Accident").keys()))
            plc_code = get_value("Did_Police_Officer_Attend_Scene_of_Accident",plc_attend)
            plc_scld = scale_value("Did_Police_Officer_Attend_Scene_of_Accident",plc_code)
            
            hr = st.selectbox("What hour of the day did the accident occur?",tuple(get_feature_dic("Hour_of_accident").keys()))
            hr_code = get_value("Hour_of_accident",hr)
            hr_scld = scale_value("Hour",hr_code)
            
            month = st.selectbox("Select the month the accident occured",tuple(get_feature_dic("Month_of_occurence").keys()))
            month_code = get_value("Month_of_occurence",month)
            month_scld = scale_value("Month",month_code)
            
            feature_values = [east_scaled,north_scaled,long_scaled,lat_scaled,pf_scld,nfv_scld,ncs_scld,df_week_scld,lc_scld,rc_scld,frn_scld,rt_scld,sl_scld,jd_scld,jc_scaled,sr_scaled,scd_number_scld,lcon_scld,wc_scaled,rsc_scld,ch_scld,ur_scld,plc_scld,hr_scld,month_scld]
            
            pretty_result = {'Location_Easting_OSGR', 'Location_Northing_OSGR', 'Longitude',
                   'Latitude', 'Police_Force', 'Number_of_Vehicles',
                   'Number_of_Casualties', 'Day_of_Week', 'Local_Authority_(District)',
                   'First_Road_Class', '1st_Road_Number', 'Road_Type', 'Speed_limit',
                   'Junction_Detail', 'Junction_Control', '2nd_Road_Class',
                   '2nd_Road_Number', 'Light_Conditions', 'Weather_Conditions',
                   'Road_Surface_Conditions', 'Carriageway_Hazards', 'Urban_or_Rural_Area',
                   'Did_Police_Officer_Attend_Scene_of_Accident', 'Hour', 'Month'}
            single_sample = np.array(feature_values).reshape(1,-1)
            
          
            if st.button("Predict"):
                preds = load_model(path).predict(single_sample)
              
                if preds == 0:
                      st.write("Fatal")
                      
                elif preds == 1:
                      st.write("Serious")
                    
                else:
                      st.write("Slight")
                    
                             
                      

                                  
                
            
            
            
            
            
            
            
            

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Definition : </h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                                     <div style="background-color:#98AFC7;padding:10px">
                                     <h4 style="color:white;text-align:center;">This Application is developed by Team Kubeflow of hamoye summer internshop 2022 using Streamlit Framework for the purpose of desmontrating traffic accident severity prediction to estimate accidents's injury degree</h4>
                                     <h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                                     </div>
                                     <br></br>
                                     <br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


        


if __name__ == "__main__":
    main()