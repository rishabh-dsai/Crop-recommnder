# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:27:19 2023

@author: asus
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(page_title='Crop Recommendation Model',layout='wide')

model=joblib.load("Crop Recommender v2.sav")
#soil_df=pd.read_excel("Model Input & Master Sheet for Apps.xlsx",sheet_name="Main V2  Fertilizer Recommendat")
env_df=pd.read_excel("new data.xlsx")
env_df=env_df.dropna()

mapping=env_df[['Crop_Name','Crop-Variety']].drop_duplicates().set_index("Crop_Name")

variety_dict={"Horticulture":" Mango, Litchi, Malta, Oranges, Lemon, Aonla, Guava,Pomegranate",
              "Wheat":"UP 2903, UP 2938, UP 2855, UP 2784, UP2628, UP2554, HD3086, HD2967, WH1105, DPW 621-50, PBW 502, WH542, BL 953",
              "Mustard":"VLToria 3, Pant Hill Toria 1, Uttara, Pant Pili Sarson 1, Pant Rye 20, Pant Rye 21, RGN",
              "Maize":"Sankul: Pant Sankul Makka-3 , Sweta, Bajora Makka 1, Vivak Sankul 11, Hybrid: H M 10, H Q PM 1, 4, Pusa HQPM5, Pant Sankar makka 1 & 4, Sartaj, P 3522, Bio 9544",
              "Potato":"Kufri Jyoti, Kufri Giriraj, Kufri Jawahar, kufri Bahar, Kufri satraj, Kufri chipsona, Kufri Lalima, Kufri Chandan",
              "Paddy":"Narendra 359, HKR 47, PR 113 & 114, Pant Dhan 10,12, 19, 24, 26 & 28, Pant Sugandha Dhan-15,17, 25 & 27, Pusa Basmati 1121, 2511 & 1509, Pant Bansmati 1 and 2, Pusa Basmati 1692, Pusa Basmati 1847, Pusa Basmati 1885, Pusa Basmati 1886, Pusa Basmati 1637, Pusa Basmati 1718, Pusa Basmati 1728",
              "Vegetable":"Onion, Chilly, Peas, Radish, Cauliflower",
              " Soyabean":"PS 1225, PS 1347, PS 24, PS 26, PS 19",
              " Maize":"Sankul: Pant Sankul Makka-3 , Sweta, Bajora Makka 1, Vivak Sankul 11, Hybrid: H M 10, H Q PM 1, 4, Pusa HQPM5, Pant Sankar makka 1 & 4, Sartaj, P 3522, Bio 9544",
              " Potato":"Kufri Jyoti, Kufri Giriraj, Kufri Jawahar, kufri Bahar, Kufri satraj, Kufri chipsona, Kufri Lalima, Kufri Chandan"}

#%%

# Header Display

st.markdown('<div style="text-align: center; color:#004F92 ;font-size:40px; font-weight:bold">DeepSpatial Agriverse Platform</div>', unsafe_allow_html=True)
st.markdown('<div style="background-color:#00609C;padding:7px"> <h2 style="color:white;text-align:center;">Crop Recommendation</h2> </div>',unsafe_allow_html=True)
st.header("")


left,right=st.columns(2)

with right:
# Take Inputs
    district=st.selectbox("District",("Dehradun", "Champawat"),disabled=True)
    block=st.selectbox("Block",("Vikasnagar", "Dalu"),disabled=True)
    village=st.selectbox("Village",env_df['Village Name'].unique())
    farm_num_df=env_df.groupby(['Village Name','Crop_Name'])['Farm_ID'].unique().reset_index()
    pivot_df=pd.pivot_table(farm_num_df,index="Village Name",columns='Crop_Name',values="Farm_ID")
    pivot_df_2=pivot_df.applymap(lambda z:z[:10])
    vlg_df=pivot_df_2.loc[village,:]
    farm_opts=np.append(np.append(vlg_df[0],vlg_df[1]),vlg_df[2])
    farm_opts=np.array(farm_opts,dtype=int)
    farm=st.selectbox("Farm Number",farm_opts)

inp_df=env_df.set_index(['Village Name','Farm_ID'])
n=inp_df.loc[(village,farm),'N']
p=inp_df.loc[(village,farm),'P']
k=inp_df.loc[(village,farm),'K']
ph=inp_df.loc[(village,farm),'pH']
area=inp_df.loc[(village,farm),'Area (Hectares)']
rain=inp_df.loc[(village,farm),'Rainfall']    
temp=inp_df.loc[(village,farm),'Temperature']
humid=inp_df.loc[(village,farm),'Humidity']

with left:
    c1,c2=st.columns(2)
    with c1:
        fig_temp = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = temp,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Temperature","font":{"size":24,"color":"red"}},  
        gauge = {
                'axis': {'range': [None, 40], 'tickwidth': 1, 'tickcolor': "darkred"},
                'bar': {'color': "#8b0000"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': '#FFFF00'},
                    {'range': [20, 40], 'color': '#FFA500'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 38}}))
        fig_temp.update_layout(height=300)
        st.plotly_chart(fig_temp,use_container_width=True)

    with c2:
        fig_humid = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = humid,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Humidity","font":{"size":24,"color":"darkblue"}},
            gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': 'cyan'},
                        {'range': [50, 100], 'color': 'royalblue'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 98}}))
        fig_humid.update_layout(height=300)
        st.plotly_chart(fig_humid,use_container_width=True)
    
# 
if st.button("Predict",use_container_width=True):
    pred=model.predict([[area,humid,k,n,p,rain,temp,ph]])
    crop_list=pred[0].split(",")
    variety_list=[variety_dict[c] for c in crop_list]
    display_df=pd.DataFrame(zip(crop_list,variety_list),columns=["Recommended Crop","Variety"],index=range(1,len(crop_list)+1))
    st.dataframe(display_df,use_container_width=True)   
    
st.header("")
image = Image.open('deepspatial.jpg')
image_1=image.resize((180,30))

cen1,cen2,cen3,cen4,cen5=st.columns(5)
with cen3:
    st.image(image_1)




