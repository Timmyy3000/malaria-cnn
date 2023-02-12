
#importing the libraries
import streamlit as st
import pickle
from PIL import Image
import numpy as np
import time


#loading the image classifier model
model = pickle.load(open('model.pkl', 'rb'))

# Designing the interface
st.write(" ### THE APPLICATION OF DEEP LEARNING ARCHITECTURE IN THE INFERENCE OF MALARIA FROM BLOOD CELL IMAGES")
# For newline
st.write('\n')

st.write('''
The goal of this deep learning project is to develop a model for the inference of malaria through cell imaging. 

The project involves collecting a dataset of microscopic images of infected and healthy red blood cells, which will be used to train the deep learning model. The model will be designed using a Convolutional Neural Network (CNN) architecture, which is well-suited for image classification tasks. 

The training process will involve using the collected dataset to optimize the model's parameters, allowing it to accurately identify infected cells based on their visual characteristics. Once the model has been trained, it can be used for real-time inference of malaria in new, unseen images. 

This project has the potential to revolutionize the way that malaria is diagnosed, providing a fast, efficient, and low-cost alternative to traditional diagnostic methods.
''')

image = Image.open('./images/title.jpeg')
show = st.image(image, use_column_width=True)

st.sidebar.title("Upload Image")

#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)

#Choose your own image
uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )

if uploaded_file is not None:
    
    u_img = Image.open(uploaded_file)
    st.sidebar.image(u_img, 'Uploaded Image', use_column_width=False)
    # We preprocess the image to fit in algorithm.
    # Load the image and resize it to match the input shape of your model

    img = u_img.resize((128, 128))

    # Convert the image to a numpy array and add an extra dimension for the batch size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

# For newline
st.sidebar.write('\n')

st.write('''
----

*Ogunme Oluwatimilehin Ayomikun		19/0706*

*Odunfa Ayomide Esther				19/1043*

*Metu-Onyeka Valerie Amarachi		19/0876*


*SUBMITTED TO THE DEPARTMENT OF COMPUTER SCIENCE, SCHOOL OF COMPUTING AND ENGINEERING SCIENCES,  BABCOCK UNIVERSITY, ILISHAN-REMO, OGUN.*

*IN PARTIAL FULFILMENT OF THE REQUIREMENTS FOR THE AWARD OF A BACHELOR OF SCIENCE(BSC) DEGREE IN COMPUTER SCIENCE.*

*UNDER THE SUPERVISION OF*

*Dr. NZENWATA U. J.*

*March, 2023*

''')
if st.sidebar.button("Click Here to Classify"):
    
    if uploaded_file is None:
        
        st.sidebar.write("Please upload an Image to Classify")
    
    else:
        
        with st.spinner('Classifying ...'):
            
            prediction = model.predict(img_array)
            time.sleep(1)
            print(prediction)
            st.success('Done!')
            
        st.sidebar.header("CNN Predicts: ")
        
        
        # Classify cell as  being infected or not
        
        if prediction > 0.5:
            
            st.sidebar.write("This is an Uninfected cell", '\n' )
            
            
                             
        else:
            st.sidebar.write("This is an Infected cell",'\n')
        
    
    