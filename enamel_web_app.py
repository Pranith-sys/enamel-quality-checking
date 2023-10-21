import streamlit as st
import numpy as np
from PIL import Image
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Load the dumped model
loaded_model = pickle.load(open(r'classifier_enamel_ml_model.sav', 'rb'))

def image_classification(image_array):
    
    #To evalute input image based on prediction probability
    predict_probability=loaded_model.predict_proba(image_array.reshape(1, -1)).ravel()
    max_perc=predict_probability.max()

    for index_value in range(0,len(predict_probability)):
        if max_perc==predict_probability[index_value]:


            predict_probability_int_value=int((max_perc)*100)
            # MAINTENANCE BREAK, COME BACK FEW MINUTES AFTER
            if predict_probability_int_value >= 95:   
                prediction = loaded_model.predict(image_array.reshape(1, -1))
                if int(prediction[0]) == 1:
                    return "THIS IS BLITERS, SO THIS IS BAD QUALITY"
                elif int(prediction[0]) == 2:
                    return "THIS IS BURN OFF, SO THIS IS BAD QUALITY"
                elif int(prediction[0]) == 3:
                    return "THIS IS CARBON BOIL, SO THIS IS BAD QUALITY"
                elif int(prediction[0]) == 4:
                    return"THIS IS COPPER HEADS, SO THIS IS BAD QUALITY"
                elif int(prediction[0]) == 5:
                    return "THIS IS ENAMEL FALL, SO THIS IS BAD QUALITY"
                elif int(prediction[0]) == 6:
                    return "THIS IS FISH SCALING, SO THIS IS BAD QUALITY"
                elif int(prediction[0]) == 7:
                    return " THIS IS GREASE MARK, SO THIS IS BAD QUALITY"
                elif int(prediction[0]) == 8:
                    return "THIS IS GRIT RESIDUALS, SO THIS IS BAD QUALITY"
                elif int(prediction[0]) == 9:
                    return"THIS IS POOR ADHESIONTHIS, SO THIS IS BAD QUALITY"
                elif int(prediction[0]) == 10:
                    return "THIS IS POP OFF, SO THIS IS BAD QUALITY"
                elif int(prediction[0]) == 11:
                    return "THIS IS WELD PENETRATION, SO THIS IS BAD QUALITY"
                elif int(prediction[0]) == 12:
                    return "THIS IS WELDING SPATTER, SO THIS IS BAD QUALITY"
                elif int(prediction[0]) == 13:
                    return "THIS IS LOW MICRON, SO THIS IS BAD QUALITY"
                elif int(prediction[0]) == 14:
                    return "THIS IS GOOD QUALITY ENAMEL"
                elif int(prediction[0]) == 15:
                    return "THIS IS GOOD QUALITY ENAMEL"
                elif int(prediction[0]) == 16:
                    return "THIS IS GOOD QUALITY ENAMEL"
                else:
                    return  "KINDLY GO TO PREPROCESS"
            
            elif predict_probability_int_value >= 80 and predict_probability_int_value < 95:
                return "I'M NOT QUITE SATISFIED WITH THIS PRODUCT, SO GO TO REWORK"
            elif predict_probability_int_value <=50:
                return "THIS IS NOT APPROPRIATE FOR HERE SO, THIS PRODUCT IS REJECTED!"
            else:
                return 'I'M NOT QUITE SATISFIED WITH THIS PRODUCT, SO GO TO REWORK'
    

def preprocess_image(image):
    
    # Convert the image to grayscale
    grayscale_image = image.convert('L')
    # Resize the image to 80x80 pixels
    resized_image = grayscale_image.resize((128,128))
    
    # Convert the image to a NumPy array and normalize it
    image_array = np.array(resized_image) / 255.0
    
    # Flatten the image array
    flattened_image = image_array.flatten()  # Make sure this results in 19200 features
    
    return flattened_image


def main():   
    # Set the page title
    st.title('Enamel Qulitiy Checking....')
    try:
    # Add a file uploader widget to allow users to upload an image
        uploaded_image = st.file_uploader("Upload an image")
    
        if uploaded_image is not None:
            # Print the name of the uploaded image file to the Python terminal
            st.write("Uploaded image file name:", uploaded_image.name)
        
            # Open the image using Pillow (PIL)
            image = Image.open(uploaded_image)

            preprocessed_image = preprocess_image(image)
        
            # Classify the image
            result = image_classification(preprocessed_image)
           
            # Display the uploaded image
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
            # Display the prediction result
            st.write("RESULT : ", result) #'Please provide either jpg or jpeg file format image....'

    except Exception as e:
        st.error(f"An error occurred: {'Please provide either jpg or jpeg file format image....'}")

if __name__ == '__main__':
    main()
