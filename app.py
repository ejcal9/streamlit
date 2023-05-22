import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from PIL import Image, ImageOps
import numpy as np
 
@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('image_classification.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()
 
st.write("""
         # Image Classification
         """
         )
 
file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data,model):
    size=(64,64)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
 
#def upload_predict(upload_image, model):
    
   #     size = (64,64)    
   #     image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
   #     image = np.asarray(image)
   #     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
        
   #     img_reshape = img[np.newaxis,...]
    
    #    prediction = model.predict(img_reshape)
    #    pred_class=decode_predictions(prediction,top=1)
        
    #    return pred_class
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    image_class = ['cat','dog','rat','house', 'fan', 'phone', 'horse', 'bear']
    st.write("The image is classified as",image_class)
    print("The image is classified as ",image_class)
