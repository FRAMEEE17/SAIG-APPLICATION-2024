## import basic libraries
import numpy as np
import streamlit as st
import cv2
from deepface import DeepFace as dfc
from PIL import Image
import os
#from deepface.basemodels import VGGFace, DeepID, DlibWrapper, ArcFace

## CV2 Function To Load Images from Deepface Framework
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

@st.cache_resource
def cached_function():
    #st.write("This should be cached.")
    return "Hello from a cached function!"

st.write(cached_function())
def face_detect(img):
    img = np.array(img.convert("RGB"))
    face = face_cascade.detectMultiScale(image=img)

    # draw rectangle around face
    for (x, y, w, h) in face:
        cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        roi = img[y:y + h, x:x + w]
    
    # TODO: Describe what you did here
    # Added return statement to return the modified image and the detected faces
    return img, face

#enforce detection tries to find a face in the image. If it cannot find a face, then function returns exception. In this way, we can verify images with highly resolution.
#In some cases, you need to still apply face recognition without face detection. We set enforce_detection to False in this case. This decreases the accuracy but it won't return exception.

def analyze_image(img):
    prediction = dfc.analyze(img_path=img,enforce_detection=False)
    return prediction





def main():
    
    st.title("Facial Emotion Detection and :blue[Insight Demo App]", )
    activiteis = ["Home", "Face Image Upload Detection","Face Analysis Live Webcam", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """  Developed by 
             Alpha จุ๊กกรู้วว   🇹🇭
              
        """)
    # C0C0C0
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:20px">
                                            <h4 style="color:white;text-align:center;">
                                            Using OpenCV, DeepFace and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        image = Image.open('messi.jpg')
        st.image(image, width = 700)
        st.write("""
                 The application has 2 functionalities.
                 
                 1. Facial Emotion Features Analysis.
                 
                 2. Face Analysis using Live Web CAM Feed.
                 
                 Future Scope - In the future, we will be adding prediction of age, race, born-gender, body language and multiple emotions 
                 detection using a better model with Webcam live feed.""")
        
    elif choice == "Face Image Upload Detection":
        st.subheader("ฉันรู้นะในใจเธอคิดอะไรอยู่")
        image_file = st.file_uploader("Upload your image here", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            #read image using PIL
            image_loaded = Image.open(image_file)
            #detect faces in image
            result_img, result_face = face_detect(image_loaded)
            st.image(result_img, use_column_width=True)
            st.success("found {} face\n".format(len(result_face)))
            #model_option = st.selectbox("Choose a model", ["VGGFace", "DeepID", "Dlib", "ArcFace"])
            if st.button("Analyze image"):
                # convert image to array
                new_image = np.array(image_loaded.convert('RGB'))
                img = cv2.cvtColor(new_image, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                #analyze features of face
                results = analyze_image(img)
                
                st.write(results)
                for i, result in enumerate(results):
                    
                    st.write(f"Analysis summary for Face {i+1}")
                    st.write("Emotion:", result["dominant_emotion"])
                    st.write("Gender:", result["dominant_gender"])
                    st.write(f"Estimated Age:", result["age"])
                    st.write("Race:", result["dominant_race"])
                    st.write("\n")
            
            else:
                
                st.write("Click on Analyze image ")
    elif choice == "Face Analysis Live Webcam":
        st.markdown(" This section is developing...")
    else:
        pass

if __name__ == '__main__':
    main()
