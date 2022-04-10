# importing required libraries
from re import A, S
import streamlit as st
import streamlit.components.v1 as components
import cv2
import logging as log # used for releasing log messages from the appication
import datetime as dt
from time import sleep


cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log', level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

# Streamlit


# Streamlit Title
st.title('Face Detection using OpenCV on Streamlit')

# creating a Frame using HTML
components.html('<html> <body> <h3>Live Video Face Detection</h3></body></html>',
                               width=700, height=100)

# Main Details
st.sidebar.subheader('Details of the person')
t1 = st.sidebar.text_input('Name of the Person')
s1 = st.sidebar.slider('Age of the Person')
How_is_streamlit = st.sidebar.selectbox('GENDER ', ['---Select Gender', 'MALE', 'FEMALE'])

st.sidebar.markdown("---")

st.write('Name: ', t1)
st.write('Age: ', s1)
st.write('GENDER: ', How_is_streamlit)

st.markdown(f'<hr style="height:2px;border:none;color:#333;background-color:#333;" />',
                      unsafe_allow_html=True)

# main working of face detection
st.header("Haar Cascade")

st.write('Click Below to Instantiate')
st.write("Enter 'q' to Exit")

if st.button("START"):
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if anterior != len(faces):
            anterior = len(faces)
            log.info("faces: " + str(len(faces)) + " at " + str(dt.datetime.now()))

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Display the resulting frame
        cv2.imshow('Video', frame)
    
st.markdown('<html> <I><b><h5>Tweaked by Prakhar :(</h5></I> <br> </html>', 
                        unsafe_allow_html=True)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()