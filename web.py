import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import cv2
import base64
from Yolo import predictImage
from Yolo import predictVideo
from Yolo import getClassifer

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.title("Traffic sign detection app")
# vid = st.button("Video")
# img = st.button("Image")
model = getClassifer()
genre = st.radio(
    "What's the data type ?",
    ('Video', 'Image'))

if genre == 'Video':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.write("")     
    video_file = st.file_uploader("Upload a video", type = ['mp4'])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        vf = cv2.VideoCapture(tfile.name)
        st.video(video_file,start_time=0)
        st.write("")
        st.write("Just a second")
        className,label,path = predictVideo(vf,model)

        video_file = open(path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes, format='video/mp4')
        st.markdown(get_binary_file_downloader_html(path, 'Video'), unsafe_allow_html=True)

else:
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.write("")
    file_up = st.file_uploader("Upload an image")
    if file_up is not None:
        image = Image.open(file_up)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Just a second")

        className,label,im = predictImage(np.array(image),model,False,True)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        st.image(im, caption='Uploaded Image.')

        #st.dataframe(label)
        st.write("Label class", label)
        st.write("Prediction class", className)
        

