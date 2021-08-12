# SOURCES:
# https://learnopencv.com/age-gender-classification-using-opencv-deep-learning-c-python/, model loading and usage code taken from there
# https://discuss.streamlit.io/t/remove-made-with-streamlit-from-bottom-of-app/1370/2,
# Hiding the hamburger menu and watermark

import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def get_face_box(net, frame, conf_threshold=0.7):
    opencv_dnn_frame = frame.copy()
    frame_height = opencv_dnn_frame.shape[0]
    frame_width = opencv_dnn_frame.shape[1]
    blob_img = cv2.dnn.blobFromImage(opencv_dnn_frame, 1.0, (300, 300), [
        104, 117, 123], True, False)

    net.setInput(blob_img)
    detections = net.forward()
    b_boxes_detect = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            b_boxes_detect.append([x1, y1, x2, y2])
            cv2.rectangle(opencv_dnn_frame, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frame_height / 150)), 8)
    return opencv_dnn_frame, b_boxes_detect


st.write("""
    # Age and Gender prediction
    """)

st.write("## Upload a picture that contains a face")

uploaded_file = st.file_uploader("Choose a file:")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    cap = np.array(image)
    cv2.imwrite('temp.jpg', cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)))
    cap=cv2.imread('temp.jpg')

    face_txt_path="opencv_face_detector.pbtxt"
    face_model_path="opencv_face_detector_uint8.pb"

    age_txt_path="age_deploy.prototxt"
    age_model_path="age_net.caffemodel"

    gender_txt_path="gender_deploy.prototxt"
    gender_model_path="gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    age_classes=['Age: ~1-2', 'Age: ~3-5', 'Age: ~6-14', 'Age: ~16-22',
                   'Age: ~25-30', 'Age: ~32-40', 'Age: ~45-50', 'Age: age is greater than 60']
    gender_classes = ['Gender:Male', 'Gender:Female']

    age_net = cv2.dnn.readNet(age_model_path, age_txt_path)
    gender_net = cv2.dnn.readNet(gender_model_path, gender_txt_path)
    face_net = cv2.dnn.readNet(face_model_path, face_txt_path)

    padding = 20
    t = time.time()
    frameFace, b_boxes = get_face_box(face_net, cap)
    if not b_boxes:
        st.write("No face Detected, Checking next frame")

    for bbox in b_boxes:
        face = cap[max(0, bbox[1] -
                       padding): min(bbox[3] +
                                    padding, cap.shape[0] -
                                    1), max(0, bbox[0] -
                                            padding): min(bbox[2] +
                                                          padding, cap.shape[1] -
                                                          1)]

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_pred_list = gender_net.forward()
        gender = gender_classes[gender_pred_list[0].argmax()]
        st.write(
            f"Gender : {gender}, confidence = {gender_pred_list[0].max() * 100}%")

        age_net.setInput(blob)
        age_pred_list = age_net.forward()
        age = age_classes[age_pred_list[0].argmax()]
        st.write(f"Age : {age}, confidence = {age_pred_list[0].max() * 100}%")

        label = "{},{}".format(gender, age)
        cv2.putText(
            frameFace,
            label,
            (bbox[0],
             bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,
             255,
             255),
            2,
            cv2.LINE_AA)
        st.image(frameFace)
