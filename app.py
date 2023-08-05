from pathlib import Path
import streamlit as st
import PIL
from PIL import Image
from ultralytics import YOLO

thermal_model_path = 'weights/best.pt'
flight_model_path = 'weights/best (1).pt'
thermal_img_path = 'Images/thermal.jpg'
flight_img_path = 'Images/flight.jpg'

# setting page layout
st.set_page_config(
    page_title="Object Detection Using YOlov8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )


with st.sidebar:
    st.header("ML Model Config")

    model_type = st.radio(
    "Select Task", ['Number of flight Detection', 'Thermal Object Detection'])

    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    
    st.subheader("Or")
    check = st.checkbox('Use demo image')
    
    confidence = float(st.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

def thermal():
    # Creating main page heading
    st.title("Thermal Image Object Detection")
    st.caption('This Object Detection predicts whether the image has :blue[Dog or Human] .')
    st.caption('Upload a thermal Image with dogs and human and hit the :blue[Detect Objects] button to check the result.')
    # Creating two columns on the main page
    # Creating two columns on the main page
    col1, col2 = st.columns(2)

    # Adding image to the first column if image is uploaded
    with col1:
        if check:
            uploaded_image = Image.open(thermal_img_path)
            st.image(thermal_img_path,
                    caption="Uploaded Image",
                    use_column_width=True
                    )

        elif source_img:
            # Opening the uploaded image
            uploaded_image = PIL.Image.open(source_img)
            # Adding the uploaded image to the page with a caption
            st.image(source_img,
                    caption="Uploaded Image",
                    use_column_width=True
                    )

    try:
        model = YOLO(thermal_model_path)
    except Exception as ex:
        st.error(
            f"Unable to load model. Check the specified path: {thermal_model_path}")
        st.error(ex)

    if st.sidebar.button('Detect Objects'):
        res = model.predict(uploaded_image,conf=confidence)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        with col2:
            st.image(res_plotted,
                    caption='Detected Image',
                    use_column_width=True
                    )
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.xywh)
            except Exception as ex:
                st.write("No image is uploaded yet!")

def flight():
    st.title("Number of Flight Detection")
    st.caption('This model is trained to spot the number of flights from the :blue[aerial view].')
    st.caption('Upload a Aerial view Image and hit the :blue[Detect Objects] button to check the result.')
    # Creating two columns on the main page
    # Creating two columns on the main page
    col1, col2 = st.columns(2)

    # Adding image to the first column if image is uploaded
    with col1:
        if check:
            uploaded_image = Image.open(flight_img_path)
            st.image(flight_img_path,
                    caption="Uploaded Image",
                    use_column_width=True
                    )

        elif source_img:
            # Opening the uploaded image
            uploaded_image = PIL.Image.open(source_img)
            # Adding the uploaded image to the page with a caption
            st.image(source_img,
                    caption="Uploaded Image",
                    use_column_width=True
                    )

    try:
        model = YOLO(flight_model_path)
    except Exception as ex:
        st.error(
            f"Unable to load model. Check the specified path: {flight_model_path}")
        st.error(ex)

    if st.sidebar.button('Detect Objects'):
        res = model.predict(uploaded_image,conf=confidence)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        with col2:
            st.image(res_plotted,
                    caption='Detected Image',
                    use_column_width=True
                    )
            try:
                count =  0 
                with st.expander("Detection Results"):
                    for box in boxes:
                        #st.write(box.xywh)
                        count += 1
                    st.write("Number of Airplanes found are" , count)
            except Exception as ex:
                st.write("No image is uploaded yet!")





if model_type == 'Number of flight Detection':
    flight()
elif model_type == 'Thermal Object Detection':
    thermal()
