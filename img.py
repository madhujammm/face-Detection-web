import streamlit as st
import cv2
import numpy as np

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return image

def main():
    st.set_page_config(
        page_icon=":camera:",
        page_title="Face Detection Fun"
    )

    st.title("🎭 Face Detection Fun 🎉")

    st.markdown("""
    Welcome to the Face Detection Fun page! Upload an image and let's see what we can find! 
    You might discover some surprises along the way. 😄
    """)

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Display the original image
        st.image(image, caption="Original Image", use_column_width=True)

        # Detect faces and display the result
        result_image = detect_faces(image)
        st.image(result_image, caption="Detected Faces", use_column_width=True)

        # Surprise button
        if st.button("🎁 Click for a Surprise!"):
            st.balloons()
            st.success("🎉 Hooray! You found a surprise! 🎈")

if __name__ == "__main__":
    main()
