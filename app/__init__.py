import base64
from io import BytesIO
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder='templates')
@app.before_request
def log_request_info():
    app.logger.info('Request: %s %s', request.method, request.url)
    print(f"Request: {request.method} {request.url}")
    app.logger.debug('Headers: %s', request.headers)
    app.logger.debug('Body: %s', request.get_data())

def base64_to_img(img_64):
    '''convert image from numpy array to image png format'''
    # convert from base64 to image
    image_data = base64.b64decode(img_64)
    # Convert the decoded data to a NumPy array
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    # Decode the NumPy array into an OpenCV image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

 
def process_for_upload(edited_img):
    '''convert images to bytes to send later'''
    _, buffer = cv2.imencode('.png', edited_img)
    # Convert the image buffer to bytes
    image_bytes = buffer.tobytes()
    return image_bytes
# filters to apply


def canny(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def sobel(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # Compute the magnitude of the gradient
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    # Convert the magnitude values to uint8
    magnitude_uint8 = np.uint8(magnitude)
    return magnitude_uint8


def perwitt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    perwitt_kernel_x = np.array([[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]])
    edges_x = cv2.filter2D(img, -1, perwitt_kernel_x)
# Apply Perwitt edge detection for vertical edges
    perwitt_kernel_y = np.array([[-1, -1, -1],
                                [0, 0, 0],
                                [1, 1, 1]])
    edges_y = cv2.filter2D(img, -1, perwitt_kernel_y)
    # Combine the horizontal and vertical edges
    edges_x = edges_x.astype(np.float64)
    edges_y = edges_y.astype(np.float64)
    edges = cv2.magnitude(edges_x, edges_y)
    return edges
def laplacian_of_gaussian(img):
    # Load an image
    image = img

    # Apply Gaussian blur
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Laplacian of Gaussian
    laplacian = cv2.Laplacian(image_blurred, cv2.CV_64F)

    # Convert the result to uint8
    laplacian = np.uint8(np.absolute(laplacian))

    return laplacian


def segment(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Flatten the image into a 2D array of pixels
    pixels = image_rgb.reshape((-1, 3))
    # Convert to float32
    pixels = np.float32(pixels)
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3  # Number of clusters
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Convert back to 8-bit values
    centers = np.uint8(centers)
    # Map the labels to the centers
    segmented_image = centers[labels.flatten()]
    # Reshape back to the original image dimensions
    segmented_image = segmented_image.reshape(image_rgb.shape)
    return segmented_image


def detect_objects(img):
    # Load the pre-trained Haarcascades classifier for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)
    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img
 
def histogram_equalization(img):
    # Convert the image to grayscale if it's not already
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    equalized_img = cv2.equalizeHist(img)

    return equalized_img 
filters = { 
    "canny": canny,
    "sobel": sobel,
    "segment": segment,
    "face-detection": detect_objects, 
    "perwitt":perwitt,
    "hist":histogram_equalization,
    "laplacian_of_gaussian":laplacian_of_gaussian
} 


@app.route("/")
def home():
    '''renders template'''
    print("connected to server")
    return render_template("index.html")

 
@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        data = request.get_json()
        if 'image' in data and 'filter' in data:
            img_64 = data['image']
            img = base64_to_img(img_64)
            filter_to_apply = filters.get(data["filter"])
            
            if filter_to_apply:
                edited_img = filter_to_apply(img)
                img_bytes = process_for_upload(edited_img)
                img_bytes = BytesIO(img_bytes)
                processed_image_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                # Return the result as a JSON response
                print("image served")
                return jsonify({
                    "processed_image": processed_image_base64,
                })
            else:
                print("filter not found")
                return jsonify({'error': 'Invalid filter name.'}), 400
        else:
            return jsonify({'error': 'Invalid request format. Please provide "image" and "filter" in the request.'}), 400
    except Exception as e:
        print(str(e))
        return jsonify({'error': f'Error processing the image: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
