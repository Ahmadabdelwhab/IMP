import base64
from io import BytesIO
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify , render_template , send_file
import cv2

app = Flask(__name__)


def base64_to_img(img_64):
    '''convert image from numpy array to image png format'''
    #convert from base64 to image
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
def canny(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    return edges
filters = {
    "canny":canny
}
@app.route("/")
def home():
    '''renders template'''
    return render_template("index.html")








@app.route('/upload', methods=['POST'])
def upload_image():
    '''apply filters and return image'''
    try:
        data = request.get_json()
        if 'image' in data: 
            img_64 = data['image']
            img = base64_to_img(img_64)
            filter = filters[data["filter"]]
            edited_img = filter(img)
            img_bytes = process_for_upload(edited_img)
            # Send the image in the response
            return send_file(BytesIO(img_bytes), mimetype='image/png')
            
        return jsonify({'error': 'No image data provided.'}), 400
    except Exception as e:
        print(str(e))
        return jsonify({'error': f'Error processing the image: {str(e)}'}), 500











if __name__ == '__main__':
    app.run(debug=True)
