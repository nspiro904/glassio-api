from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from predict import predict_face_shape  # Ensure this is defined in predict.py
from glasses_recommend import recommend_glasses  # Ensure this is defined in glasses_recommend.py
from glasses_overlay import overlay_glasses  # Ensure this is defined in glasses_overlay.py
import numpy as np
import cv2
import base64

# Create the FastAPI app instance
app = FastAPI()

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to Glassio API!"}

# Predict route
@app.post("/predict")
async def predict_shape(file: UploadFile = File(...)):
    # Save the uploaded image temporarily
    image = await file.read()
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(image)

    # Get face shape prediction
    face_shape = predict_face_shape(image_path)
    glasses_options = recommend_glasses(face_shape)
    
    return JSONResponse(content={
        "message": "Face shape identified",
        "face_shape": face_shape,
        "glasses_options": glasses_options
    })

# Overlay route
@app.post("/overlay")
async def overlay_glasses_on_image(file: UploadFile = File(...), glass_type: str = "style1"):
    
       # Read the uploaded image directly into memory
    image_bytes = await file.read()
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Get face shape prediction and landmarks (using image directly)
    landmarks = predict_face_shape(image, return_landmarks=True)
    glasses_image = f"glasses/{glass_type}.png"  # Choose the style based on glass_type
    
    # Apply glasses overlay
    result_image = overlay_glasses(image, landmarks, glasses_image)
    
    # Encode the result image to base64
    _, encoded_image = cv2.imencode('.png', result_image)
    base64_image = base64.b64encode(encoded_image).decode('utf-8')
    
    return JSONResponse(content={
        "message": "Glasses overlay applied successfully",
        "result_image": f"data:image/png;base64,{base64_image}"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# @app.post("/overlay")
# async def overlay_glasses_on_image(file: UploadFile = File(...), glass_type: str = "style1"):
    ## Save the uploaded image temporarily
    # image = await file.read()
    # image_path = "temp_image.jpg"
    # with open(image_path, "wb") as f:
        # f.write(image)
    # 
    ## Get face shape prediction and landmarks
    # landmarks = predict_face_shape(image_path, return_landmarks=True)
    # glasses_image = f"glasses/{glass_type}.png"  # Choose the style based on glass_type
    # 
    ##Apply glasses overlay
    # result_image = overlay_glasses(image_path, landmarks, glasses_image)
    # 
    ##Save result image and return response
    # result_image_path = "result_with_glasses.png"
    # cv2.imwrite(result_image_path, result_image)
    # 
    # return JSONResponse(content={
        # "message": "Glasses overlay applied successfully",
        # "result_image": result_image_path
    # })
# 