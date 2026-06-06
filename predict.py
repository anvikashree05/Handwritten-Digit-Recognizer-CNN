import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("model.keras")

img = Image.open("digit.png").convert("L")

img = img.resize((28,28))

img = np.array(img)

img = 255 - img

img = img / 255.0

img = img.reshape(1,28,28,1)

prediction = model.predict(img)

digit = np.argmax(prediction)

print("Predicted Digit:", digit)