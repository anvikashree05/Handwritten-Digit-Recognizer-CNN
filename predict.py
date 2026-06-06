import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model("model.keras")

img = Image.open("image1.png").convert("L")

img = img.resize((28,28))

img = np.array(img)

img = 255 - img

# Show what the model sees
plt.imshow(img, cmap="gray")
plt.show()

img = img / 255.0

img = img.reshape(1,28,28,1)

prediction = model.predict(img)

print("Prediction Probabilities:")
print(prediction)

digit = np.argmax(prediction)

print("Predicted Digit:", digit)