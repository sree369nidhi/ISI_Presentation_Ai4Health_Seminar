import os
import cv2
import numpy as np

from tqdm import tqdm
import shutil


# print('Imga')
print('\nImage Preprocessing - Optic Disc Localization')

files = [f for f in os.listdir("/content/Dataset_C0GM/data/assets/Image Dataset/URL") if f.endswith('.jpg')]
resultfrac = 0.4
discfrac = 0.35
step = 5
shrink = 512
minbcorner = 0.20
toobrightcorners = 0.9
maxspread = 0.25
maxbrightcorners = 3
toobright = 0.99
minbrightdisc = 0.50

for file in files:
    print(file)
    im = cv2.imread(os.path.join("/content/Dataset_C0GM/data/assets/Image Dataset/URL", file))
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    x, y = g.shape
    mx = max(x, y)
    scale = mx / shrink
    x = int(x / scale)
    y = int(y / scale)
    g = cv2.resize(g, (y, x))
    mn = min(x, y)
    sqsize = int(mn * resultfrac)
    sq2 = int(sqsize / 2)
    discsize = int(sqsize * discfrac)
    d2 = int(discsize / 2)
    margin = int((sqsize - discsize) / 2.0)
    m2 = int(margin / 2)
    midstart = sq2 - m2
    bestx = -1
    besty = -1
    best = -1

    for i in range(0, x - (sqsize + step), step):
        for j in range(discsize * 2, y - (sqsize * 2), step):
            suba = g[i + margin:i + margin + discsize, j + margin:j + margin + discsize]
            tot = np.mean(suba)
            if tot >= best:
                best2 = tot

                c1 = np.mean(g[i + midstart:i + midstart + d2, j:j + d2])
                c2 = np.mean(g[i + midstart:i + midstart + d2, j + (sqsize - d2):j + sqsize])
                c3 = np.mean(g[i:i + d2, j + midstart:j + d2 + midstart])
                c4 = np.mean(g[i + (sqsize - d2):i + sqsize, j + midstart:j + d2 + midstart])
                c5 = np.mean(g[i:i + d2, j:j + d2])
                c6 = np.mean(g[i:i + d2, j + (sqsize - d2):j + sqsize])
                c7 = np.mean(g[i + (sqsize - d2):i + sqsize, j:j + d2])
                c8 = np.mean(g[i + (sqsize - d2):i + sqsize, j + (sqsize - d2):j + sqsize])

                mx = max([c1, c2, c3, c4, c5, c6, c7, c8])
                mn = min([c1, c2, c3, c4, c5, c6, c7, c8])
                av = np.mean([c1, c2, c3, c4, c5, c6, c7, c8])
                brightcount = sum([1 for c in [c1, c2, c3, c4, c5, c6, c7, c8] if c > tot * toobright])

                if mn > minbcorner and tot > minbrightdisc and tot * toobrightcorners > av and brightcount < maxbrightcorners:
                    bestx = i
                    besty = j
                    best = tot

    if bestx > 0:
        i = bestx
        j = besty
        c1 = int(100 * np.mean(g[i + midstart:i + midstart + d2, j:j + d2]))
        c2 = int(100 * np.mean(g[i + midstart:i + midstart + d2, j + (sqsize - d2):j + sqsize]))
        c3 = int(100 * np.mean(g[i:i + d2, j + midstart:j + d2 + midstart]))
        c4 = int(100 * np.mean(g[i + (sqsize - d2):i + sqsize, j + midstart:j + d2 + midstart]))
        c5 = int(100 * np.mean(g[i:i + d2, j:j + d2]))
        c6 = int(100 * np.mean(g[i:i + d2, j + (sqsize - d2):j + sqsize]))
        c7 = int(100 * np.mean(g[i + (sqsize - d2):i + sqsize, j:j + d2]))
        c8 = int(100 * np.mean(g[i + (sqsize - d2):i + sqsize, j + (sqsize - d2):j + sqsize]))

        st = str(int(100 * best)) + "-" + str(c1) + "-" + str(c2) + "-" + str(c3) + "-" + str(c4) + "-" \
            + "-" + str(c5) + "-" + str(c6) + "-" + str(c7) + "-" + str(c8) + "-"
        scaledx = int(bestx * scale)
        scaledy = int(besty * scale)
        scaledxx = int((bestx + sqsize) * scale)
        scaledyy = int((besty + sqsize) * scale)
        bestdisc = im[scaledx:scaledxx, scaledy:scaledyy, :]
        # cv2.imshow("Best Disc", bestdisc)
        # cv2.waitKey(0)
        # cv2.imwrite("../Suspect/cropped2/c_" + file, bestdisc)
            # Right before saving the images
        output_directory = "/content/Dataset_C0GM/data/assets/Image Dataset/URL_Cropped/"
        os.makedirs(output_directory, exist_ok=True)

        # bestxy.save("../Suspect/cropped2/c_" + file)
        # bestxy.resize((224, 224)).save("../Suspect/cropped2/224c_" + file)
        cv2.imwrite("/content/Dataset_C0GM/data/assets/Image Dataset/URL_Cropped/" + file, cv2.resize(bestdisc, (224, 224)))
        # cv2.imwrite("../Suspect/cropped2/299c_" + file, cv2.resize(bestdisc, (299, 299)))
    else:
        print('first algo failed')
        # Read the image and convert it to grayscale
        img = cv2.imread(os.path.join("/content/Dataset_C0GM/data/assets/Image Dataset/URL", file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # print('hi')
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Apply the Hough Circle Transform to find the optic disc
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
        if circles is None:
            print('second algo failed')

        # Get the circle with the highest score
        circle = circles[0][0]
        x, y, radius = int(circle[0]), int(circle[1]), int(circle[2])

        # Create a mask for the circular area
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), radius, 255, -1)
        
        # Apply the mask to the original image and crop the circular area
        cropped_img = cv2.bitwise_and(img, img, mask=mask)
        x1, y1 = x - radius, y - radius
        x2, y2 = x + radius, y + radius
        cropped_img = cropped_img[y1:y2, x1:x2]

        cv2.imwrite("/content/Dataset_C0GM/data/assets/Image Dataset/URL_Cropped/" + file, cv2.resize(cropped_img, (224, 224)))

# ! git clone https://github.com/sree369nidhi/ISI_Presentation_Ai4Health_Seminar

print('\nPredicting the Test data')

import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import preprocess_input
 
path = '/content/ISI_Presentation_Ai4Health_Seminar/model_weights.keras'

rescale = tf.keras.layers.Rescaling(1./255.0)
preprocess_input = tf.keras.applications.vgg19.preprocess_input

# Create the base model from the pre-trained model MobileNet V2
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                            include_top=False,
                                            weights='imagenet')

base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 17

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False


intermediate_layer_1 = tf.keras.layers.Dense(128, activation='relu')
intermediate_layer_2 = tf.keras.layers.Dense(128, activation='relu')
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid') #)


inputs = tf.keras.Input(shape=(224, 224,  3))
# x = data_augmentation(inputs)
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = intermediate_layer_1(x, )#activation='relu')
x = tf.keras.layers.Dropout(0.2)(x)
x = intermediate_layer_2(x)#, activation='relu')
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


# Set learning rates for layers
new_layers_lr = 1e-4
other_layers_lr = 1e-5

optimizer = tf.keras.optimizers.Adam(learning_rate=other_layers_lr)

# Update learning rates for new layers
for layer in [intermediate_layer_1, intermediate_layer_2, prediction_layer]:
    layer_optimizer = tf.keras.optimizers.Adam(learning_rate=new_layers_lr)
    layer.trainable_weights[0].optimizer = layer_optimizer


model.compile(optimizer=optimizer, loss=    tf.keras.losses.BinaryFocalCrossentropy(from_logits=False), metrics=['accuracy', 'AUC'])

model.load_weights(path)

# model = tf.keras.models.load_model("model.keras")


import json

# Load the JSON file
with open('/content/Diagnosis.json', 'r') as file:
    metadata = json.load(file)

# Iterate over each image in the metadata
for entry in metadata:
    filename = entry['Filename']
    image_path = os.path.join('/content/Dataset_C0GM/data/assets/Image Dataset/URL_Cropped', filename)

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, IMG_SIZE)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Predict on the image
    prediction = model.predict(image)[0][0]

    # Update the Image_Output field
    if prediction >= 0.65:
        entry['Diagnosis_Vocab'] = 'C0FW'
    else:
        entry['Diagnosis_Vocab'] = 'C0FY'

# Save the updated JSON file
with open('/content/Diagnosis.json', 'w') as file:
    json.dump(metadata, file)

print('\nUpdated JSON file: Diagnosis.json')
