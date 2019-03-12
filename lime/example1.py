
import os
import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications import vgg16 #import VGG16
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
print('Notebook run using keras:', keras.__version__)
import cv2

#inet_model = inc_net.InceptionV3()
inet_model=vgg16.VGG16()

def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(224, 224))
        #img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        #x = inc_net.preprocess_input(x)
        x = vgg16.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

images = transform_img_fn([os.path.join('data','origin-0.png')])
plt.imshow(images[0] / 2 + 0.5)
#plt.show()
#plt.savefig('origin.png',bbox_inches='tight',transparent=True, pad_inches=0)
preds = inet_model.predict(images)
for x in decode_predictions(preds)[0]:
    print(x)

import os,sys
#try:
import lime
#except:
#    sys.path.append(os.path.join('..', '..')) # add the current directory
#    import lime
from lime import lime_image

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(images[0], inet_model.predict, top_labels=5, hide_color=0, num_samples=1000)
from skimage.segmentation import mark_boundaries
temp, mask = explanation.get_image_and_mask(920, positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.savefig("5-features.pdf", bbox_inches="tight")
#cv2.imwrite("5-features.png", mark_boundaries(temp / 2 + 0.5, mask))
#cv2.imwrite("5b-features.png", 255.0*mark_boundaries(temp / 2 + 0.5, mask))

temp, mask = explanation.get_image_and_mask(920, positive_only=True, num_features=10, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.savefig("10-features.pdf",bbox_inches="tight")
#cv2.imwrite("10-features.png", mark_boundaries(temp / 2 + 0.5, mask))
#cv2.imwrite("10b-features.png", 255.0*mark_boundaries(temp / 2 + 0.5, mask))


temp, mask = explanation.get_image_and_mask(920, positive_only=True, num_features=20, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.savefig("20-features.pdf",bbox_inches="tight")
#cv2.imwrite("20-features.png", mark_boundaries(temp / 2 + 0.5, mask))
#cv2.imwrite("20b-features.png", 255.0*mark_boundaries(temp / 2 + 0.5, mask))
