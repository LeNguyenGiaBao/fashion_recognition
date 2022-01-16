# https://github.com/kb22/Color-Identification-using-Machine-Learning
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import numpy as np 
import cv2 
from collections import Counter
import webcolors

def get_color(image, number_of_colors):
    h, w, c = image.shape
    modified_image = image.reshape(w*h, 3)
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    max_color_index = max(counts, key=counts.get)
    center_colors = clf.cluster_centers_
    color = center_colors[max_color_index]
    rms_lst = []
    for img_clr, img_hex in webcolors.CSS3_NAMES_TO_HEX.items():
        cur_clr = webcolors.hex_to_rgb(img_hex)
        rmse = np.sqrt(mean_squared_error(color, cur_clr))
        rms_lst.append(rmse)
    closest_color = rms_lst.index(min(rms_lst))
    color_name = list(webcolors.CSS3_NAMES_TO_HEX.items())[closest_color][0]

    return color_name

if __name__ == "__main__":
    img = cv2.imread('../image/1_crop.jpg')
    print(img.shape)

    rgb = get_color(img, 4)
    print(rgb)
