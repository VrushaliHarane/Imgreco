import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
import glob
import pickle
from datetime import datetime




fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in glob.glob("static/feature/*"):
    features.append(pickle.load(open(feature_path, 'rb')))
    img_paths.append('static/img/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')


        #upload query image in file to save in static/uploaded or else directly save der and extract from der
        file = request.files['query_img']

        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat() + "_" + file.filename
        img.save(uploaded_img_path)

        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)  # Do search
        ids = np.argsort(dists)[:10] # Top 10 results
        scores = [(dists[id], img_paths[id]) for id in ids]

     
    
