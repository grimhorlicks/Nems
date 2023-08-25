#!/usr/bin/env python
# coding: utf-8



import sys

import joblib

sys.modules['sklearn.externals.joblib'] = joblib


def predict(data):
 joblib_knn_model = joblib.load(joblib_file)
 return joblib_knn_model.predict(data)

