import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import logging
import streamlit as st
import os
import zipfile

def write_model(intercept, coefs, model_type):
    coefficients_string = '%s' % (','.join([str(x) for x in coefs[0]]))
    file_content = \
'''
#include "%s"
// clang-format off
const double %s_coefficients[%d] = {%s};
double %s_intercept = %lf;
// clang-format on
''' % (f'{model_type}_model.h', model_type, len(coefs[0]), coefficients_string, model_type, intercept)
    file_name = f'{model_type}_model.cpp'
    # file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'generated', file_name)
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'generated', file_name)

    with open(file_path, 'w') as f:
        f.write(file_content)

def print_dataset_info(data):
    x, y = data
    first_class_ids = [idx[0] for idx in enumerate(y) if idx[1] == 0]
    second_class_ids = [idx[0] for idx in enumerate(y) if idx[1] == 1]
    x_first_class = list()
    x_second_class = list()
    
    for i, x_data in enumerate(x):
        if i in first_class_ids:
            x_first_class.append(x_data.tolist())
        elif i in second_class_ids:
            x_second_class.append(x_data.tolist())
    second_class_np = np.array(x_second_class)
    first_class_np = np.array(x_first_class)

    st.divider()
    st.code(f'''
    1st Class Dataset Info:
    Mean:
    {np.mean(first_class_np, axis=0)}

    2nd Class Dataset Info:
    Mean:
    {np.mean(second_class_np, axis=0)}
    ''')

def train_regression(data, metric:str='neural'):
    model = LogisticRegression(solver='liblinear', max_iter=4000,
                                penalty='l2', random_state=2, fit_intercept=True, intercept_scaling=0.2)
    with st.status('Logistic Regression'):
        scores = cross_val_score(model, data[0], data[1], cv=5, scoring='f1_macro', n_jobs=8)
        st.code('f1 macro %s' % str(scores))
        model.fit(data[0], data[1])
        st.write('Model Fit!')
        st.code(f'model.intercept_: {model.intercept_}')
        st.code(f'model.coef_: {model.coef_}')
        initial_type = [(f'{metric}_input', FloatTensorType([1, 5]))]
        onx = convert_sklearn(model, initial_types=initial_type, target_opset=11, options={type(model): {'zipmap': False}})
        st.write('Converted to ONNX format.')
        with open('./models/logreg_model.onnx', 'wb') as f:
            f.write(onx.SerializeToString())
        write_model(model.intercept_, model.coef_, f'{metric}')

def train_svm(data, metric:str='neural'):
    model = SVC(kernel='linear', verbose=True, random_state=1, class_weight='balanced', probability=True)
    with st.status('SVM'):
        scores = cross_val_score(model, data[0], data[1], cv=5, scoring='f1_macro', n_jobs=8)
        st.code('f1 macro %s' % str(scores))
        model.fit(data[0], data[1])
        st.write('Model Fit!')
        initial_type = [(f'{metric}_input', FloatTensorType([1, 5]))]
        onx = convert_sklearn(model, initial_types=initial_type, target_opset=11, options={type(model): {'zipmap': False}})
        st.write('Converted to ONNX format.')
        with open('./models/svm_model.onnx', 'wb') as f:
            f.write(onx.SerializeToString())

def train_random_forest(data, metric:str='neural'):
    model = RandomForestClassifier(class_weight='balanced', random_state=1, n_jobs=15, n_estimators=200)
    with st.status('Random Forest'):
        scores = cross_val_score(model, data[0], data[1], cv=5, scoring='f1_macro', n_jobs=15)
        st.code('f1 macro %s' % str(scores))
        model.fit(data[0], data[1])
        st.write('Model Fit!')
        initial_type = [(f'{metric}_input', FloatTensorType([1, 5]))]
        onx = convert_sklearn(model, initial_types=initial_type, target_opset=11, options={type(model): {'zipmap': False}})
        st.write('Converted to ONNX format.')
        with open('./models/forest_model.onnx', 'wb') as f:
            f.write(onx.SerializeToString())

def train_knn(data, metric:str='neural'):
    model = KNeighborsClassifier(n_neighbors=10, n_jobs=8)
    with st.status('KNN'):
        scores = cross_val_score(model, data[0], data[1], cv=5, scoring='f1_macro', n_jobs=15)
        st.code('f1 macro %s' % str(scores))
        model.fit(data[0], data[1])
        st.write('Model Fit!')
        initial_type = [(f'{metric}_input', FloatTensorType([1, 5]))]
        onx = convert_sklearn(model, initial_types=initial_type, target_opset=11, options={type(model): {'zipmap': False}})
        st.write('Converted to ONNX format.')
        with open('./models/knn_model.onnx', 'wb') as f:
            f.write(onx.SerializeToString())

def train_mlp(data, metric:str='neural'):
    model = MLPClassifier(hidden_layer_sizes=(100, 20),learning_rate='adaptive', max_iter=1000,
                          random_state=1, verbose=True, activation='logistic', solver='adam')
    with st.status('MLP'):
        scores = cross_val_score(model, data[0], data[1], cv=5, scoring='f1_macro', n_jobs=15)
        st.code('f1 macro %s' % str(scores))
        model.fit(data[0], data[1])
        st.write('Model Fit!')
        initial_type = [(f'{metric}_input', FloatTensorType([1, 5]))]
        onx = convert_sklearn(model, initial_types=initial_type, target_opset=11, options={type(model): {'zipmap': False}})
        st.write('Converted to ONNX format.')
        with open('./models/mlp_model.onnx', 'wb') as f:
            f.write(onx.SerializeToString())

def train_stacking_classifier(data, metric:str='neural'):
    model1 = MLPClassifier(hidden_layer_sizes=(100, 20),learning_rate='adaptive', max_iter=1000,
                          random_state=1, verbose=True, activation='logistic', solver='adam')
    model2 = KNeighborsClassifier(n_neighbors=10, n_jobs=8)
    model3 = RandomForestClassifier(class_weight='balanced', random_state=1, n_jobs=8, n_estimators=200)
    with st.status('Stacking Classifiers'):
        st.write('Built MLP Classifier')
        st.write('Built KNN Classifier')
        st.write('Built Random Forest Classifier Classifier')

        meta_model = LogisticRegression()
        sclf = StackingClassifier(estimators=[('MLPClassifier', model1), ('KNeighborsClassifier', model2), ('RandomForestClassifier', model3)],
                                final_estimator=meta_model, n_jobs=15,
                                passthrough=True)
        st.info('STACKING')
        scores = cross_val_score(sclf, data[0], data[1], cv=5, scoring='f1_macro', n_jobs=15)
        st.code('f1 macro %s' % str(scores))
        sclf.fit(data[0], data[1])
        st.write('Model Fit!')
        initial_type = [(f'{metric}_input', FloatTensorType([1, 5]))]
        onx = convert_sklearn(sclf, initial_types=initial_type, target_opset=11, options={type(sclf): {'zipmap': False}})
        st.write('Converted to ONNX format.')
        with open('./models/stacking_model.onnx', 'wb') as f:
            f.write(onx.SerializeToString())

# Download Helper
def download_helper(label, buttonlabel, datatype, filepath):
    col1, col2 = st.columns(2)
    with col1: 
        st.write(label)
    with col2:
        with open(filepath, "rb") as file:
            st.download_button(label=buttonlabel, data=file, file_name=filepath)

def zip_folder(source_folder, output_file):
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_folder):
            for file in files:
                zipf.write(os.path.join(root, file))