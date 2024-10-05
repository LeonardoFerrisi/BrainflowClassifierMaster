import streamlit as st
import yaml
import os
import zipfile
from silverhand_helpers import train_regression, train_svm, train_knn, train_random_forest, train_mlp, train_stacking_classifier, print_dataset_info, download_helper, zip_folder
import pickle

# Config parse script

with open('config.yaml', 'r') as file:
    # Parse the YAML content
    yaml_data = yaml.safe_load(file)

    use_fileuploader = yaml_data['use_file_uploader']
    device_options   = yaml_data['device_options']
    metrics_quantity = yaml_data['metrics_quantity']
    ignore_paths     = yaml_data['to_ignore']
    dataset_path     = os.path.join(os.path.dirname(__file__), 
                                  yaml_data['dataset_location']) 
    # check if datset path exists
        
#

st.title('SILVERHAND')
st.info('Please ensure that you have already generated datasets')

local_datasets = {}

dataset_uploaded = False

st.divider()

if not use_fileuploader:
    reuse_dataset_toggle = st.checkbox("Reuse Datasets?", value=False)
    if reuse_dataset_toggle:
        available_folders = [path for path in os.listdir(os.path.dirname(__file__)) if (os.path.isdir(path) and path[0]!='.' and path not in ignore_paths)]
        dataset_path = st.selectbox(label="dataset path", 
                                    options=available_folders)
        if not os.path.isdir(dataset_path):
            st.error('Provided path does not exist! Please create a folder at this location or use a different path.')
        elif len(os.listdir(dataset_path)) == 0:
            st.warning('Select folder has no contents! Please provide them or use another folder.')
        else:
            # Parse Datasets
            # TODO: Verify that all files are pickle files
            dataset_path_full = os.path.join(os.path.dirname(__file__), dataset_path)

            st.write('Dataset Files:')
            for i, datafile in enumerate(os.listdir(dataset_path_full)):
                datafile_path_full = os.path.join(dataset_path_full, datafile)
                st.write(datafile_path_full)
                with open(datafile_path_full, 'rb') as f:
                    local_datasets[f'dataset_{i}'] = pickle.load(f)
            # dataset_uploaded = True 
    else:
        st.warning("PREPARE data is not ready yet!")
else:
    uploaded_file = st.file_uploader("Choose a Dataset File", type="zip", key="dataset_uploader")
    st.info("Accepts .zip files.")
    if uploaded_file is not None:
        try:
            # Extract the ZIP file
            with zipfile.ZipFile(uploaded_file) as zip_ref:
                zip_ref.extractall("./extracted_files")

            # Display extracted files
            st.success("Files successfully extracted!")

            extracted_folderpath = f'./extracted_files/{uploaded_file.name.split(".")[0]}'

            # List extracted files  
            extracted_files = os.listdir(extracted_folderpath)
            st.subheader("Extracted Files:")
            for file in extracted_files:
                st.markdown(f"- {file}")
            
            # Parse Datasets
            for i, datafile in enumerate(os.listdir(dataset_path)):
                local_datasets[f'dataset_{i}'] = datafile
            # dataset_uploaded = True 

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# if dataset_uploaded:
if len(local_datasets) > 0:

    st.divider()

    st.info('[INFO] Data prepared')
    
    if metrics_quantity>1:
        st.error('Current version can only handle a metric quantity of 1')
    else:
        metric_label = st.text_input(label='Please enter a label for your metric: ',
                                        value='focus')
        if len(local_datasets) == 2:

            st.divider()
            st.write('Select the models you want to train:')

            use_regession           = st.checkbox("Use Regression?", value=False)
            use_svm                 = st.checkbox("Use SVM?", value=False)
            use_knn                 = st.checkbox("Use KNN?", value=False)
            use_random_forest       = st.checkbox("Use Random Forest?", value=False)
            use_mlp                 = st.checkbox("Use MLP?", value=False)
            use_stacking_classifier = st.checkbox("Use Stacking Classifier?", value=False) 

            data = [local_datasets[key] for key, value in local_datasets.items()]

            if st.button(label='TRAIN DATASETS'):

                print_dataset_info(data)
                if use_regession: train_regression(data, metric_label)
                if use_svm: train_svm(data, metric_label)
                if use_knn: train_knn(data, metric_label)
                if use_random_forest: train_random_forest(data, metric_label)
                if use_mlp: train_mlp(data, metric_label)
                if use_stacking_classifier: train_stacking_classifier(data, metric_label)

                # Zip all models
                source_folder = './models'
                output_file = './export/models.zip'
                zip_folder(source_folder, output_file)
                st.info(f"Folder {source_folder} has been successfully zipped to {output_file}")


                st.divider()
                st.write('Download Models: ')
                with open(output_file, "rb") as all_models:
                    all_models_bytes = all_models.read()
                    st.download_button(
                        label="Download all models",
                        data=all_models_bytes,
                        file_name="all_models.zip",
                    )

                # if use_regession: download_helper(label='Logistic Regression', buttonlabel='Logistic Regression', datatype='.onnx', filepath='./models/logreg_model.onnx')
                # if use_svm: download_helper(label='SVM', buttonlabel='SVM', datatype='.onnx', filepath='./models/svm_model.onnx')
                # if use_knn: download_helper(label='KNN', buttonlabel='KNN', datatype='.onnx', filepath='./models/knn_model.onnx')
                # if use_random_forest: download_helper(label='Random Forest', buttonlabel='Random Forest', datatype='.onnx', filepath='./models/forest_model.onnx')
                # if use_mlp: download_helper(label='MLP', buttonlabel='MLP', datatype='.onnx', filepath='./models/mlp_model.onnx')
                # if use_stacking_classifier: download_helper(label='Stacking Classifier (MLP + KNN + Random Forest)', buttonlabel='Stacking Classifier', datatype='.onnx', filepath='./models/stacking_model.onnx')

        else:
            st.error('This application is currently only designed to be trained on single features (focus, mindfullness, etc...)')

    





    





