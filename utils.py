import os
import urllib.request
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from random import sample

class DataFetch(object):

    def __init__(self):
        self.DOWNLOAD_ROOT = "https://raw.githubusercontent.com/Barcelona-Technology-School/InnoSpark-edir/main/"
        self.DS_PATH = os.path.join("Datasets")

    def _get_csv_path(self, folder, file_name):

        # if not os.path.isdir(folder):
        #     os.makedirs(folder)

        name = os.path.basename(os.getcwd())
        while name!='InnoSpark-edir':
            #print(name)
            if name == 'InnoSpark-edir':
                #path = os.getcwd()
                continue
            else:
                os.chdir('..')
                name = os.path.basename(os.getcwd())

        csv_path = os.path.join(os.getcwd(),'Datasets',folder,file_name)
        # URL = self.DOWNLOAD_ROOT + folder + file_name
        # urllib.request.urlretrieve(URL, csv_path)

        return csv_path

    def _get_file_names(self, source, disease, df_csv, col_filename):
        file_names=[]
        #MESSIDOR
        if source == '8-Messidor':
            df = df_csv.loc[df_csv['adjudicated_dr_grade']!=0].copy()
            file_names = np.array(df.image_id)
            #file_names = df.image_id.values
        #KAGGLE
        if source == 'Kaggle':
            if disease == 'DR':
                left_eye=df_csv[(df_csv['Left-Diagnostic Keywords'].str.contains('diabet') & df_csv['D']==1)][col_filename].values
                right_eye=df_csv[(df_csv['Right-Diagnostic Keywords'].str.contains('diabet') & df_csv['D']==1)][col_filename].values
                file_names = np.concatenate((left_eye,right_eye),axis=0)
            if disease == 'cataracts':
                left_eye=df_csv[(df_csv['Left-Diagnostic Keywords'].str.contains('catar') & df_csv['C']==1)][col_filename].values
                right_eye=df_csv[(df_csv['Right-Diagnostic Keywords'].str.contains('catar') & df_csv['C']==1)][col_filename].values
                file_names = np.concatenate((left_eye,right_eye),axis=0)
            if disease == 'glaucoma':
                left_eye=df_csv[(df_csv['Left-Diagnostic Keywords'].str.contains('glau') & df_csv['G']==1)][col_filename].values
                right_eye=df_csv[(df_csv['Right-Diagnostic Keywords'].str.contains('glau') & df_csv['G']==1)][col_filename].values
                file_names = np.concatenate((left_eye,right_eye),axis=0)
        #KAGGLE2
        if source == 'Kaggle2':
            file_names = np.array(df_csv.name)
        #ORIGA
        if source == 'ORIGA':
            file_names = np.array(df_csv['New names'])

        return file_names

    # def _append_normal(self, file_names):
    #     labels = [1] * len(file_names)
    #     normal=[]
    #     file_names_total=[]
    #     left_eye=df_csv[(df_csv['Left-Diagnostic Keywords'].str.contains('normal') & df_csv['N']==1)]['filename'].values
    #     right_eye=df_csv[(df_csv['Right-Diagnostic Keywords'].str.contains('normal') & df_csv['N']==1)]['filename'].values
    #     normal = np.concatenate((left_eye,right_eye),axis=0)
    #     normale_labels = [0] * len(normal)
    #     labels = np.concatenate((labels,normale_labels),axis=0)
    #     file_names_total = np.concatenate((file_names,normal),axis=0)
    #     return file_names_total, labels

    def _append_normal(self, images, labels):
        #labels = [1] * len(file_names)
        normal=[]
        images_total=[]

        if(images.ndim == 4):
            num, height, width, chann = images.shape
        else:
            num, height, width = images.shape

        folder = 'Kaggle'
        file_name = 'full_df.csv'
        csv_path = os.path.join(os.getcwd(),'Datasets',folder,file_name)
        df_csv = pd.read_csv(csv_path)#READ CSV
        left_eye=df_csv[(df_csv['Left-Diagnostic Keywords'].str.contains('normal') & df_csv['N']==1)]['filename'].values
        right_eye=df_csv[(df_csv['Right-Diagnostic Keywords'].str.contains('normal') & df_csv['N']==1)]['filename'].values
        file_names_normal = np.concatenate((left_eye,right_eye),axis=0)
        #Decide the length of the normal eyes sample
        if len(file_names_normal)>=len(images):
            file_names_normal = sample(file_names_normal.tolist(), len(images))
        #else:
            #file_names_normal = sample(file_names_normal, len(images))
        normal = self._read_images(os.path.join('Datasets','Kaggle','preprocessed_images'), file_names_normal, height, width)
        images_total = np.concatenate((images, normal), axis=0)
        normal_labels = [0] * len(normal)
        labels = np.concatenate((labels,normal_labels),axis=0)

        return images_total, labels

    def _read_images(self, path, file_names, row, col):

        image_data = []

        for image_name in tqdm(file_names):
        #for image_name in file_names:

            try:
                #image = cv2.imread(image_path,cv2.IMREAD_COLOR)
                #image = cv2.imread(os.path.join(path,image_name))
                #print(os.path.join(path,image_name))
                image = cv2.imread(os.path.join(path,image_name))
                image = cv2.resize(image,(row,col))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_data.append(image)

            except:

                continue

        image_data = np.array(image_data)

        return image_data

    def save_images(self,filename, images):
        path = os.path.join('Datasets',filename)
        np.save(path,images)

    def load_images(self, file_name):

        name = os.path.basename(os.getcwd())
        while name!='InnoSpark-edir':
            if name == 'InnoSpark-edir':
                continue
            else:
                os.chdir('..')
                name = os.path.basename(os.getcwd())

        path = os.path.join(os.getcwd(),'Datasets',file_name)
        images = np.load(path)

        return images


    def load_DR(self, row, col):
        #Folder, Disease, CsvFileName, column in Csv, folder with images
        sources = np.array([['Kaggle','DR','full_df.csv', 'filename', 'preprocessed_images'],
                            ['8-Messidor','DR','messidor_data.csv','image_id','DR']])
        #images = np.array()
        images = []
        #labels = []

        for source in sources:

            folder = source[0]
            file_name = source[2]
            csv_path = self._get_csv_path(folder, file_name)#GET PATH OF csv
            df = pd.read_csv(csv_path)#READ CSV
            file_names = self._get_file_names(source[0], source[1], df, source[3])#GET FILE NAMES
            #file_names_total, labels = self._append_normal(file_names)
            path = os.path.join('Datasets',source[0],source[4])
            image_data = self._read_images(path, file_names, row, col)
            #print(image_data.shape)
            images.append(image_data)

        images_dr = np.concatenate((images[0],images[1]),axis=0)
        labels_dr = [1] * len(images_dr)
        images_balanced, labels_balanced = self._append_normal(images_dr, labels_dr)
        #return images_dr, labels_dr
        return images_balanced, labels_balanced

    def load_Cataracts(self, row, col):
        #Folder, Disease, CsvFileName, column in Csv, folder with images
        sources = np.array([['Kaggle','cataracts','full_df.csv', 'filename', 'preprocessed_images'],
                            ['Kaggle2','cataracts','kaggle2Cataract.csv','name','2_cataract']])
        #images = np.array()
        images = []

        for source in sources:

            folder = source[0]
            file_name = source[2]
            csv_path = self._get_csv_path(folder, file_name)#GET PATH OF csv
            df = pd.read_csv(csv_path)#READ CSV
            file_names = self._get_file_names(source[0], source[1], df, source[3])#GET FILE NAMES
            path = os.path.join('Datasets',source[0],source[4])
            image_data = self._read_images(path, file_names, row, col)
            #print(image_data.shape)
            images.append(image_data)

        images_cat = np.concatenate((images[0],images[1]),axis=0)
        labels_cat = [1] * len(images_cat)
        images_balanced, labels_balanced = self._append_normal(images_cat, labels_cat)

        return images_balanced, labels_balanced

    def load_Glaucoma(self, row, col):
        #Folder, Disease, CsvFileName, column in Csv, folder with images
        sources = np.array([['Kaggle','glaucoma','full_df.csv', 'filename', 'preprocessed_images'],
                            ['ORIGA','glaucoma','imgNamesORIGA.csv','New names','glaucoma']])
        #images = np.array()
        images = []

        for source in sources:

            folder = source[0]
            file_name = source[2]
            csv_path = self._get_csv_path(folder, file_name)#GET PATH OF csv
            df = pd.read_csv(csv_path)#READ CSV
            file_names = self._get_file_names(source[0], source[1], df, source[3])#GET FILE NAMES
            path = os.path.join('Datasets',source[0],source[4])
            image_data = self._read_images(path, file_names, row, col)
            #print(image_data.shape)
            images.append(image_data)

        images_glau = np.concatenate((images[0],images[1]),axis=0)
        labels_glau = [1] * len(images_glau)
        images_balanced, labels_balanced = self._append_normal(images_glau, labels_glau)

        return images_balanced, labels_balanced

    def load_catAnt(self):
        folder = 'Kaggle-Anterior'
        file_name = 'eye_dataset.csv'
        csv_path = self._get_csv_path(folder, file_name)#GET PATH OF csv
        df = pd.read_csv(csv_path)
        #df[df['Type']=='cat'].shape
        Y = df[df['Type']=='cat']["Type"]
        X = df[df['Type']=='cat'].drop(['Type'],axis=1)
        X = X.values.reshape(-1,151,332,1)
        X = X/255.0

        return X
