from flask import Flask, request, Response, render_template,flash
from werkzeug.utils import secure_filename
from PIL import Image
import pyautogui
import os, sys
import cv2
import shutil
import numpy as np
import math
import pandas as pd
import warnings
import time
import face_recognition
from progressbar import ProgressBar
from pathlib import Path
import joblib
from sklearn.decomposition import PCA
import face_recognition
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from datetime import datetime
from flask import Flask, flash, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler as sc
pbar = ProgressBar()
app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

x_train = []
y_train = []
X = []
Y = []
name_grp = []
data_fr = []
G = []
pca_mean = 0
msg = ""
data_changed = False
reloaded = False
pred_name = ""
pred_prob = ""
method = "Eigen"


def encode_faces():
    global name_grp,X,Y,data_fr,pca_mean          
    faces = {}
    X=[]
    Y=[]
    name_grp = set()
    for image in pbar(os.listdir('faces_l')):
        person_name = image.split("_")[0]            
        if(method=="Eigen"):
            face_enc = np.array(cv2.imdecode(np.fromfile(f"faces_l\{image}", dtype=np.uint8), cv2.IMREAD_COLOR))            
        else:
            face_image = face_recognition.load_image_file(f"faces_l\{image}")
            face_enc = np.array(face_recognition.face_encodings(face_image)[0])
        face_enc = face_enc.flatten()
        # print(face_enc.shape,image)
        X.append(face_enc)
        Y.append(person_name)
        name_grp.add(person_name)
    X = np.array(X)
    # print(X)
    # print(X.shape)
    if(method=="Eigen"):
        pca = PCA(n_components=10).fit(X)
        X = pca.transform(X)
        joblib.dump(pca,'models_l\\PCA.pkl')
    print(X.shape)
    name_grp = list(name_grp) 
    if(method=="Eigen"):
        joblib.dump(name_grp,'models_l\\names.pkl')
    else:
        joblib.dump(name_grp,'models_l\\names_r.pkl')
    print("Data fetched Successfully..")  
    data_fr = pd.DataFrame(X)
    temp = []
    for k in Y:
        temp.append(name_grp.index(k)+1)
    Y = temp
    data_fr['y'] = Y
    new_data = data_fr.copy()
    new_data = new_data.sample(frac = 1)
    if(method=="Eigen"):
        filepath = Path('data_l.csv')
    else:
        filepath = Path('data_rec.csv')  
    new_data.to_csv(filepath,index=False,header=False) 
    print("Data Saved..")

def train():
    global data_fr,X,Y
    if(method=="Eigen"):
        data_fr = pd.read_csv("data_l.csv")
    else:
        data_fr = pd.read_csv("data_rec.csv")
    Y = data_fr.iloc[:,-1]
    X = data_fr.iloc[:,:-1]
    print(X.shape)
    print(Y.shape)
    knn = KNeighborsClassifier()
    knn_params = dict(n_neighbors=list(range(1, 5)))
    grid_knn = GridSearchCV(knn, knn_params, cv=3, scoring='accuracy', return_train_score=False)
    grid_knn.fit(X, Y)
    joblib.dump(grid_knn,'models_l\\KNN.pkl')
    print("KNN trained")        

    clf = svm.SVC(probability=True)

    clf_params={
        "C":[0.001,0.01],
        "gamma":[0.001,0.01],
        "kernel":["rbf"]
    }
    grid_svc = GridSearchCV(clf,clf_params,refit=True,verbose=3)
    grid_svc.fit(X,Y)
    joblib.dump(grid_svc,'models_l\\SVC.pkl')
    print("Support Vector Classifier Trained")


    
    RF_classifier= RandomForestClassifier()  
    RF_classifier.fit(X,Y)  
    RF_params = { 
        'n_estimators': [5, 20],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8],
        'criterion' :['gini', 'entropy']
    }
    grid_RF = GridSearchCV(RF_classifier, RF_params, cv= 3)
    joblib.dump(grid_RF,'models_l\\RF.pkl')
    print("Random Forest Classifier Trained")


    
    nn = MLPClassifier(hidden_layer_sizes=(50,40,20),activation="relu",solver="adam",learning_rate="constant",learning_rate_init=0.001,max_iter=1000)
    nn.fit(X,Y)
    joblib.dump(nn,'models_l\\MPL.pkl')
    print("MultiLayer Protocol Trained")

    

    ensbl = VotingClassifier(estimators = [('knn', knn), ('svc', grid_svc),('rf',RF_classifier),('ANN',nn)],voting='soft')
    ensbl.fit(X,Y)      
    if(method=="Eigen"):  
        joblib.dump(ensbl,'models_l\\ENSBL.pkl')
    else:
        joblib.dump(ensbl,'models_l\\ENSBL_r.pkl')
    print("Ensemble Trained")
    
    epoch = 5
    n_dimensions = data_fr.shape[1]-1
    Acc = []

    for e in range(epoch):

        train_data = data_fr.sample(frac=0.80)

        train_data = train_data.values
        train_x = train_data[:,:-1]
        train_y = train_data[:,-1].ravel()

        test_data = data_fr.sample(frac=0.20)
        test_data = test_data.values
        test_x = test_data[:,:-1]
        test_y = test_data[:,-1].ravel()

        ensbl.fit(train_x,train_y)
        y_pred = ensbl.predict(test_x)
        Acc.append(accuracy_score(test_y,y_pred))
        print(Acc[-1])


    print("accuracy_score : {:.5f}".format(np.sum(Acc)/epoch))

@app.route('/')
def hello_world():
    return render_template('index.html')
@app.route('/<string:other>')
def other(other):
    return redirect(url_for('hello_world'))
def gen():
    IMAGE_FILES = []
    filename = []
    dir_path = r'faces_l'
    name_grp =  joblib.load('models_l\\names_r.pkl')
    ensbl = joblib.load('models_l\\ENSBL_r.pkl')

    # for images in os.listdir(dir_path):
    #     img_path = os.path.join(dir_path, images)
    #     img_path = face_recognition.load_image_file(img_path)  # reading image and append to list
    #     IMAGE_FILES.append(img_path)
    #     filename.append(images.split("_", 1)[0])

    # def encoding_img(IMAGE_FILES):
    #     encodeList = []
    #     for img in pbar(IMAGE_FILES):
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         encode = face_recognition.face_encodings(img)[0]
    #         encodeList.append(encode)
    #     return encodeList

    # encodeListknown = encoding_img(IMAGE_FILES)
    # print(len('sucesses'))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        cap = cv2.VideoCapture(0)

        while True:
            success, img = cap.read()
            imgc = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            fasescurrent = face_recognition.face_locations(imgc)
            encode_fasescurrent = face_recognition.face_encodings(imgc, fasescurrent)
            x_test = []
            for enc in encode_fasescurrent:         
                x_test.append(enc)
            
            for encodeFace, faceloc in zip(encode_fasescurrent, fasescurrent):
                person_count = 0
                if(len(x_test)):
                    y_pred = ensbl.predict(x_test)
                    prob = ensbl.predict_proba(x_test)
                    # print(prob)
                    # print(y_pred)
                    if(len(y_pred)>0):
                        ite=0  
                        # print(name_grp)
                        for loc in y_pred:
                            name=name_grp[loc-1] + str(round(prob[ite][loc-1],4)*100)+" %"
                            if(prob[ite][loc-1]<0.4):
                                name="No known Person in the frame.."
                            
                            person_count+=1
                            ite+=1
                if(person_count==0):
                    name="No known Person in the frame.."

                y1, x2, y2, x1 = faceloc
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), 2, cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            key = cv2.waitKey(20)
            if key == 27:
                break
    


@app.route('/videofeed')
def videofeed():
    print("hello")
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/live')
def live():
    return render_template('olduserlive.html')

@app.route('/live/train')
def live_train():
    global method
    method = "face_rec"
    encode_faces()
    train()
    return redirect(url_for('live'))

@app.route('/olduser')
def olduser():
    return render_template('olduser.html',msg=msg)

@app.route('/olduser/train')
def Eigen_train():
    global method
    method = "Eigen"
    encode_faces()
    train()
    return redirect(url_for('olduser'))

@app.route('/olduser/test')
def olduser_test():
    global msg,page_no,reloaded,pred_name,pred_prob
    if(not reloaded):
        time.sleep(2)
        reloaded = True
        return redirect(url_for('olduser_test'))
    else:
        reloaded = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        name_grp = joblib.load('models_l\\names.pkl')
        test_faces = []
        img_files = os.listdir('temp_img')
        for img in img_files:
            if(img.startswith("Test")):
                test_faces.append(img)
        print(test_faces)
        if(len(test_faces)>0):
            if(len(test_faces)==1):
                test_img = test_faces[-1]
            else:
                test_img = test_faces[-2]
            print(test_img)
            x_test = cv2.imdecode(np.fromfile(f"temp_img/{test_img}", dtype=np.uint8), cv2.IMREAD_COLOR)
            if(len(test_faces)>=4):
                for img in test_faces:
                    os.remove(f"temp_img/{img}")              
            print(x_test.shape,x_test.flatten().shape)
            x_test = [x_test.flatten()]
            pca = joblib.load('models_l\\PCA.pkl')            
            x_test = pca.transform(x_test)

            ensbl = joblib.load('models_l\\ENSBL.pkl')

            y_pred = ensbl.predict(x_test)
            prob = ensbl.predict_proba(x_test)[0]
            known_persons = 0
            print(prob)
            print(name_grp)
            for loc in y_pred:
                if((prob[int(loc)-1])>0.30):
                    name=name_grp[int(loc)-1]
                    known_persons+=1
                    msg = name +" : " + str(prob[int(loc)-1])
                    pred_name = name
                    pred_prob= str(prob[int(loc)-1])
                    # print(name)
            if(known_persons==0):
                msg = "No Known person in the frame"
                                
            print(msg)
            return redirect(url_for('olduser'))
        else:
            print("No image for testing")
        
            
        
        

@app.route('/newuser/<string:person_name>')
def newuser_register(person_name):
    global data_changed,method
    print(person_name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        files = os.listdir('temp_img')
        img_count=1
        for img in files:
            if(img.startswith('Sample')):
                shutil.move(f'temp_img\{img}',f'faces_l\{person_name}_{img_count}.jpg')
                data_changed = True
                img_count+=1
        print("Done..")     
        method = "Face_rec"  

        return redirect(url_for('hello_world'))


@app.route('/Redirecting', methods=['POST','GET'])
def Redirecting():
    return render_template('Redirect.html')

@app.route('/newuser', methods=['POST','GET'])
def newuser():
    return render_template('newuser.html')


if __name__ == "__main__":
    # encode_faces()
    # train()
    app.run(debug=True)
