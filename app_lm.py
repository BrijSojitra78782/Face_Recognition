import pyautogui
import os, sys
import cv2
import numpy as np
import math
import pandas as pd
import warnings
import face_recognition
from progressbar import ProgressBar
from pathlib import Path
import joblib
from sklearn.decomposition import PCA
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import VotingClassifier
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from datetime import datetime
from flask import Flask, flash, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler as sc

pbar = ProgressBar()

x_train = []
y_train = []
X = []
Y = []
name_grp = []
data_fr = []
G = []
pca_mean = 0
method = "Eigen"


class FaceRecognition:
    person_no = 1  
    def __init__(self):
        self.encode_faces()
        self.train()
        # self.test("Test_2.jpg")
    
    def encode_faces(self):
        global name_grp,X,Y,data_fr,pca_mean          
        faces = {}
        name_grp = set()
        for image in pbar(os.listdir('faces_l')):
            person_name = image.split("_")[0]            
            self.person_no+=1;
            # face_image = face_recognition.load_image_file(f"faces_l\{image}",mode="RGB")
            # face_image = face_image.flatten()
            # print(face_image.shape)
            # faces[image] = cv2.imdecode(face_image.flatten(), cv2.IMREAD_COLOR)
            faces[image] = cv2.imdecode(np.fromfile(f"faces_l\{image}", dtype=np.uint8), cv2.IMREAD_COLOR)
            X.append(faces[image].flatten())
            Y.append(person_name)
            name_grp.add(person_name)
        # print(len(X),len(X[0]))
        # X = np.array(X)
        pca = PCA(n_components=10).fit(X)
        X = pca.transform(X)
        joblib.dump(pca,'models_l\\PCA.pkl')
    #     n_components = 128
    #     eigenfaces = pca.components_[:n_components]
    #     X = eigenfaces@(X - pca.mean_).T
    #     X = X.T
        name_grp = list(name_grp) 
        joblib.dump(name_grp,'models_l\\names.pkl')
        print("Data fetched Successfully..")  
        data_fr = pd.DataFrame(X)
        temp = []
        for k in Y:
            temp.append(name_grp.index(k)+1)
        Y = temp
        data_fr['y'] = Y
        new_data = data_fr.copy()
        new_data = new_data.sample(frac = 1)
        filepath = Path('data_l1.csv')  
        # filepath.parent.mkdir(parents=True, exist_ok=True)  
        new_data.to_csv(filepath,index=False,header=False) 
        print("Data Saved..")
        # print(data_fr)
        # print(y_train)

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
    #         SVC_params = grid.best_params_()
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

    #         classifier.

        
        nn = MLPClassifier(hidden_layer_sizes=(50,40,20),activation="relu",solver="adam",learning_rate="constant",learning_rate_init=0.001,max_iter=1000)
        nn.fit(X,Y)
    #         MLP_params = nn.get_params()      
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
        # n_datapoints = result.shape[0]

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


    def test(self,test_img):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            global data_fr,X,Y,name_grp
            name_grp = joblib.load('models_l\\names.pkl')
            x_test = cv2.imdecode(np.fromfile(f"temp_img\{test_img}", dtype=np.uint8), cv2.IMREAD_COLOR)
            # print(len(x_test.flatten()),x_test.shape) 
            x_test = [x_test.flatten()]
            pca = joblib.load('models_l\\PCA.pkl')
            # print(pca)
                
            x_test = pca.transform(x_test)

            # n_components = 128
            # eigenfaces = pca.components_[:n_components]
            # x_test = eigenfaces@(x_test - pca_mean).T
            # x_test = x_test.T
            # print(x_test)
            ensbl = joblib.load('models_l\\ENSBL.pkl')

            y_pred = ensbl.predict(x_test)
            prob = ensbl.predict_proba(x_test)[0]
            # print(y_pred)            
            known_persons = 0
            print(prob)
            print(name_grp)
            for loc in y_pred:
                if(prob[int(loc)-1]>0.4):
                    name=name_grp[int(loc)-1]
                    known_persons+=1
                    print(name)
            if(known_persons==0):
                print("No Known person in the frame")
            
        
        


fr = FaceRecognition()



UPLOAD_FOLDER = r'E:\Sem_5\ML\lab\Innovative_\Face-Recognition-Attendance-System-main\IMAGE_FILES'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/success', methods=['GET', 'POST'])
def success():
    if 'file' not in request.files:
        # flash('No file part')
        return render_template('upload.html')
    file = request.files['file']
    if file.filename == '':
        # flash('No image selected for uploading')
        return render_template('upload.html')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # print('upload_image filename: ' + filename)
        # flash('Image successfully uploaded and displayed below')
        return render_template('upload.html')
    else:
        # flash('Allowed image types are -> png, jpg, jpeg, gif')
        return render_template('upload.html')


@app.route('/index')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    IMAGE_FILES = []
    filename = []
    dir_path = r'E:\Sem_5\ML\lab\Innovative_\Face-Recognition-Attendance-System-main\IMAGE_FILES'

    for imagess in os.listdir(dir_path):
        img_path = os.path.join(dir_path, imagess)
        img_path = face_recognition.load_image_file(img_path)  # reading image and append to list
        IMAGE_FILES.append(img_path)
        filename.append(imagess.split(".", 1)[0])

    def encoding_img(IMAGE_FILES):
        encodeList = []
        for img in IMAGE_FILES:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    def takeAttendence(name):
        with open('attendence.csv', 'r+') as f:
            mypeople_list = f.readlines()
            nameList = []
            for line in mypeople_list:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                datestring = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{datestring}')

    encodeListknown = encoding_img(IMAGE_FILES)
    # print(len('sucesses'))


    cap = cv2.VideoCapture(0)
    


    while True:
        success, img = cap.read()
        imgc = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        # converting image to RGB from BGR
        imgc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fasescurrent = face_recognition.face_locations(imgc)
        encode_fasescurrent = face_recognition.face_encodings(imgc, fasescurrent)
        x_test = []
        for enc in encode_fasescurrent:         
            x_test.append(enc)
        # person_count = 0
        # if(len(x_test)):
        #     y_pred = ensbl.predict(x_test)
        #     prob = ensbl.predict_proba(x_test)
        #     print(prob)
        #     print(y_pred)
        #     if(len(y_pred)>0):
        #         ite=0  
        #         for loc in y_pred:
        #             name=name_grp[loc-1],prob[ite][loc-1]
        #             person_count+=1
        #             ite+=1
        # if(person_count==0):
        #     name="No known Person in the frame.."

        # faceloc- one by one it grab one face location from fasescurrent
        # than encodeFace grab encoding from encode_fasescurrent
        # we want them all in same loop so we are using zip
        for encodeFace, faceloc in zip(encode_fasescurrent, fasescurrent):
            # matches_face = face_recognition.compare_faces(encodeListknown, encodeFace)
            # face_distence = face_recognition.face_distance(encodeListknown, encodeFace)
            # # print(face_distence)
            # finding minimum distence index that will return best match
            # matchindex = np.argmin(face_distence)
            person_count = 0
            if(len(x_test)):
                y_pred = ensbl.predict(x_test)
                prob = ensbl.predict_proba(x_test)
                print(prob)
                print(y_pred)
                if(len(y_pred)>0):
                    ite=0  
                    for loc in y_pred:
                        name=name_grp[loc-1] + str(prob[ite][loc-1])
                        if(prob[ite][loc-1]<0.4):
                            name="No known Person in the frame.."
                        
                        person_count+=1
                        ite+=1
            if(person_count==0):
                name="No known Person in the frame.."

            # if matches_face[matchindex]:
                # name = filename[matchindex].upper()
            # print(name)
            y1, x2, y2, x1 = faceloc
            # multiply locations by 4 because we above we reduced our webcam input image by 0.25
            # y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), 2, cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # takeAttendence(name)  # taking name for attendence function above

        # cv2.imshow("campare", img)
        # cv2.waitKey(0)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    pass
    # app.run(debug=True)
