import tkinter as tk
import cv2, os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import csv

window = tk.Tk()
window.title("Attendence System")
window.geometry('350x180')

lbl1 = tk.Label(window, text="Enter ID", width=10)
lbl1.place(x=80, y=5)

lbl2 = tk.Label(window, text="Enter Name", width=10)
lbl2.place(x=80, y=55)

txt1 = tk.Entry(window, width=10)
txt1.place(x=160, y=5)

txt2 = tk.Entry(window, width=10)
txt2.place(x=160, y=55)

message = tk.Label(window, text="")
message.place(x=160, y=79)


def clear():
    txt1.delete(0, 'end')
    txt2.delete(0, 'end')
    res = ""
    message.configure(text=res)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def TakeImages():
    Id = (txt1.get())
    if (is_number(Id)):
        face_classifier = cv2.CascadeClassifier('E:/python/fd/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        def face_extractor(img):

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            if faces is ():
                return None

            for (x, y, w, h) in faces:
                cropped_face = img[y:y + h, x:x + w]

            return cropped_face

        cap = cv2.VideoCapture(0)
        count = 0

        while True:
            ret, frame = cap.read()
            if face_extractor(frame) is not None:
                count += 1
                face = cv2.resize(face_extractor(frame), (500, 500))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                file_name_path = "E:/python/fd/TrainingImage/Train.User."+Id+'.'+ str(count) + ".jpg"
                cv2.imwrite(file_name_path, face)

                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Face cropper', face)
                print("FACE FOUND")

            else:
                print("Face not Found")
                pass

            if cv2.waitKey(1) == 13 or count == 100:
                break

        cap.release()
        cv2.destroyAllWindows()
        print('Colleting Samples Complete!!!')

        res = "Images Saved for " + Id
        message.configure(text=res)
    else:
        res = "Enter Numeric Id"
        message.configure(text=res)


def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "E:/python/fd/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    TrainingImagePath = 'E:/python/fd/TrainingImage/'
    faces, Ids = getImagesAndLabels(TrainingImagePath)
    recognizer.train(faces, np.array(Ids))
    recognizer.write("E:/python/fd/TrainingImageLabel/Trainner.yml")
    res = "Image Trained"
    message.configure(text=res)


def getImagesAndLabels(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    print(imagePaths)


    faces = []

    Ids = []

    for imagePath in imagePaths:

        pilImage = Image.open(imagePath).convert('L')

        imageNp = np.array(pilImage, 'uint8')

        Id = int(os.path.split(imagePath)[-1].split(".")[2])

        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("E:/python/fd/TrainingImageLabel/Trainner.yml")
    harcascadePath = "E:/python/fd/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    df = pd.read_csv("E:/python/fd/StudentDetails/StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['ID', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                #attendance.loc[len(attendance)] = [Id, date, timeStamp]
                aa = df.loc[df['ID'] == Id]['Name'].values
                tt = str(Id) + "-" + aa[0]
                print(str(Id)+" "+aa[0])


                #insert into attendence values(%s,%s,%s)
                val=(id,aa[0],date)
            else:
                Id = 'Unknown'
                tt = str(Id)
            if (conf > 75):
                noOfFile = len(os.listdir("E:/python/fd/ImagesUnknown/")) + 1
                cv2.imwrite("E:/python/fd/ImagesUnknown/Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(keep='first', subset=['ID'])
        cv2.imshow('im', im)
        if (cv2.waitKey(1) == ord('q')):
            break

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "E:/python/fd/Attendance/Attendance_" + date + ".csv"

    attendance.to_csv(fileName, index=False)
    cam.release()
    cv2.destroyAllWindows()
    print(attendance)


clearButton1 = tk.Button(window, text="Clear", command=clear)
clearButton1.place(x=230, y=5)

clearButton2 = tk.Button(window, text="Clear", command=clear)
clearButton2.place(x=230, y=55)

takeImg = tk.Button(window, text="Take Images", command=TakeImages)
takeImg.place(x=20, y=105)
trainImg = tk.Button(window, text="Train Images", command=TrainImages)
trainImg.place(x=110, y=105)
trackImg = tk.Button(window, text="Track Images", command=TrackImages)
trackImg.place(x=206, y=105)
quitWindow = tk.Button(window, text="Quit", command=window.destroy)
quitWindow.place(x=300, y=105)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0, )
copyWrite.tag_configure("superscript", offset=4)
copyWrite.insert("insert", "Developed by KAISH", "", "TM", "superscript")
copyWrite.configure(state="disabled")
copyWrite.pack(side="top")
copyWrite.place(x=95, y=140)

window.mainloop()