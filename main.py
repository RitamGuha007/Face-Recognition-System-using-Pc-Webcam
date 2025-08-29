import os
import cv2
import numpy as np
import face_recognition

KNOWN_DIR="Known_Faces"
TOLERANCE=0.5
MODEL="hog"
DOWNSCALE=0.25
CAM_INDEX=0

known_encodings=[]
known_names=[]

for file in os.listdir(KNOWN_DIR):
    if file.lower().endswith((".jpeg",".jpg",".png")):
        path=os.path.join(KNOWN_DIR, file)
        img=face_recognition.load_image_file(path)
        encs=face_recognition.face_encodings(img)
        if encs:
            known_encodings.append(encs[0])
            name=os.path.splitext(file)[0]
            known_names.append(name)
        else:
            print(f"[WARN] No face found in {file} (skipped)")

print(f"[INFO] Loaded {known_encodings} recognized as {known_names}")

video=cv2.VideoCapture(CAM_INDEX)
if not video.isOpened():
    raise RuntimeError("Could not open WebCam, Try changing the Cam Index maybe it could help")

while True:
    ok,frame=video.read();
    if not ok:
        raise RuntimeError("Failed to Capture Frame from webcam!")
        break
    if DOWNSCALE!=1.0:
        small_frame=cv2.resize(frame,(0,0),fx=DOWNSCALE,fy=DOWNSCALE)
    else:
        small_frame=frame
    #converts the downscaled pic from BGR to RGB format as OpenCV accepts RGB pic
    rgb_pic=cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    #detects the face locations in the possible Downscaled frame
    face_locations=face_recognition.face_locations(rgb_pic,model=MODEL)
    
    #create a 128D Numpy Array for the located face in the Downscaled frame to uniquely identify it
    face_encodings=face_recognition.face_encodings(rgb_pic,face_locations)

    face_names=[]

    for face_2 in face_encodings:
        if not known_encodings:
            face_names.append("Unknown")
            continue
        
        # Compute distances between this face and all known encodings
        distances=face_recognition.face_distance(known_encodings,face_2)

         # Find the best (smallest) distance and its index | Lower distance = more accurate
        best_index=np.argmin(distances)
        best_distance=distances[best_index]

        if best_distance<=TOLERANCE:
            face_names.append(known_names[best_index])
        else:
            face_names.append("Unknown")
    
    #Drawing the results back on the original size frame
    for(top,right,bottom,left),name in zip(face_locations,face_names):
        if DOWNSCALE!=1.0:
            top=int(top/DOWNSCALE)
            right=int(right/DOWNSCALE)
            bottom=int(bottom/DOWNSCALE)
            left=int(left/DOWNSCALE)

            cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),3)

            label=name

            (txt_w,txt_h), baseline=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
            cv2.rectangle(frame,
                          (left,top-txt_h-baseline-6),
                          (left+txt_w+6,top),
                          (0,255,0),
                          cv2.FILLED)
            cv2.putText(frame,label,(left+3,top-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

    cv2.imshow("Ritam's Face Recognition System",frame)

    if cv2.waitKey(1) & 0xFF==ord('e'):
        break

video.release()
cv2.destroyAllWindows()


              






        

        
        


