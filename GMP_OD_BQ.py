# Import Libraries
import cv2
import re
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import datetime
import pandas as pd
import json
import requests
import firebase_admin
from firebase_admin import credentials, firestore
import time
import glob
import os
from google.cloud import bigquery

# Function for load labels
def load_labels(path='labels.txt'):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels

# Function for input tensor before detection
def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)

# Funtion for get result from detection
def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

# Function for run detection
def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  scores = get_output_tensor(interpreter, 0)
  boxes = get_output_tensor(interpreter, 1)
  count = int(get_output_tensor(interpreter, 2))
  classes = get_output_tensor(interpreter, 3)
  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {'bounding_box': boxes[i], 'class_id': classes[i], 'score': scores[i]}
      results.append(result)
  return results

if __name__ == "__main__":
    # Auth Google Drive
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")
    drive = GoogleDrive(gauth)

    # Process auth to BigQuery
    PROJECT_ID = 'cctv-object-detection'
    client = bigquery.Client(project = PROJECT_ID)
    dataset_id = 'iot_cctv_detection'
    table_id = 'detection'
    # Set parameter for import csv to BigQuery
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.autodetect = True
    
    # Load label and model
    labels = load_labels()
    interpreter = Interpreter('tflite/detect-md-V15-frozen.tflite')
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    
    # Create outlet variable
    outlet = "Foresta"
    
    # Create variable for skip frame
    count = 0
    
    # Start stream
    cap = cv2.VideoCapture("rtsp://username:pass@192.1.1.254:554/Streaming/channels/502")
    
    # Get size frame
    CAMERA_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    CAMERA_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Display log time start_capture
    print("Start capture at: "+str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
    
    # Loop detection
    while(cap.isOpened()):
        # Read frame
        ret, frame = cap.read()
        if ret == True:
            # Resize frame into 320x320
            img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320,320))
            #img = cv2.resize(frame, (320,320))
            # Run detection
            res = detect_objects(interpreter, img, 0.20) # Threshold 20%
            # Variable ts_now for BigQuery
            ts_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Variable time now for lark
            time_now = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            # Variable for name image
            name_img = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # Loop for get result detection
            for result in res:
                # Display log result detection
                l = labels[int(result['class_id'])]
                s = int(result['score']*100)
                print('label: {}  score: {}  time: {}'.format(l, s, time_now))

                # Check condition for input into database
                if (l == "without_mask" and s > 45) or (l == "without_hat" and s > 35):
                    # Save capture into local computer
                    cv2.imwrite('bukti/'+outlet+'/'+str(name_img)+'.jpg', frame)
                    # Read capture from local computer
                    upload_file_list = ['bukti/'+outlet+'/'+str(name_img)+'.jpg']
                    # Process upload capture into GoogleDrive
                    for upload_file in upload_file_list:
                        # Set upload image into folder "pelanggaran" in Google Drive
                        gfile = drive.CreateFile({'parents': [{'id': '1B_lnHS0gR0h4aiIDyHsIhXJsqzGS4jPX'}]})
                        # Read file and set it as the content of this instance.
                        gfile.SetContentFile(upload_file)
                        # Upload file into Google Drive
                        gfile.Upload()
                    # Process get image link in folder "pelanggaran"
                    files = drive.ListFile({'q': "'1B_lnHS0gR0h4aiIDyHsIhXJsqzGS4jPX' in parents and trashed=false"}).GetList()
                    for file in files:
                        keys = file.keys()
                        if file['shared']:
                            if file['originalFilename'] == upload_file:
                                link = 'https://drive.google.com/file/d/' + file['id'] + '/view?usp=sharing'
                    # Delete all images on local computer
                    dir = glob.glob('bukti/'+outlet+'/*')
                    for d in dir:
                        os.remove(d)
                    # Create dataframe and input data to dataframe
                    df = pd.DataFrame({'CCTV': outlet, 'date': ts_now, 'url': link, 'label': l, 'score': s}, index=[0])
                    # Export the last detection (dataframe) to csv
                    df.to_csv('bukti/bukti-deteksi-terakhir.csv', index=False)
                    # Import csv to BigQuery
                    with open('bukti/bukti-deteksi-terakhir.csv', "rb") as source_file:
                        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)
                    job.result()
                    # Create variable data for json
                    data = {'CCTV': outlet, 'date': time_now, 'url': link, 'label': labels[int(result['class_id'])], 'score': int(result['score']*100)}                    
                    # Send Data to Lark Webhook
                    requests.post('https://www.larksuite.com/flow/api/trigger-webhook/190e1e3474148679060a2a45315c2db0', data=json.dumps(data), headers={'Content-Type': 'application/json'})
                    #time.sleep(5)
            # Set duration skip frame
            #count += 10 # i.e. at 1 fps, 1*5=5
            # Set frame
            #cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            # Display log start_capture
            print("Recapture after crash at: "+str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
            # Recapture stream
            cap = cv2.VideoCapture("rtsp://admin:Yummyfood21@172.17.1.80:554/Streaming/channels/502")
    
    # Stop stream
    cap.release()
