import numpy as np
import time
import cv2
from CNN_kaggle import CNN
from PIL import Image
import torch
import torchvision.transforms as transforms
import pandas as pd


def getClassifer():
    N_FEATURES = 27
    OUTPUT_DIM = 80
    model = CNN(N_FEATURES, OUTPUT_DIM)
    checkpoint = torch.load("Full_model")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def classification(image_array, m):
    # image = Image.open(image_array)
    mean = torch.tensor(0.3653)
    std = torch.tensor(0.2976)
    image = Image.fromarray(image_array)
    device = "cuda:0"
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean,std)])
    image = torch.unsqueeze(transform(image), 0)
    image = image.to(device)
    m.to(device)
    pred = m(image)

    return(pred.argmax(dim=1))


def predictVideo(INPUT_FILE, model):

	if isinstance(INPUT_FILE, str):
		cap = cv2.VideoCapture(INPUT_FILE)
	else:
		cap = INPUT_FILE
	Rclasses = []
	Rlabels = []
	# get the video frames' width and height for proper saving of videos
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	# create the `VideoWriter()` object
	out = cv2.VideoWriter('video_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
	# detect objects in each frame of the video
	m = getClassifer()
	while cap.isOpened():
		ret, frame = cap.read()
		if ret:
			image = frame
			
			c,l, im = predictImage(image, m, False, True)
			Rclasses.append(c)
			Rlabels.append(l)

			# cv2.imshow('image', im)
			out.write(im)
			if cv2.waitKey(10) & 0xFF == ord('q'):
				break
		else:
			break

	cap.release()
	cv2.destroyAllWindows()
	return Rclasses,Rlabels, 'video_result.mp4'

def predictImage(INPUT_FILE, model=None, show=False, returnImg=False):
    LABELS_FILE = 'kaggle_classes.csv'
    CONFIG_FILE = 'yolov4-obj.cfg'
    WEIGHTS_FILE = 'yolov4-4000.weights'
    CONFIDENCE_THRESHOLD = 0.3
    if model is None:
	    model= getClassifer()
    # Return values
    Rclasses = []
    Rlabels = []
    Rxmin = []
    Rymin = []
    Rhieght = []
    Rwiedht = []

    df = pd.read_csv(LABELS_FILE, sep=',')
    LABELS = df.to_numpy()

    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

    if isinstance(INPUT_FILE, str):
        image = cv2.imread(INPUT_FILE)
    else:
        image = cv2.cvtColor(np.array(INPUT_FILE), cv2.COLOR_RGB2BGR)

    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE_THRESHOLD:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(
        boxes, confidences, CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [0, 200, 20]

            crop_img = image[y:y+h, x:x+w]
            if crop_img.size:
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                labelid = classification(crop_img, model).item()

                Rclasses.append(LABELS[labelid][1])
                Rlabels.append(labelid)
                Rxmin.append(x)
                Rymin.append(y)
                Rhieght.append(h)
                Rwiedht.append(w)

                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}".format(LABELS[labelid][1])

                cv2.putText(image, text, (x-5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if show:
        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(im)
        image.show()
    if returnImg:
        return Rclasses, Rlabels, image
    return Rclasses, Rlabels


if __name__ == "__main__":
	predictVideo()
