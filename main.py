import cv2
import numpy as np

cap = cv2.VideoCapture(0)
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

# classes = []
# with open("coco.names", "r") as f:
#   classes = [line.strip() for line in f.readlines()]
#  layer_names = net.getUnconnectedOutLayersNames()
classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "cell phone", "ship"]

gender_classes = ["Male", "Female"]

confidence_threshold = 0.2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    gender_blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                        swapRB=False)

    net.setInput(blob)
    detections = net.forward()

    gender_net.setInput(gender_blob)
    gender_preds = gender_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])

            if classes[class_id] == "person":
                gender_class_id = np.argmax(gender_preds)
                gender_label = f"{gender_classes[gender_class_id]}"
                print("Gendr:- " + gender_label)
                confidence = gender_preds[0, gender_class_id]

                if gender_label == "Male":
                    label = (f" ({classes[class_id]}) detected : {confidence:.2f} Hello sir im here for you, how can "
                             f"i help you")
                    print("Hello sir im here for you, how can i help you")
                else:
                    label = (f" ({classes[class_id]}) detected : {confidence:.2f} Hello madam im here for you, how can "
                             f"i help you")
                    print("Hello madam im here for you, how can i help you")

                obj = classes[class_id]
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                color = (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15

                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, f"Gender: {gender_label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2)
                print(f"Object is {classes[class_id]}")

            elif classes[class_id] == "bird":
                label = f"Class {class_id} ({classes[class_id]}): {confidence:.2f}"
                obj = classes[class_id]
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                color = (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(f"Object is {classes[class_id]}")

            elif classes[class_id] == "ship":
                label = f"Class {class_id} ({classes[class_id]}): {confidence:.2f}"
                obj = classes[class_id]
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                color = (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(f"Object is {classes[class_id]}")

            elif classes[class_id] == "car":
                label = f"Class {class_id} ({classes[class_id]}): {confidence:.2f}"
                obj = classes[class_id]
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                color = (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(f"Object is {classes[class_id]}")

            elif classes[class_id] == "aeroplane":
                label = f"Class {class_id} ({classes[class_id]}): {confidence:.2f}"
                obj = classes[class_id]
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                color = (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(f"Object is {classes[class_id]}")

            elif classes[class_id] == "bus":
                label = f"Class {class_id} ({classes[class_id]}): {confidence:.2f}"
                obj = classes[class_id]
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                color = (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(f"Object is {classes[class_id]}")

            elif classes[class_id] == "boat":
                label = f"Class {class_id} ({classes[class_id]}): {confidence:.2f}"
                obj = classes[class_id]
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                color = (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(f"Object is {classes[class_id]}")

            elif classes[class_id] == "dog":
                label = f"Class {class_id} ({classes[class_id]}): {confidence:.2f}"
                obj = classes[class_id]
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                color = (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(f"Object is {classes[class_id]}")

            elif classes[class_id] == "cat":
                label = f"Class {class_id} ({classes[class_id]}): {confidence:.2f}"
                obj = classes[class_id]
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                color = (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(f"Object is {classes[class_id]}")

            elif classes[class_id] == "horse":
                label = f"Class {class_id} ({classes[class_id]}): {confidence:.2f}"
                obj = classes[class_id]
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                color = (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(f"Object is {classes[class_id]}")

            elif classes[class_id] == "train":
                label = f"Class {class_id} ({classes[class_id]}): {confidence:.2f}"
                obj = classes[class_id]
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                color = (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(f"Object is {classes[class_id]}")

            elif classes[class_id] == "motorbike":
                label = f"Class {class_id} ({classes[class_id]}): {confidence:.2f}"
                obj = classes[class_id]
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                color = (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(f"Object is {classes[class_id]}")

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
