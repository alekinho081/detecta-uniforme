# server_detect_uniform_optimized.py
import cv2
import numpy as np
import time
import requests
import paho.mqtt.client as mqtt
import threading
import traceback

# -------- CONFIGURAÇÕES --------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_ALERT = "projeto/uniforme/alert"

CALLMEBOT_PHONE = "5548996476451"   # número com DDD (ex: 5511999999999)
CALLMEBOT_APIKEY = "7067430"        # sua chave do CallMeBot (se aplicável)

UNIFORM_THRESHOLD_PERCENT = 20.0    # % do torso que deve ser da cor do uniforme
LOWER_HSV = np.array([35, 20, 120])
UPPER_HSV = np.array([85, 255, 255])

# Performance / comportamento
FRAME_SKIP = 5           # roda detecção a cada N frames
CONFIDENCE_MIN = 0.6     # limiar de confiança do detector
ALERT_INTERVAL = 10      # segundos entre alertas (rate-limit)
CAM_WIDTH = 640
CAM_HEIGHT = 480

# Paths do modelo (coloque esses arquivos na pasta do script)
PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"

# classes do MobileNet-SSD (person = 15)
PERSON_CLASS_ID = 15
# --------------------------------

# Função de envio de WhatsApp em thread (não bloqueia o loop)
def send_whatsapp_alert_async(msg):
    def worker(m):
        try:
            url = f"https://api.callmebot.com/whatsapp.php?phone={CALLMEBOT_PHONE}&text={requests.utils.quote(m)}&apikey={CALLMEBOT_APIKEY}"
            r = requests.get(url, timeout=8)
            print("CallMeBot status:", r.status_code)
        except Exception as e:
            print("Erro ao enviar CallMeBot:", e)
            traceback.print_exc()
    t = threading.Thread(target=worker, args=(msg,), daemon=True)
    t.start()

# Cria cliente MQTT (API de callback VERSION2 para evitar DeprecationWarning)
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
except Exception as e:
    print("Erro ao conectar no broker MQTT:", e)

# Carrega modelo (OpenCV DNN)
print("Carregando modelo...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
print("Modelo carregado.")

# Abre câmera e diminui resolução
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
if not cap.isOpened():
    raise RuntimeError("Erro ao acessar a webcam")

frame_count = 0
detections_cache = None
last_alert_time = 0
prev_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame não lido da câmera, encerrando.")
            break

        frame_count += 1
        (h, w) = frame.shape[:2]

        # roda detecção só a cada FRAME_SKIP frames
        if detections_cache is None or (frame_count % FRAME_SKIP == 0):
            # redimensiona antes de criar blob (300x300 para MobileNet-SSD)
            small = cv2.resize(frame, (300, 300))
            blob = cv2.dnn.blobFromImage(small, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections_cache = net.forward()

        detections = detections_cache
        count_with_uniform = 0
        count_without = 0

        # percorre detecções
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < CONFIDENCE_MIN:
                continue
            class_id = int(detections[0, 0, i, 1])
            if class_id != PERSON_CLASS_ID:
                continue

            # bounding box (normaliza para imagem original)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            # extrai pessoa e define ROI do torso
            person = frame[startY:endY, startX:endX]
            if person.size == 0:
                continue
            ph, pw = person.shape[:2]
            torso_y1 = int(ph * 0.15)
            torso_y2 = int(ph * 0.55)
            torso_x1 = int(pw * 0.2)
            torso_x2 = int(pw * 0.8)
            if torso_y2 <= torso_y1 or torso_x2 <= torso_x1:
                continue
            torso = person[torso_y1:torso_y2, torso_x1:torso_x2]
            if torso.size == 0:
                continue

            # converte e aplica máscara HSV
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.equalizeHist(hsv[:, :, 1])
            mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            uniform_pixels = cv2.countNonZero(mask)
            total_pixels = mask.size
            percent = 100.0 * (uniform_pixels / (total_pixels + 1e-9))
            label = f"{percent:.1f}%"

            # decide status e desenha
            if percent >= UNIFORM_THRESHOLD_PERCENT:
                count_with_uniform += 1
                box_color = (0, 255, 0)
                status = "uniform"
            else:
                count_without += 1
                box_color = (0, 0, 255)
                status = "no_uniform"

            cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)
            cv2.rectangle(frame, (startX + torso_x1, startY + torso_y1), (startX + torso_x2, startY + torso_y2), box_color, 1)
            cv2.putText(frame, label, (startX, startY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Se houver pessoas sem uniforme, publica 1 alerta resumido (rate-limited)
        now = time.time()
        if count_without > 0 and (now - last_alert_time) > ALERT_INTERVAL:
            payload = {"time": now, "sem_uniforme": int(count_without), "com_uniforme": int(count_with_uniform)}
            try:
                client.publish(MQTT_TOPIC_ALERT, str(payload))
            except Exception as e:
                print("Erro publish MQTT:", e)
            # envia WhatsApp em thread (não bloqueante)
            send_whatsapp_alert_async(f"Alerta: {int(count_without)} pessoa(s) sem uniforme detectada(s)!")
            last_alert_time = now

        # overlay de status e FPS
        cv2.putText(frame, f"Com uniforme: {count_with_uniform}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Sem uniforme: {count_without}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # calcula FPS aproximado
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0.0
        prev_time = cur_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Deteccao", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    try:
        cap.release()
        cv2.destroyAllWindows()
    except:
        pass
    try:
        client.loop_stop()
        client.disconnect()
    except:
        pass
    print("Encerrado.")
