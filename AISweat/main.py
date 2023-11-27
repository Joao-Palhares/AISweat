import sys
import cv2
import math
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
counter = 0
calculate_distance = False
calculate_distance_inicial = True
mudar_distancia = False
esta_lado = False
esta_lado_mid = False
half_second_timer = 0
distancia_ombros_in = 0
distancia_ombro_quadr_l_in = 0
distancia_ombro_quadr_r_in = 0

model = YOLO("best.pt")


def change_color_one_bad(image, membro1, membro2):
    cv2.line(
        image,
        tuple(np.multiply(membro1, [1280, 720]).astype(int)),
        tuple(np.multiply(membro2, [1280, 720]).astype(int)),
        (247, 43, 7),
        3,
    )


def change_color_one_good(image, membro1, membro2):
    cv2.line(
        image,
        tuple(np.multiply(membro1, [1280, 720]).astype(int)),
        tuple(np.multiply(membro2, [1280, 720]).astype(int)),
        (6, 97, 50),
        3,
    )


def check(image, ombro_l, cotovelo_l, angulo, valor1, valor2):
    if valor1 <= angulo <= valor2:
        change_color_one_good(image, ombro_l, cotovelo_l)
    else:
        change_color_one_bad(image, ombro_l, cotovelo_l)


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def draw_keypoints(image, landmarks, results):
    for landmark in landmarks:
        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(101, 242, 13), thickness=3, circle_radius=2),
            mp_drawing.DrawingSpec(color=(7, 98, 247), thickness=3, circle_radius=2),
        )


def printAng(image, angle, part):
    cv2.putText(
        image,
        str(angle),
        tuple(np.multiply(part, [1280, 720]).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


class Menu(QMainWindow):
    def correction(self, exercicio):
        global counter, calculate_distance, half_second_timer, cap, distancia_ombros_in, distancia_ombro_quadr_l_in, distancia_ombro_quadr_r_in, calculate_distance_inicial, esta_lado, esta_lado_mid
        cap = cv2.VideoCapture(0)
        try:

            if not cap.isOpened():
                self.mostrar_img_nao_disponivel()
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            with mp_pose.Pose(
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            ) as pose:
                while cap.isOpened():

                    if self.stop_correction:
                        self.camera_label.clear()
                        self.camera_label.hide()
                        self.mostrar_img_nao_disponivel()
                        break

                    ret, frame = cap.read()

                    if not ret:
                        break

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    image = cv2.flip(image, 1)

                    results = pose.process(image)

                    if results.pose_landmarks and results:
                        try:
                            landmarks = results.pose_landmarks.landmark

                            if landmarks:
                                # Right side

                                dedom_r = [
                                    landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y,
                                ]
                                ombro_r = [
                                    landmarks[
                                        mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                                    ].x,
                                    landmarks[
                                        mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                                    ].y,
                                ]
                                cotovelo_r = [
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                                ]
                                pulso_r = [
                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                                ]
                                quadril_r = [
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                                ]
                                joelho_r = [
                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                                ]
                                cintura_r = [
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                                ]
                                tornozelo_r = [
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                                ]

                                angle_cotov_r = calculate_angle(
                                    ombro_r, cotovelo_r, pulso_r
                                )
                                angle_omb_r = calculate_angle(
                                    quadril_r, ombro_r, cotovelo_r
                                )
                                angle_pulso_r = calculate_angle(
                                    cotovelo_r, pulso_r, dedom_r
                                )
                                angle_quadril_r = calculate_angle(
                                    joelho_r, quadril_r, ombro_r
                                )
                                angle_joelho_r = calculate_angle(
                                    tornozelo_r, joelho_r, quadril_r
                                )

                                printAng(image, angle_pulso_r, pulso_r)
                                printAng(image, angle_cotov_r, cotovelo_r)
                                printAng(image, angle_omb_r, ombro_r)
                                printAng(image, angle_quadril_r, quadril_r)
                                printAng(image, angle_joelho_r, joelho_r)

                                # Left side

                                dedom_l = [
                                    landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y,
                                ]
                                ombro_l = [
                                    landmarks[
                                        mp_pose.PoseLandmark.LEFT_SHOULDER.value
                                    ].x,
                                    landmarks[
                                        mp_pose.PoseLandmark.LEFT_SHOULDER.value
                                    ].y,
                                ]
                                cotovelo_l = [
                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                                ]
                                pulso_l = [
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                                ]
                                quadril_l = [
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                                ]
                                joelho_l = [
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                                ]
                                tornozelo_l = [
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                                ]

                                angle_cotov_l = calculate_angle(
                                    ombro_l, cotovelo_l, pulso_l
                                )
                                angle_omb_l = calculate_angle(
                                    quadril_l, ombro_l, cotovelo_l
                                )
                                angle_pulso_l = calculate_angle(
                                    cotovelo_l, pulso_l, dedom_l
                                )
                                angle_quadril_l = calculate_angle(
                                    joelho_l, quadril_l, ombro_l
                                )
                                angle_joelho_l = calculate_angle(
                                    tornozelo_l, joelho_l, quadril_l
                                )

                                printAng(image, angle_pulso_l, pulso_l)
                                printAng(image, angle_cotov_l, cotovelo_l)
                                printAng(image, angle_omb_l, ombro_l)
                                printAng(image, angle_quadril_l, quadril_l)
                                printAng(image, angle_joelho_l, joelho_l)

                                draw_keypoints(image, landmarks, results)

                                if esta_lado:
                                    if exercicio == 'Squat':
                                        check(image, ombro_l, cotovelo_l, angle_omb_l, 60, 140)
                                        check(image, ombro_r, cotovelo_r, angle_omb_r, 60, 140)
                                        check(image, joelho_r, tornozelo_r, angle_joelho_r, 40, 180)
                                        check(image, joelho_l, tornozelo_l, angle_joelho_l, 40, 180)
                                        check(image, quadril_r, joelho_r, angle_quadril_r, 50, 170)
                                        check(image, quadril_l, joelho_l, angle_quadril_l, 50, 170)
                                    elif exercicio == 'Push-Up':
                                        check(image, ombro_l, cotovelo_l, angle_omb_l, 45, 160)
                                        check(image, ombro_r, cotovelo_r, angle_omb_r, 45, 160)
                                        check(image, cotovelo_l, pulso_l, angle_cotov_l, 70, 180)
                                        check(image, cotovelo_r, pulso_r, angle_cotov_r, 70, 180)
                                    elif exercicio == 'Sit-Up':
                                        check(image, ombro_l, cotovelo_l, angle_omb_l, 30, 180)
                                        check(image, ombro_r, cotovelo_r, angle_omb_r, 30, 180)
                                        check(image, cotovelo_l, pulso_l, angle_cotov_l, 40, 170)
                                        check(image, cotovelo_r, pulso_r, angle_cotov_r, 40, 170)
                                else:
                                    if exercicio == 'Squat':
                                        check(image, quadril_r, joelho_r, angle_quadril_r, 130, 180)
                                        check(image, quadril_l, joelho_l, angle_quadril_l, 130, 180)
                                        check(image, joelho_l, tornozelo_l, angle_joelho_l, 40, 180)
                                        check(image, joelho_r, tornozelo_r, angle_joelho_r, 40, 180)
                                    elif exercicio == 'Push-Up':
                                        check(image, ombro_l, cotovelo_l, angle_omb_l, 45, 160)
                                        check(image, ombro_r, cotovelo_r, angle_omb_r, 45, 160)
                                        check(image, cotovelo_l, pulso_l, angle_cotov_l, 70, 180)
                                        check(image, cotovelo_r, pulso_r, angle_cotov_r, 70, 180)
                                    elif exercicio == 'Sit-Up':
                                        check(image, ombro_l, cotovelo_l, angle_omb_l, 30, 180)
                                        check(image, ombro_r, cotovelo_r, angle_omb_r, 30, 180)
                                        check(image, cotovelo_l, pulso_l, angle_cotov_l, 40, 170)
                                        check(image, cotovelo_r, pulso_r, angle_cotov_r, 40, 170)

                                if calculate_distance:
                                    distancia_ombros = math.dist(ombro_r, ombro_l)
                                    distancia_ombro_quadr_r = math.dist(
                                        ombro_r, quadril_r
                                    )
                                    distancia_ombro_quadr_l = math.dist(
                                        ombro_l, quadril_l
                                    )

                                    distancia_ombros = round(distancia_ombros, 2)
                                    distancia_ombro_quadr_r = round(
                                        distancia_ombro_quadr_r, 2
                                    )
                                    distancia_ombro_quadr_l = round(
                                        distancia_ombro_quadr_l, 2
                                    )

                                    if (
                                        distancia_ombro_quadr_r_in / 1.1
                                        > distancia_ombro_quadr_r
                                        and distancia_ombro_quadr_l_in / 1.1
                                        > distancia_ombro_quadr_l
                                    ):
                                        print("Está a afastar!")
                                        calculate_distance_inicial = True
                                    elif (
                                        distancia_ombro_quadr_r_in * 1.1
                                        < distancia_ombro_quadr_r
                                        and distancia_ombro_quadr_l_in * 1.1
                                        < distancia_ombro_quadr_l
                                    ):
                                        print("Está a aproximar!")
                                        calculate_distance_inicial = True

                                    if calculate_distance_inicial:
                                        if (
                                            not esta_lado
                                            or distancia_ombros_in < distancia_ombros
                                        ):
                                            distancia_ombros_in = math.dist(
                                                ombro_r, ombro_l
                                            )
                                            esta_lado = False
                                            esta_lado_mid = False
                                        distancia_ombro_quadr_r_in = math.dist(
                                            ombro_r, quadril_r
                                        )
                                        distancia_ombro_quadr_l_in = math.dist(
                                            ombro_l, quadril_l
                                        )
                                        calculate_distance_inicial = False

                                    elif not calculate_distance_inicial:
                                        if distancia_ombros < distancia_ombros_in / 7:
                                            esta_lado = True
                                            print("Está de lado!")
                                        elif (
                                            distancia_ombros < distancia_ombros_in / 1.3
                                        ):
                                            print("Está de lado!")
                                            esta_lado = True
                                        else:
                                            print("Está de frente!")
                                            esta_lado = False
                                    calculate_distance = False

                                if half_second_timer >= 15:
                                    calculate_distance = True
                                    half_second_timer = 0
                                half_second_timer += 1

                            h, w, ch = image.shape
                            qimage = QImage(
                                image.data, w, h, ch * w, QImage.Format_RGB888
                            )
                            pixmap = QPixmap.fromImage(qimage)
                            image.flags.writeable = True

                            self.camera_label.setPixmap(pixmap)

                            QApplication.processEvents()

                        except Exception as e:
                            print(f"An error occurred: {str(e)}")

                self.mostrar_img_nao_disponivel()
                if self.stop_correction:
                    self.img_nao_disponivel.hide()
                    self.camera_label.clear()
                    self.camera_label.hide()
                    return
                return

            cap.release()
            cv2.destroyAllWindows()

        except cv2.error as e:
            self.camera_label.clear()
            self.mostrar_img_nao_disponivel()
            if self.stop_correction:
                self.img_nao_disponivel.hide()
                return
            return

    def classification(self):
        global counter, calculate_distance, half_second_timer, cap
        cap = cv2.VideoCapture(0)
        try:

            if not cap.isOpened():
                self.mostrar_img_nao_disponivel()
                return

            self.esconder_img_nao_disponivel()
            self.start_detect = True
            self.count_detect = 0
            self.last_detect = ""

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            with mp_pose.Pose(
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            ) as pose:
                while cap.isOpened():

                    if self.stop_classification:
                        self.camera_label.clear()
                        self.camera_label.hide()
                        break

                    ret, frame = cap.read()

                    if not ret:
                        break

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    image = cv2.flip(image, 1)

                    results = pose.process(image)

                    if results.pose_landmarks and results:
                        if half_second_timer >= 15:
                            if self.count_detect == 5:
                                self.new_img = QPixmap(names[int(c)] + ".jpg")
                                self.yes_butt.setEnabled(True)
                                self.exercicio = names[int(c)]
                                self.label_img_poss.setPixmap(self.new_img)
                            elif self.count_detect < 5:
                                result = model(frame)[0]
                                names = model.names
                                for c in result.boxes.cls:
                                    if (
                                        names[int(c)] == "Squat"
                                        or names[int(c)] == "Push-Up"
                                        or names[int(c)] == "Sit-Up"
                                    ):
                                        if self.start_detect:
                                            self.last_detect = names[int(c)]
                                            self.count_detect = 1
                                            self.start_detect = False
                                        else:
                                            if self.last_detect == names[int(c)]:
                                                self.count_detect += 1
                                            else:
                                                self.count_detect = 0
                                                self.start_detect = True

                                    else:
                                        self.start_detect = True
                                        self.count_detect = 0
                                        self.last_detect = ""
                                        self.new_img = QPixmap("NotFound.png")
                                        self.yes_butt.setEnabled(False)

                            calculate_distance = True
                            half_second_timer = 0
                        half_second_timer += 1

                        h, w, ch = image.shape
                        qimage = QImage(image.data, w, h, ch * w, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimage)
                        image.flags.writeable = True

                        self.camera_label.setPixmap(pixmap)

                        QApplication.processEvents()

                self.mostrar_img_nao_disponivel()
                if self.stop_classification:
                    self.img_nao_disponivel.hide()
                    self.camera_label.clear()
                    self.camera_label.hide()
                    return
                return

            cap.release()
            cv2.destroyAllWindows()

        except cv2.error as e:
            self.camera_label.clear()
            self.camera_label.hide()
            self.mostrar_img_nao_disponivel()
            if self.stop_classification:
                self.img_nao_disponivel.hide()
                return
            return

    def mostrar_img_nao_disponivel(self):
        self.img_nao_disponivel.show()
        self.camera_label.hide()

    def esconder_img_nao_disponivel(self):
        self.img_nao_disponivel.hide()
        self.camera_label.show()

    def closeEvent(self, event):
        cap.release()
        cv2.destroyAllWindows()
        QCoreApplication.quit()

    def start(self):
        self.hide_butt.setGeometry(100, 200, 400, 40)
        self.inf_butt.hide()
        self.label_cho.hide()
        self.conf_butt.hide()
        self.play_butt.hide()
        self.stop_butt.hide()
        self.label.hide()
        self.gif_label.hide()
        self.label_img_sit.hide()
        self.label_img_squat.hide()
        self.label_img_push.hide()
        self.label_text_push.hide()
        self.label_text_squat.hide()
        self.label_text_sit.hide()
        self.label_exe_text.show()
        self.label_img_poss.show()
        self.yes_butt.show()
        self.no_butt.show()
        self.change_butt.hide()
        self.label_inf_text.show()
        self.hide_butt.show()
        self.new_img = QPixmap("NotFound.png")
        self.label_img_poss.setPixmap(self.new_img)
        self.stop_classification = False
        self.stop_correction = False
        self.classification()
        self.update()

    def back(self):
        self.hide_butt.setGeometry(100, 200, 400, 40)
        self.play_butt.hide()
        self.stop_butt.hide()
        self.inf_butt.show()
        self.inf_page.hide()
        self.inf_page_man.hide()
        self.gif_label.hide()
        self.label.hide()
        self.label_cho.show()
        self.conf_butt.show()
        self.label_img_sit.show()
        self.label_img_squat.show()
        self.label_img_push.show()
        self.label_text_push.show()
        self.label_text_squat.show()
        self.label_text_sit.show()
        self.label_exe_text.hide()
        self.label_img_poss.hide()
        self.yes_butt.hide()
        self.change_butt.hide()
        self.no_butt.hide()
        self.hide_butt.hide()
        self.camera_label.hide()
        self.label_inf_text.hide()
        self.stop_classification = True
        self.stop_correction = True
        self.esconder_img_nao_disponivel()

    def correct(self):
        self.stop_correction = False
        self.stop_classification = True
        self.hide_butt.show()
        self.gif_label.show()
        self.label.show()
        self.change_butt.show()
        self.label_exe_text.hide()
        self.label_img_poss.hide()
        self.yes_butt.hide()
        self.no_butt.hide()
        self.play_butt.show()
        self.stop_butt.show()
        self.label_inf_text.hide()
        self.correction(self.exercicio)
        self.update()

    def info(self):
        self.inf_butt.hide()
        self.label_cho.hide()
        self.conf_butt.hide()
        self.play_butt.hide()
        self.stop_butt.hide()
        self.label.hide()
        self.hide_butt.show()
        self.hide_butt.setGeometry(100, 900, 400, 40)
        self.gif_label.hide()
        self.label_img_sit.hide()
        self.label_img_squat.hide()
        self.label_img_push.hide()
        self.label_text_push.hide()
        self.label_text_squat.hide()
        self.label_text_sit.hide()
        self.change_butt.hide()
        self.inf_page.show()
        self.inf_page_man.show()

    def __init__(self):
        super().__init__()

        self.new_mov = None
        self.start_detect = True
        self.count_detect = 0
        self.last_detect = ""
        self.exercicio = ""
        self.stop_classification = False
        self.stop_correction = False

        # Características da window principal
        self.palette = QPalette()
        self.palette.setColor(QPalette.Background, QColor(17, 16, 16))
        self.setPalette(self.palette)
        self.setWindowTitle("AiSweat")
        self.setWindowIcon(QIcon("logo.png"))
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Imagem logo central da window
        self.image_label = QLabel(self)
        self.original_pixmap = QPixmap("logo.png")
        self.desired_size = QSize(300, 200)
        self.pixmap = self.original_pixmap.scaled(self.desired_size, Qt.KeepAspectRatio)
        self.image_label.setPixmap(self.pixmap)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        # Texto de opções
        self.label_cho = QLabel(self)
        self.label_cho.setText("Opções de exercícios para realizar:")
        self.label_cho.setGeometry(100, 200, 400, 60)
        self.label_cho.resize(400, 30)
        self.label_cho.setFont(QFont("Times", 12))
        self.color_effect = QGraphicsColorizeEffect()
        self.color_effect.setColor(Qt.white)
        self.label_cho.setGraphicsEffect(self.color_effect)

        # Texto sob imagem Push-Up
        self.label_text_push = QLabel(self)
        self.label_text_push.setText("Push-Up")
        self.label_text_push.setGeometry(330, 250, 200, 60)
        self.label_text_push.setFont(QFont("Times", 15))
        self.color_effect = QGraphicsColorizeEffect()
        self.color_effect.setColor(Qt.white)
        self.label_text_push.setGraphicsEffect(self.color_effect)

        # Imagem Push-Up
        self.label_img_push = QLabel(self)
        self.label_img_push.setScaledContents(True)
        self.label_img_push.setGeometry(100, 300, 540, 500)
        self.pixmap_push = QPixmap("Push-Up.jpg")
        self.label_img_push.setPixmap(self.pixmap_push)

        # Texto sob imagem Squat
        self.label_text_squat = QLabel(self)
        self.label_text_squat.setText("Squat")
        self.label_text_squat.setGeometry(930, 250, 200, 60)
        self.label_text_squat.setFont(QFont("Times", 15))
        self.color_effect = QGraphicsColorizeEffect()
        self.color_effect.setColor(Qt.white)
        self.label_text_squat.setGraphicsEffect(self.color_effect)

        # Imagem Squat
        self.label_img_squat = QLabel(self)
        self.label_img_squat.setScaledContents(True)
        self.label_img_squat.setGeometry(680, 300, 540, 500)
        self.pixmap_squat = QPixmap("Squat.jpg")
        self.label_img_squat.setPixmap(self.pixmap_squat)

        # Texto sob imagem Sit-Up
        self.label_text_sit = QLabel(self)
        self.label_text_sit.setText("Sit-Up")
        self.label_text_sit.setGeometry(1500, 250, 200, 60)
        self.label_text_sit.setFont(QFont("Times", 15))
        self.color_effect = QGraphicsColorizeEffect()
        self.color_effect.setColor(Qt.white)
        self.label_text_sit.setGraphicsEffect(self.color_effect)

        # Imagem SitUp
        self.label_img_sit = QLabel(self)
        self.label_img_sit.setScaledContents(True)
        self.label_img_sit.setGeometry(1260, 300, 540, 500)
        self.pixmap_sit = QPixmap("SitUp.jpg")
        self.label_img_sit.setPixmap(self.pixmap_sit)

        # Botão de confirmação
        self.conf_butt = QPushButton(self)
        self.palette_butt = QPalette()
        self.palette_butt.setColor(QPalette.Background, QColor(17, 16, 16))
        self.conf_butt.setText("Começar")
        self.conf_butt.setFont(QFont("Times", 12))
        self.conf_butt.setStyleSheet(
            "background-color: #EF6E17; border-radius: 5px; padding: 10px; color: white;"
        )
        self.conf_butt.setGeometry(100, 820, 400, 40)
        self.conf_butt.clicked.connect(self.start)

        # Botão de informações
        self.inf_butt = QPushButton(self)
        self.palette_butt = QPalette()
        self.palette_butt.setColor(QPalette.Background, QColor(17, 16, 16))
        self.inf_butt.setText("Como Usar")
        self.inf_butt.setFont(QFont("Times", 12))
        self.inf_butt.setStyleSheet(
            "background-color: #EF6E17; border-radius: 5px; padding: 10px; color: white;"
        )
        self.inf_butt.setGeometry(100, 880, 400, 40)
        self.inf_butt.clicked.connect(self.info)

        # Botão de mudar exercício
        self.change_butt = QPushButton(self)
        self.palette_butt_hide = QPalette()
        self.palette_butt_hide.setColor(QPalette.Background, QColor(17, 16, 16))
        self.change_butt.setText("Mudar Exercício")
        self.change_butt.setFont(QFont("Times", 12))
        self.change_butt.setStyleSheet(
            "background-color: #EF6E17; border-radius: 5px; padding: 10px; color: white;"
        )
        self.change_butt.setGeometry(100, 300, 400, 40)
        self.change_butt.clicked.connect(self.start)
        self.change_butt.hide()

        # Texto indicativo de ação
        self.label = QLabel(self)
        self.label.setText("Exemplificação do exercicio a realizar:")
        self.label.setGeometry(100, 450, 400, 60)
        self.label.resize(400, 30)
        self.label.setFont(QFont("Times", 12))
        self.color_effect = QGraphicsColorizeEffect()
        self.color_effect.setColor(Qt.white)
        self.label.setGraphicsEffect(self.color_effect)
        self.label.hide()

        # Gifs identificativos
        self.movie = QMovie("Squat.gif")
        self.gif_label = QLabel(self)
        self.desired_size = QSize(400, 340)
        self.movie.setScaledSize(self.desired_size)
        self.gif_label.setMovie(self.movie)
        self.gif_label.setGeometry(
            100, 500, self.desired_size.width(), self.desired_size.height()
        )
        self.gif_label.hide()

        # Botao de parar gif
        self.stop_butt = QPushButton(self)
        self.original_pixmap = QPixmap("stop.png")
        self.desired_size = QSize(400, 300)
        self.pixmap = self.original_pixmap.scaled(self.desired_size, Qt.KeepAspectRatio)
        self.stop_butt.setIcon(QIcon(self.pixmap))
        self.stop_butt.setStyleSheet(
            "background-color: #EF6E17; border-radius: 5px; padding: 10px; color: white;"
        )
        self.stop_butt.setGeometry(100, 860, 180, 50)
        self.stop_butt.clicked.connect(self.movie.stop)
        self.stop_butt.hide()

        # Botao de play gif
        self.play_butt = QPushButton(self)
        self.original_pixmap = QPixmap("play.png")
        self.desired_size = QSize(400, 300)
        self.pixmap = self.original_pixmap.scaled(self.desired_size, Qt.KeepAspectRatio)
        self.play_butt.setIcon(QIcon(self.pixmap))
        self.play_butt.setStyleSheet(
            "background-color: #EF6E17; border-radius: 5px; padding: 10px; color: white;"
        )
        self.play_butt.setGeometry(320, 860, 180, 50)
        self.play_butt.clicked.connect(self.movie.start)
        self.play_butt.hide()

        # Camera e suas caractirísticas
        self.camera_label = QLabel(self)
        self.camera_label.setGeometry(600, 170, 1280, 800)
        self.camera_label.hide()

        # Imagem quando a câmera não está disponível
        self.img_nao_disponivel = QLabel(self)
        self.img_nao_disponivel.setScaledContents(True)
        self.img_nao_disponivel.setGeometry(600, 200, 1280, 715)
        self.img_nao_disponivel_img = QPixmap("webnotworking.png")
        self.img_nao_disponivel.setPixmap(self.img_nao_disponivel_img)
        self.img_nao_disponivel.hide()

        # Texto indicativo de como posicionar
        self.label_inf_text = QLabel(self)
        self.label_inf_text.setText(
            "Posicione-se afastado da câmera, até ser \ndetetado um exercício"
        )
        self.label_inf_text.setGeometry(100, 280, 400, 400)
        self.label_inf_text.resize(400, 50)
        self.label_inf_text.setFont(QFont("Times", 12))
        self.color_effect = QGraphicsColorizeEffect()
        self.color_effect.setColor(Qt.white)
        self.label_inf_text.setGraphicsEffect(self.color_effect)
        self.label_inf_text.hide()

        # Texto indicativo da ação
        self.label_exe_text = QLabel(self)
        self.label_exe_text.setText("Está a realizar este exercício?")
        self.label_exe_text.setGeometry(100, 350, 400, 60)
        self.label_exe_text.resize(400, 30)
        self.label_exe_text.setFont(QFont("Times", 12))
        self.color_effect = QGraphicsColorizeEffect()
        self.color_effect.setColor(Qt.white)
        self.label_exe_text.setGraphicsEffect(self.color_effect)
        self.label_exe_text.hide()

        # Botão de voltar ao menu
        self.hide_butt = QPushButton(self)
        self.palette_butt_hide = QPalette()
        self.palette_butt_hide.setColor(QPalette.Background, QColor(17, 16, 16))
        self.hide_butt.setText("Menu Inicial")
        self.hide_butt.setFont(QFont("Times", 12))
        self.hide_butt.setStyleSheet(
            "background-color: #EF6E17; border-radius: 5px; padding: 10px; color: white;"
        )
        self.hide_butt.setGeometry(100, 200, 400, 40)
        self.hide_butt.clicked.connect(self.back)
        self.hide_butt.hide()

        # Imagem de possível ação
        self.label_img_poss = QLabel(self)
        self.label_img_poss.setScaledContents(True)
        self.label_img_poss.setGeometry(100, 400, 400, 400)
        self.pixmap_push = QPixmap("NotFound.png")
        self.label_img_poss.setPixmap(self.pixmap_push)
        self.label_img_poss.hide()

        # Botao confirmar
        self.yes_butt = QPushButton(self)
        self.yes_butt.setText("Sim")
        self.desired_size = QSize(400, 300)
        self.pixmap = self.original_pixmap.scaled(self.desired_size, Qt.KeepAspectRatio)
        self.yes_butt.setFont(QFont("Times", 12))
        self.yes_butt.setStyleSheet(
            "background-color: #EF6E17; border-radius: 5px; padding: 10px; color: white;"
        )
        self.yes_butt.setGeometry(100, 850, 180, 50)
        self.yes_butt.clicked.connect(self.correct)
        self.yes_butt.setEnabled(False)
        self.yes_butt.hide()

        # Botao negar
        self.no_butt = QPushButton(self)
        self.no_butt.setText("Não")
        self.desired_size = QSize(400, 300)
        self.no_butt.setFont(QFont("Times", 12))
        self.pixmap = self.original_pixmap.scaled(self.desired_size, Qt.KeepAspectRatio)
        self.no_butt.setStyleSheet(
            "background-color: #EF6E17; border-radius: 5px; padding: 10px; color: white;"
        )
        self.no_butt.setGeometry(320, 850, 180, 50)
        self.no_butt.clicked.connect(self.start)
        self.no_butt.hide()

        # Página Informações
        self.inf_page = QLabel(self)
        self.inf_page.setText(
            "Recomendações de utilização da aplicação:\n\n\n"
            "1- Manter uma distância de cerca de 2 a 3 metros da câmera.\n\n"
            "2- Manter a câmera a uma altura dos joelhos.\n\n"
            "3- Posicionar-se de forma a captar todo o corpo.\n\n"
            "4- Manter o local bem iluminado.\n\n"
            "5- Preferencialmente, realizar os exercícios de frente para a câmera.\n\n"
            "6- Remover todos os objetos que se encontrem entre a câmera e o utilizador.\n\n"
            "7 - Não existirem fontes luminosas em locais vísiveis pela câmara."
        )
        self.inf_page.setGeometry(1200, 250, 700, 400)
        self.inf_page.setFont(QFont("Times", 12))
        self.color_effect = QGraphicsColorizeEffect()
        self.color_effect.setColor(Qt.white)
        self.inf_page.setGraphicsEffect(self.color_effect)
        self.inf_page.hide()

        # Manual Intruções
        self.inf_page_man = QLabel(self)
        self.inf_page_man.setText(
            "Como Usar:\n\n\n"
            "O objetivo da aplicação é realizar a deteção de um exercício e corrigir\na postura durante a realização do mesmo\n\n"
            "Para isso é disponibilizado um botão Começar.\n\n"
            "Ao clicar nesse botão, o processo de deteção é iniciado.\n\n"
            "Após isso, o utilizador deve permanecer cerca de 2,5 segundos na posíção\ndo exercício que pretende realizar\n\n"
            "Ao detetar um exercício, ficará disponível a opção de corrigir, clicando em sim.\nCaso o utlizador mude de ideias, ou seja feita uma deteção incorreta o utilizador pode clicar em não.\n\n"
            "No processo de correção, é disponibilizado um gif do execício em questão.\n\n"
            "Nesta página o utilizador consegue observar os locais onde está a fazer\nmal o exercício, indicados através de uma linha vermelha sobre o membro.\n\n"
            "É disponibilizada a opção de mudar de exercício, voltando a ser feita a\netapa de identificação."
            "Em todas as páginas é disponibilizado um botão para regressar ao menu inicial."
        )
        self.inf_page_man.setGeometry(100, 250, 1000, 600)
        self.inf_page_man.setFont(QFont("Times", 12))
        self.color_effect = QGraphicsColorizeEffect()
        self.color_effect.setColor(Qt.white)
        self.inf_page_man.setGraphicsEffect(self.color_effect)
        self.inf_page_man.hide()

        self.movie.start()
        self.showMaximized()
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Menu()
    sys.exit(app.exec_())
