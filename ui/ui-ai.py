import time
import pandas as pd
import os
import sys
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import random
import pyautogui
import json
import re
from scipy.ndimage import gaussian_filter
import pickle
import cv2

from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QFont, QKeySequence
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer, QTime
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QShortcut
from PyQt5 import uic, QtWidgets

from utils import truncate_hu, normalazation
import pylink
import pygame
from pygame.locals import *
from string import ascii_letters, digits
from calibrate import Calibrate
from CalibrationGraphicsPygame import CalibrationGraphics


import pdb



class ImageLabel(QLabel):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)

        self.main_window = main_window
        self.start_pos = None
        self.end_pos = None
        self.pix_map = None
        self.raw_image = None
        self.image_scale = 5
        self.slice_num = 0
        self.line1 = []
        self.line2 = []
        self.line2_slope = []
        self.draw_first_line = True
        self.draw_second_line = False
        self.image_position = QPoint(0, 0)
        self.image_array = None
        self.mean_density = []
        self.ellipses = []
        self.recordline = []
        self.measurement = []
        self.mode = "default"
        self.spacing = None
        self.point = None
        self.heatmap_group = {}
        self.box_missed_state = None
        self.raw_gaze_all = None
        self.show_box_flag = False
        self.showlooked_flag = False
        self.current_patient_id = None
        self.boxes = []
        self.dragging_for_scroll = False
        self.last_drag_y = None
        self.drawn_boxes = []
        self.box_start_end = []
        self.json_data = None

    def set_point(self, point):
        self.point = point
        self.update() # redraw the widget

    def set_image_array(self, image_array):
        self.image_array = image_array

    def set_mode(self, mousemode):
        self.mode = mousemode

    def mousePressEvent(self, event):
        if self.mode == "default" and event.button() == Qt.LeftButton:
            self.dragging_for_scroll = True
            self.last_drag_y = event.pos().y()

        elif self.mode == "density" and event.button() == Qt.LeftButton:
            self.ellipses.append((event.pos(), event.pos(), self.slice_num))
            self.mean_density.append(0)
            self.update()

        elif self.mode == "volume" and event.button() == Qt.LeftButton:
            self.line1.append((event.pos(), event.pos(), self.slice_num))
            self.update()            

        elif self.mode == "default" and event.button() == Qt.RightButton:
            start_pos = event.pos()
            end_pos = event.pos()
            self.box_start_end.append((start_pos, end_pos, self.slice_num))
            self.update()

    def mouseMoveEvent(self, event):
        if self.mode == "default" and self.dragging_for_scroll:
            current_y = event.pos().y()
            delta_y = current_y - self.last_drag_y

            # Define how many pixels per slice
            sensitivity = 5
            if abs(delta_y) >= sensitivity:
                direction = 1 if delta_y > 0 else -1
                self.main_window.scrollbar.setValue(self.main_window.scrollbar.value() + direction)
                self.last_drag_y = current_y  # update only after scrolling
            self.update()

        elif self.mode == "density" and event.buttons() == Qt.LeftButton:
            self.ellipses[-1] = (self.ellipses[-1][0], event.pos(), self.slice_num)
            self.mean_density[-1] = self.calculate_mean_density()
            self.update()

        elif self.mode == "volume" and event.buttons() == Qt.LeftButton:
            self.line1[-1] = (self.line1[-1][0], event.pos(), self.slice_num)
            self.update()

        elif self.mode == "default" and event.buttons() == Qt.RightButton:
            if self.box_start_end != []:
                self.box_start_end[-1] = (self.box_start_end[-1][0], event.pos(), self.slice_num)
            self.update()

    def mouseReleaseEvent(self, event):
        if self.mode == "default" and event.button() == Qt.LeftButton:
            self.dragging_for_scroll = False
            self.last_drag_y = None
            self.update()
        elif event.button() == Qt.LeftButton and self.mode == "density":
            self.ellipses[-1] = (self.ellipses[-1][0], event.pos(), self.slice_num)
            self.update()
            print("density measure")
        elif event.button() == Qt.LeftButton and self.mode == "volume":
            self.line1[-1] = (self.line1[-1][0], event.pos(), self.slice_num)
            self.update()
            print("volume measure")    
        elif event.button() == Qt.RightButton and self.mode == "default":
            if self.box_start_end != []:
                self.box_start_end[-1] = (self.box_start_end[-1][0], event.pos(), self.slice_num)
                start_pos = self.box_start_end[-1][0]; end_pos = self.box_start_end[-1][1]
                x0 = int((min(start_pos.x(), end_pos.x()) - self.image_position[0]) / self.image_scale)
                x1 = int((max(start_pos.x(), end_pos.x()) - self.image_position[0]) / self.image_scale)
                y0 = int((min(start_pos.y(), end_pos.y()) - self.image_position[1]) / self.image_scale)
                y1 = int((max(start_pos.y(), end_pos.y()) - self.image_position[1]) / self.image_scale)
                self.drawn_boxes.append([self.slice_num, x0, y0, self.slice_num, x1, y1])
                self.measurement.append([self.slice_num, x0, y0, self.slice_num, x1, y1])
                print(f"Box recorded: {self.drawn_boxes[-1]}")
                self.update()        

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

        if self.ellipses:
            for ellipse, mean_density in zip(self.ellipses, self.mean_density):
                start_pos, end_pos, slice_num = ellipse
                if slice_num != self.slice_num:
                    continue
                rect = QRect(start_pos, end_pos)
                painter.setPen(QColor("red"))
                painter.setBrush(Qt.transparent)
                painter.drawEllipse(rect)
                if mean_density is not None:
                    text_pos = QPoint(end_pos.x()+1, start_pos.y()+1)
                    painter.drawText(text_pos, f"{mean_density:.2f}")

        if self.line1:
            for line1 in self.line1:
                start_pos, end_pos, slice_num = line1
                if slice_num != self.slice_num:
                    continue
                painter.setPen(QColor("red"))
                painter.drawLine(start_pos, end_pos)
                text_pos = QPoint(start_pos.x() - 10, start_pos.y() - 10)
                painter.setFont(QFont("Arial", 10))
                painter.drawText(text_pos, f"{self.calculate_line_length(start_pos, end_pos):.2f}mm")

        if self.box_start_end:
            for box_start_end in self.box_start_end:
                start_pos, end_pos, current_slice = box_start_end
                if current_slice == self.slice_num:
                    rect = QRect(start_pos, end_pos)
                    painter.setPen(QColor('lightgreen'))
                    painter.setBrush(Qt.transparent)
                    painter.drawRect(rect)


        if self.show_box_flag and self.box_missed_state is None:
            for box in self.boxes:
                z0, x0, y0, z1, x1, y1 = box
                if z0 <= self.slice_num <= z1:
                    x = (min(x0, x1)-5)*self.image_scale + self.image_position[0]
                    y = (min(y0, y1)-5)*self.image_scale + self.image_position[1]
                    w = (abs(x1 - x0)+10)*self.image_scale
                    h = (abs(y1 - y0)+10)*self.image_scale
                    painter.setPen(QColor("yellow"))
                    painter.drawRect(x, y, w, h)

        elif self.show_box_flag and self.box_missed_state is not None:
            for box in self.boxes:
                z0, x0, y0, z1, x1, y1 = box
                if z0 <= self.slice_num <= z1:
                    x = min(x0, x1)*self.image_scale + self.image_position[0]
                    y = min(y0, y1)*self.image_scale + self.image_position[1]
                    w = abs(x1 - x0)*self.image_scale
                    h = abs(y1 - y0)*self.image_scale
                    if self.box_missed_state[self.current_patient_id][tuple(box)]:
                        color = QColor(("yellow"))
                        painter.setPen(QPen(color, 1))
                        painter.drawRect(x, y, w, h)
                    else:
                        continue


        if self.showlooked_flag:
            if self.json_data is not None:
                annotated_boxes = self.json_data.get(self.current_patient_id, [])
                for box in annotated_boxes:
                    z0, x0, y0, z1, x1, y1 = box
                    if z0 <= self.slice_num <= z1:
                        x = min(x0, x1)*self.image_scale + self.image_position[0]
                        y = min(y0, y1)*self.image_scale + self.image_position[1]
                        w = abs(x1 - x0)*self.image_scale
                        h = abs(y1 - y0)*self.image_scale
                        painter.setPen(QPen(QColor("lightblue"), 2))
                        painter.drawRect(x, y, w, h)

        # if self.showlooked_flag and self.current_patient_id in self.heatmap_group:
        #     heatmap_list = self.heatmap_group[self.current_patient_id]
        #     if 0 <= self.slice_num < len(heatmap_list):
        #         heatmap = heatmap_list[self.slice_num]

        #         # Threshold and normalize
        #         heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        #         threshold = 25  # adjust based on your heatmap intensity scale
        #         mask = heatmap_norm > threshold

        #         # Apply colormap
        #         heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)

        #         # Set background transparent where below threshold
        #         heatmap_color[~mask] = [0, 0, 0]  # or leave unchanged
        #         alpha_mask = (mask.astype(np.uint8) * 128).astype(np.uint8)  # 0~255, semi-transparent

        #         # Convert to QImage with alpha channel
        #         h, w = heatmap_color.shape[:2]
        #         heatmap_rgba = np.concatenate([heatmap_color, alpha_mask[..., None]], axis=-1)
        #         qimage = QImage(heatmap_rgba.data, w, h, QImage.Format_RGBA8888)

        #         # Convert to QPixmap and draw scaled
        #         pix = QPixmap.fromImage(qimage)
        #         painter.drawPixmap(self.image_position[0], self.image_position[1],
        #                         pix.scaled(w * self.image_scale, h * self.image_scale))

    def bigger(self):
        self.image_scale += 0.5 
        self.update_image(self.pix_map, self.slice_num)
        print("change image scale to: {}".format(self.image_scale))

    def smaller(self):
        self.image_scale -= 0.5 
        self.update_image(self.pix_map, self.slice_num)
        print("change image scale to: {}".format(self.image_scale))

    def set_patient_id(self, pid):
        self.current_patient_id = pid

    def calculate_line_length(self, point1, point2):
        point1_world = nib.affines.apply_affine(self.spacing, [point1.x() / self.image_scale, point1.y() / self.image_scale, self.slice_num])
        point2_world = nib.affines.apply_affine(self.spacing, [point2.x() / self.image_scale, point2.y() / self.image_scale, self.slice_num])
        return np.linalg.norm(point2_world - point1_world)


    def calculate_mean_density(self):
        if self.image_array is not None and len(self.ellipses) > 0:
            ellipse = self.ellipses[-1]
            start_pos, end_pos, _ = ellipse
        
            # Calculate the rectangle in image space
            x1 = int((start_pos.x() - self.image_position[0]) / self.image_scale)
            y1 = int((start_pos.y() - self.image_position[1]) / self.image_scale)
            x2 = int((end_pos.x() - self.image_position[0]) / self.image_scale)
            y2 = int((end_pos.y() - self.image_position[1]) / self.image_scale)

            if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                return float('nan')
            
            # Crop the image array to the rectangle
            cropped_image = self.image_array \
                [y1:y2, x1:x2]
            
            # Calculate the mean density value
            return np.mean(cropped_image)

    def reset_image(self, scale=None):
        self.image_scale = 5 if scale is None else scale
        self.image_position = QPoint(0, 0)
        self.ellipses = []
        self.mean_density = []
        self.line1 = []
        self.line2 = []
        self.line2_slope = []
        self.recordline = []
        self.draw_first_line = True
        self.draw_second_line = False
        self.mode = "default"
        self.update_image(self.pix_map, self.slice_num)
        print("reset image")

    def update_image(self, pixmap, slice_num=0):
        self.pix_map = pixmap
        self.slice_num = slice_num

        scaled_size = self.pix_map.size() * self.image_scale
        self.setPixmap(pixmap.scaled(scaled_size))
        self.set_image_array(self.raw_image[self.slice_num, ...])
        label_width = self.width()
        label_height = self.height()
        image_width = scaled_size.width()
        image_height = scaled_size.height()
        x_offset = (label_width - image_width) / 2
        y_offset = (label_height - image_height) / 2
        self.image_position = (x_offset, y_offset)



class CustomInputDialog(QInputDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QInputDialog { background-color: white; }")

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self, uifile='ui-nu.ui'):
        super().__init__()

        # Load the UI file
        uic.loadUi(uifile, self)
        self.setStyleSheet("QMainWindow { background-color: black; }")  # Set the window background color

        screen_width, screen_height = pyautogui.size()
        print("**screen size:{}, {}".format(screen_width, screen_height), "**")\
        
        screen_rect = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(screen_rect)

        self.layoutwin = self.findChild(QHBoxLayout, 'horizontalLayout_3')
        self.file_button = self.findChild(QPushButton, 'file')
        self.calibration_button = self.findChild(QPushButton, 'calibration')
        self.starteyetracking_button = self.findChild(QPushButton, 'starteyetracking')
        self.stopeyetracking_button = self.findChild(QPushButton, 'stopeyetracking')
        self.nextcase_button = self.findChild(QPushButton, 'nextcase')
        self.lastcase_button = self.findChild(QPushButton, 'lastcase')
        self.nextmissing_button = self.findChild(QPushButton, 'nextmissing')
        self.lastmissing_button = self.findChild(QPushButton, 'lastmissing')
        self.showbox_button = self.findChild(QPushButton, 'showbox')
        self.showlooked_button = self.findChild(QPushButton, 'showlooked')
        self.pause_button = self.findChild(QPushButton, 'pause')
        self.close_button = self.findChild(QPushButton, 'closeButton')
        self.scrollbar = self.findChild(QtWidgets.QScrollBar, 'ScrollBar')
        self.counter = self.findChild(QLabel, 'counter')
        self.scrollbar.setHidden(True)

        self.file_button.setStyleSheet("background-color: white; color: black;")
        self.calibration_button.setStyleSheet("background-color: white; color: black;")
        self.starteyetracking_button.setStyleSheet("background-color: white; color: black;")
        self.stopeyetracking_button.setStyleSheet("background-color: white; color: black;")
        self.pause_button.setStyleSheet("background-color: white; color: black;")
        self.showbox_button.setStyleSheet("background-color: white; color: black;")
        self.showlooked_button.setStyleSheet("background-color: white; color: black;")
        self.close_button.setStyleSheet("background-color: white; color: black;")
        self.nextcase_button.setStyleSheet("background-color: white; color: black;")
        self.lastcase_button.setStyleSheet("background-color: white; color: black;")
        self.nextmissing_button.setStyleSheet("background-color: white; color: black;")
        self.lastmissing_button.setStyleSheet("background-color: white; color: black;")
        self.counter.setStyleSheet("background-color: black; color: white;")

        self.oldlabel = self.findChild(QLabel, 'label')
        self.label = ImageLabel(main_window=self, parent=self)

        self.label.setObjectName("label")
        self.layoutwin.replaceWidget(self.oldlabel, self.label)
        self.oldlabel.deleteLater()
        self.showFullScreen()

        # define function of the widgets
        self.file_button.clicked.connect(self.choose_image)
        self.calibration_button.clicked.connect(self.doCalibrate)
        self.starteyetracking_button.clicked.connect(self.thread)
        self.stopeyetracking_button.clicked.connect(self.stopthread)
        self.nextcase_button.clicked.connect(self.next_case)
        self.showbox_button.clicked.connect(self.show_box)
        self.showlooked_button.clicked.connect(self.show_looked)
        self.lastcase_button.clicked.connect(self.last_case)
        self.nextmissing_button.clicked.connect(self.next_miss)
        self.lastmissing_button.clicked.connect(self.last_miss)
        self.pause_button.clicked.connect(self.pause_operation)
        self.close_button.clicked.connect(self.close_operation)
        self.scrollbar.valueChanged.connect(self.on_scrollbar_value_changed)

        # keyboard shortcuts
        QShortcut(QKeySequence("Up"), self, self.label.bigger)
        QShortcut(QKeySequence("Down"), self, self.label.smaller)

        # define variables
        self.imgpath = None
        self.pixmap = None
        self.image_position = QPoint(0, 0)
        self.raw_img = None
        self.mode = "lung"
        self.start_pos = None  # Stores the starting position of the ellipse
        self.end_pos = None  # Stores the ending position of the ellipse    
        self.eyetracker = None
        self.csvname = None
        self.data = []
        self.measurement = {}
        self.current_patient = 0
        self.trial = 1
        self.boxes = []
        self.missed_slices = None
        self.showbox_flag = False
        self.endRecord_flag = True
        self.result_dir = None
        self.patient_list = []
        self.nodules_id_dict = {}
        self.id = None
        self.json_data = None
        sys.excepthook = self.exceptionHandler

        self.timer = QTimer()
        self.timer.timeout.connect(self.record)

    def exceptionHandler(self, exc_type, exc_value, exc_traceback):
        # This method will be called on unhandled exceptions
        # Add your code here to handle the exception and save data

        self.save_incremental()
        print(self.current_patient)

        # Print the exception to the console (optional)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

        # Exit the application
        sys.exit()

    def thread(self):
        print("----------------------start eye tracking----------------------")
        self.timer.start(10)
        if self.data != []:
            self.counter.setStyleSheet("background-color: green; color: white;")
        self.endRecord_flag = False

    def stopthread(self):
        print("----------------------stop eye tracking----------------------")
        self.timer.stop()
        if self.label.measurement != []:
            self.measurement[str(self.id)] = self.label.measurement
            self.label.measurement = []
            self.label.box_start_end = []
        self.save_incremental()
        self.eyetracker.close()
        pylink.closeGraphics()
        self.counter.setStyleSheet("background-color: black; color: white;")
        self.endRecord_flag = True

    def pause_operation(self):
        print("----------------------pause eye tracking----------------------")
        self.timer.stop()
        self.save_incremental()
        self.counter.setStyleSheet("background-color: yellow; color: black;")

    def close_operation(self):
        print("----------------------close eye tracking----------------------")
        self.timer.stop()

        # Save current annotation before closing
        if self.label.measurement != []:
            self.measurement[str(self.id)] = self.label.measurement
            self.label.measurement = []
            self.label.box_start_end = []

        if self.data != []:
            self.save_incremental()
        self.eyetracker.close()
        pylink.closeGraphics()
        self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.label.reset_image()
        elif event.key() == Qt.Key_1:
            self.change_contrast("soft_tissue")
        elif event.key() == Qt.Key_2:
            self.change_contrast("lung")
        elif event.key() == Qt.Key_3:
            self.change_contrast("bone")
        elif event.key() == Qt.Key_Q:
            self.label.set_mode("default" if self.label.mode == "density" else "density")
        elif event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            if self.label.box_start_end:
                self.label.box_start_end.pop()
            if self.label.drawn_boxes:
                self.label.drawn_boxes.pop()
            if self.label.measurement:
                self.label.measurement.pop()
            self.label.update()
            print("Undo last box")
        elif event.key() == Qt.Key_W:
            self.label.set_mode("default" if self.label.mode == "volume" else "volume")

    def wheelEvent(self, event):
        if self.pixmap == None:
            return
        scroll_amount = event.angleDelta().y() / 120
        self.scrollbar.setValue(self.scrollbar.value() - int(scroll_amount))
        self.pixmap = self.show3dimg()
        self.label.update_image(self.pixmap, self.raw_img.shape[0] - 1 - self.scrollbar.value())

    def on_scrollbar_value_changed(self):
        self.pixmap = self.show3dimg()
        self.label.update_image(self.pixmap, self.raw_img.shape[0] - 1 - self.scrollbar.value())

    def choose_image(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder', '')
        if not folder:
            return
        self.patient_list = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(('nii','nii.gz'))]
        self.imgpath = self.patient_list[self.current_patient]
        filename = self.imgpath
        print("**image name:", self.imgpath, "**")
        self.id = os.path.basename(self.imgpath).replace("_vol.nii.gz", "")
        self.label.set_patient_id(self.id)

        # read nodule box json file
        self.nodulepath = folder.replace("vol", "ai_nodules.json")
        print("**nodule name:", self.nodulepath, "**")
        if self.nodulepath:
            with open(self.nodulepath, "r") as file:
                self.nodules_id_dict = json.load(file)

        # find the boxes according to id
        self.boxes = self.nodules_id_dict[self.id]
        self.label.boxes = self.boxes

        _, ext = os.path.splitext(filename)
        if filename:
            if ext.lower() == '.gz' or ext.lower() == '.nii':
                sitkimage = sitk.ReadImage(filename)
                img = sitk.GetArrayFromImage(sitkimage)
                print("**image shape:", img.shape, "**")
                self.img_shape = img.shape
                self.raw_img = img
                self.label.raw_image = img
                self.scrollbar.setRange(0, self.raw_img.shape[0] - 1)
                self.counter.setText(f"{self.current_patient + 1}")
                self.pixmap = self.show3dimg()

                nii_img = nib.load(filename)
                self.label.spacing = nii_img.affine
                self.label.setAlignment(Qt.AlignCenter)
                self.label.update_image(self.pixmap, self.raw_img.shape[0] - 1 - self.scrollbar.value())


    def last_case(self):
        if self.current_patient == 0:
            print("**this is the first patient**")
            pass
        else:
            self.label.box_start_end = []
            self.current_patient -= 1
            self.counter.setText(f"{self.current_patient + 1}")
            self.scrollbar.setValue(0)
            self.imgpath = self.patient_list[self.current_patient]
            filename = self.imgpath
            print("**image name:", self.imgpath, "**")
            self.id = os.path.basename(self.imgpath).replace("_vol.nii.gz", "")
            self.label.set_patient_id(self.id)

            # find the boxes according to id
            self.boxes = self.nodules_id_dict[self.id]
            self.label.boxes = self.boxes

            _, ext = os.path.splitext(filename)
            if filename:
                if ext.lower() == '.gz' or ext.lower() == '.nii':
                    sitkimage = sitk.ReadImage(filename)
                    img = sitk.GetArrayFromImage(sitkimage)
                    print("**image shape:", img.shape, "**")
                    self.img_shape = img.shape
                    self.raw_img = img
                    self.label.raw_image = img
                    self.scrollbar.setRange(0, self.raw_img.shape[0] - 1)
                    self.pixmap = self.show3dimg()

                    nii_img = nib.load(filename)
                    self.label.spacing = nii_img.affine
                    self.label.setAlignment(Qt.AlignCenter)
                    self.label.update_image(self.pixmap, self.raw_img.shape[0] - 1 - self.scrollbar.value())


    def next_case(self):
        if self.label.measurement != []:
            self.measurement[str(self.id)] = self.label.measurement
            self.label.measurement = []
            self.label.box_start_end = []

        # Incremental save before switching case
        self.save_incremental()

        self.current_patient += 1
        if self.current_patient >= len(self.patient_list):
            print("**no more patient**")
        else:
            self.scrollbar.setValue(0)
            self.counter.setText(f"{self.current_patient + 1}")
            self.imgpath = self.patient_list[self.current_patient]
            filename = self.imgpath
            print("**image name:", self.imgpath, "**")
            self.id = os.path.basename(self.imgpath).replace("_vol.nii.gz", "")
            self.label.set_patient_id(self.id)

            # find the boxes according to id
            self.boxes = self.nodules_id_dict[self.id]
            self.label.boxes = self.boxes

            _, ext = os.path.splitext(filename)
            if filename:
                if ext.lower() == '.gz' or ext.lower() == '.nii':
                    sitkimage = sitk.ReadImage(filename)
                    img = sitk.GetArrayFromImage(sitkimage)
                    print("**image shape:", img.shape, "**")
                    self.img_shape = img.shape
                    self.raw_img = img
                    self.label.raw_image = img
                    self.scrollbar.setRange(0, self.raw_img.shape[0] - 1)
                    self.pixmap = self.show3dimg()

                    nii_img = nib.load(filename)
                    self.label.spacing = nii_img.affine
                    self.label.setAlignment(Qt.AlignCenter)
                    self.label.update_image(self.pixmap, self.raw_img.shape[0] - 1 - self.scrollbar.value())

    def next_miss(self):
        if self.label.raw_gaze_all is None:
            print("**no gaze yet**")
            QMessageBox.information(self, "Information", f"There is no raw gaze data in this case", QMessageBox.Ok)
            return
        missing_slices = self.missed_slices[self.id]
        if missing_slices == []:
            print("**no missing slices**")
            QMessageBox.information(self, "Information", f"There is no missing nodules in this case", QMessageBox.Ok)
            return
        
        current_slice = self.raw_img.shape[0] - 1 - self.scrollbar.value()
        next_bigger = None
        for x in missing_slices:
            if x < current_slice and (next_bigger is None or x > next_bigger):
                next_bigger = x
                
        print("current slice: {}, missing slices: {}, next bigger slice: {}".format(current_slice, missing_slices, next_bigger))

        if next_bigger is None:
            print("**no next missing slice**")
            QMessageBox.information(self, "Information", f"There is no next missing slice in this case", QMessageBox.Ok)
            return

        self.scrollbar.setValue(self.raw_img.shape[0] - 1 - next_bigger)
        self.pixmap = self.show3dimg()
        self.label.ellipses = []
        self.label.mean_density = []
        self.label.line1 = []; self.label.line2 = []; self.label.line2_slope = []
        self.label.draw_first_line = True; self.label.draw_second_line = False
        self.label.update_image(self.pixmap, self.raw_img.shape[0] - 1 - self.scrollbar.value())

    def last_miss(self):
        if self.label.raw_gaze_all is None:
            print("**no gaze yet**")
            QMessageBox.information(self, "Information", f"There is no raw gaze data in this case", QMessageBox.Ok)
            return        
        missing_slices = self.missed_slices[self.id]
        if missing_slices == []:
            print("**no missing slices**")
            QMessageBox.information(self, "Information", f"There is no missing nodules in this case", QMessageBox.Ok)
            return
        
        current_slice = self.raw_img.shape[0] - 1 - self.scrollbar.value()
        last_smaller = None
        for x in missing_slices:
            if x > current_slice and (last_smaller is None or x < last_smaller):
                last_smaller = x
                
        print("current slice: {}, missing slices: {}, last smaller slice: {}".format(current_slice, missing_slices, last_smaller))

        if last_smaller is None:
            print("**no last missing slice**")
            QMessageBox.information(self, "Information", f"There is no last missing slice in this case", QMessageBox.Ok)
            return

        self.scrollbar.setValue(self.raw_img.shape[0] - 1 - last_smaller)
        self.pixmap = self.show3dimg()
        self.label.ellipses = []
        self.label.mean_density = []
        self.label.line1 = []; self.label.line2 = []; self.label.line2_slope = []
        self.label.draw_first_line = True; self.label.draw_second_line = False
        self.label.update_image(self.pixmap, self.raw_img.shape[0] - 1 - self.scrollbar.value())

    def show_box(self):
        if self.pixmap == None:
            return
        self.showbox_flag = not self.showbox_flag
        if self.showbox_flag:
            self.showbox_button.setStyleSheet("background-color: lightgreen")
            print("*******Show Box*******")
            self.label.show_box_flag = True
            self.label.boxes = self.boxes
        else:
            self.showbox_button.setStyleSheet("background-color: white; color: black;")
            print("*******Hide Box*******")
            self.label.show_box_flag = False
            self.label.boxes = []
            self.label.reset_image(self.label.image_scale)
        
        self.label.update()

    def show_looked(self):
        if self.label.raw_gaze_all is None:
            print("**no gaze yet**")
            QMessageBox.information(self, "Information", f"There is no raw gaze data in this case", QMessageBox.Ok)
            return
        if self.pixmap == None:
            return
        self.label.showlooked_flag = not self.label.showlooked_flag
        if self.endRecord_flag and self.label.showlooked_flag:
            self.showlooked_button.setStyleSheet("background-color: lightgreen")
            print("*******Show looked*******")
            
        elif self.endRecord_flag and not self.label.showlooked_flag:
            self.showlooked_button.setStyleSheet("background-color: white; color: black;")
            print("*******Hide looked*******")
            self.label.reset_image(self.label.image_scale)
        self.label.update()

    def compute_iou_3d(self, box1, box2):
        z0_1, x0_1, y0_1, z1_1, x1_1, y1_1 = box1
        z0_2, x0_2, y0_2, z1_2, x1_2, y1_2 = box2

        # Compute intersection box
        iz0 = max(z0_1, z0_2)
        ix0 = max(x0_1, x0_2)
        iy0 = max(y0_1, y0_2)
        iz1 = min(z1_1, z1_2)
        ix1 = min(x1_1, x1_2)
        iy1 = min(y1_1, y1_2)

        inter_depth = max(0, iz1 - iz0 + 1)
        inter_width  = max(0, ix1 - ix0 + 1)
        inter_height = max(0, iy1 - iy0 + 1)

        inter_volume = inter_depth * inter_width * inter_height

        # Compute volumes
        vol1 = (z1_1 - z0_1 + 1) * (x1_1 - x0_1 + 1) * (y1_1 - y0_1 + 1)
        vol2 = (z1_2 - z0_2 + 1) * (x1_2 - x0_2 + 1) * (y1_2 - y0_2 + 1)

        union = vol1 + vol2 - inter_volume

        if union == 0:
            return 0.0
        else:
            return inter_volume / union

    def get_missed_nodule_slices_from_raw_gaze(self):
        """
        Determine missed slices based on raw gaze data, without heatmap.
        A box is considered 'seen' if any gaze falls within the box (+tolerance) on any of its slices.
        """
        missed_slices_per_patient = {}

        grouped = self.label.raw_gaze_all
        boxes_dict = self.nodules_id_dict
        tolerance = 10
        iou_threshold = 0.01
        
        # Group by patient ID
        # grouped = raw_gaze_df.groupby("Patient ID")
        self.label.box_missed_state = {}

        for pid, boxes in boxes_dict.items():
            if pid not in grouped.groups:
                continue

            df = grouped.get_group(pid)
            gaze_by_slice = df.groupby("Slice Number")

            missed_slices = []
            box_missed_per_patient = {}
            annotated_boxes = self.json_data.get(pid, [])

            for box in boxes:
                z0, x0, y0, z1, x1, y1 = map(int, box)

                x0e = x0 - tolerance
                x1e = x1 + tolerance
                y0e = y0 - tolerance
                y1e = y1 + tolerance

                box_looked = False

                for z in range(z0, z1 + 1):
                    if z not in gaze_by_slice.groups:
                        continue
                    gaze_points = gaze_by_slice.get_group(z)[["Gaze_Location_X", "Gaze_Location_Y"]].values

                    for gx, gy in gaze_points:
                        if x0e <= gx <= x1e and y0e <= gy <= y1e:
                            box_looked = True
                            break
                    if box_looked:
                        break

                # Check human annotation overlap
                box_annotated = False
                for anno_box in annotated_boxes:
                    iou = self.compute_iou_3d(box, anno_box)
                    if iou >= iou_threshold:
                        box_annotated = True
                        break

                if not box_annotated:
                    missed_slices.append(z1)
                    box_missed_per_patient[tuple(box)] = True
                else:
                    box_missed_per_patient[tuple(box)] = False

            if missed_slices:
                missed_slices_per_patient[pid] = sorted(missed_slices)
            else:
                missed_slices_per_patient[pid] = []
            
            self.label.box_missed_state[pid] = box_missed_per_patient

        print(self.label.box_missed_state)
        save_json = {}
        for pid in self.label.box_missed_state.keys():
            save_box = []
            for box, missed in self.label.box_missed_state[pid].items():
                if missed:
                    save_box.append(list(box))
            save_json[pid] = save_box
        print(save_json)
        # save as json
        with open("missed.json", "w") as f:
            json.dump(save_json, f, indent=2)

        return missed_slices_per_patient



    def _analysis(self):

        folder = QFileDialog.getExistingDirectory(self, 'Select Folder', '')
        if not folder:
            return
        self.csv_record = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(('csv'))]
        self.json_record = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(('json'))]

        data = pd.DataFrame()
        for csv_file in self.csv_record:
            file_path = os.path.join(folder, csv_file)
            _data = pd.read_csv(file_path)
            data = pd.concat([data, _data], ignore_index=True)

        self.json_data = {}
        for json_file in self.json_record:
            file_path = os.path.join(folder, json_file)
            with open(file_path, 'r') as f:
                _data = json.load(f)
                self.json_data.update(_data)

        data["Patient ID"] = data["File Name"].apply(lambda x: re.search(r'LIDC-IDRI-\d+', x).group())
        eyegaze_grouped_by_id = data.groupby("Patient ID")
        self.label.raw_gaze_all = eyegaze_grouped_by_id
        self.label.json_data = self.json_data

        # get Set ID
        first_path = data["File Name"].iloc[0]
        set_id = re.search(r'Set\d+', first_path).group()
        print(f"Set ID: {set_id}")

        # get missed slices
        self.missed_slices = self.get_missed_nodule_slices_from_raw_gaze()
        print("Missed slices: ", self.missed_slices)

        print("*******Finish loading of the Eye Gaze*********")
        self.analysis_button.setStyleSheet("background-color: lightgreen")

    def show3dimg(self):
        medimg = self.raw_img[self.raw_img.shape[0] - 1 - self.scrollbar.value(),...]
        medimg = truncate_hu(medimg, self.mode)
        medimg = normalazation(medimg)
        medimg = (np.dstack([medimg, medimg, medimg]))
        self.image = medimg
        bytesPerline = 3 * medimg.shape[1]
        pixmap = QPixmap.fromImage((QImage(medimg.data, medimg.shape[1], medimg.shape[0], bytesPerline, QImage.Format_RGB888)))

        return pixmap
    
    def change_contrast(self, mode):
        if self.pixmap == None:
            return
        print("change contrast to {}".format(mode))
        self.mode = mode
        self.pixmap = self.show3dimg()
        self.label.update_image(self.pixmap, self.raw_img.shape[0] - 1 - self.scrollbar.value())

    def doCalibrate(self):

        label_top_left = self.label.mapToGlobal(QPoint(0,0))
        self.label_top_left_pos = [label_top_left.x(), label_top_left.y()]
        print("**label size:", self.label.size(), "**")
        print("**label top left corner coordinate in screen space:", self.label_top_left_pos, "**")


        input_dialog = CustomInputDialog(self)
        input_dialog.setLabelText("Type your first name:")
        if input_dialog.exec_() == QInputDialog.Accepted:
            text = input_dialog.textValue()
            print(f"**User name: {text}", "**")
        else:
            return

        edf_fname = text.rstrip().split('.')[0]

        # Only create new result_dir and csvname on first calibration
        if self.result_dir is None:
            time_str = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
            self.result_dir = os.path.join("../Collections" ,edf_fname + time_str)
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
            self.csvname = edf_fname + time_str + ".csv"
        
        debug = False
        if debug:
            tracker = pylink.EyeLink(None)
        else:
            try:
                tracker = pylink.EyeLink("100.1.1.1")
            except RuntimeError as error:
                print('ERROR:', error)
                pygame.quit()
                sys.exit()

        self.eyetracker = Calibrate(debug=debug, edf_fname=edf_fname, el_tracker=tracker, session_folder=self.result_dir)
        self.eyetracker.setOfflineMode()

        try:
            self.eyetracker.startRecording(1,1,1,1)
        except RuntimeError as error:
            print("ERROR:", error)

        self.eye_used = self.eyetracker.eyeAvailable()
        if self.eye_used == 1:
            self.eyetracker.sendMessage("EYE_USED 1 RIGHT")
        elif self.eye_used == 0 or self.eye_used == 2:
            self.eyetracker.sendMessage("EYE_USED 0 LEFT")
            self.eye_used = 0
        else:
            print("ERROR: Could not get eye information!")
            return pylink.TRIAL_ERROR

        print("----------------------calibration done----------------------")

        # dt = self.eyetracker.getNewestSample()
        # if dt is None:
        #     gaze_x, gaze_y = (-327678, -32768)
        # else:
        #     if self.eye_used == 1 and dt.isRightSample():
        #         gaze_x, gaze_y = dt.getRightEye().getGaze()
        #     elif self.eye_used == 0 and dt.isLeftSample():
        #         gaze_x, gaze_y = dt.getLeftEye().getGaze()


    def drawFixation(self, fix, colour):
        """ Send a command to draw a cross or filled box on the tracker screen

        Draw a cross or a filled box representing a fixation or 'fixation update'
        event on the tracker display.

        Parameters:
        fix: a two or four-element tuple to be interpreted as follows:
            if 'fix' contains two elements a cross is requested.
                Interpret as fix[0]=x, fix[1]=y
            if  'fix' contains four elements a filled box is requested.
                Interpret as fix[0]=left, fix[1]=top, fix[2]=width, fix[3]=height
        colour: numerical value from 0 to 15 to represent the colour of the target
        """

        # get the currently active tracker object (connection)
        el_tracker = pylink.getEYELINK()

        err = "drawFixation expects a 2- or 4-element tuple\n"

        if len(fix) == 2:
            el_tracker.drawCross(fix[0], fix[1], colour)

        elif len(fix) == 4:
            el_tracker.drawFilledBox(fix[0], fix[1], fix[2], fix[3], colour)
        else:
            print(err)

    def record(self):

        if self.imgpath is None:
            return
        if not hasattr(self, 'label_top_left_pos'):
            return
        if self.eyetracker:
            gaze_x = None; gaze_y = None; duration = None
            # ltype = self.eyetracker.getNextData()

            # if ltype == pylink.ENDFIX:
            #     ldata = self.eyetracker.getFloatData()
            #     if ldata.getEye() == self.eye_used:
            #         gaze_x, gaze_y = ldata.getAverageGaze()
            #         # self.drawFixation((gaze_x, gaze_y), 15)
            #         print("current gaze location:{}, {}".format(gaze_x, gaze_y))
            #         start_time = ldata.getStartTime()
            #         end_time = ldata.getEndTime()
            #         duration = end_time-start_time
            #         print("start time {} end time {} duration {}".format(start_time, end_time, duration))

            dt = self.eyetracker.getNewestSample()
            if dt is None:
                return
            else:
                # if self.eye_used == 1 and dt.isRightSample():
                gaze_x1, gaze_y1= dt.getRightEye().getGaze()
                # elif self.eye_used == 0 and dt.isLeftSample():
                gaze_x2, gaze_y2 = dt.getLeftEye().getGaze()
                gaze_x = (gaze_x1 + gaze_x2) /2
                gaze_y = (gaze_y1 + gaze_y2) /2



            duration = 1
            # gaze_x = random.randint(1800, 2000)
            # gaze_y = random.randint(900, 1100)
            if gaze_x and gaze_y and duration:
                label_coord = [gaze_x - self.label_top_left_pos[0],
                                    gaze_y - self.label_top_left_pos[1]]
                self.label.set_point(QPoint(int(label_coord[0]), int(label_coord[1])))

                # calulate the data to record including current time, current gaze location on image space, and current slice number
                current_image_scale = self.label.image_scale
                current_image_corner_loc = self.label.image_position
                current_time = QTime.currentTime().toString("hh:mm:ss")
                current_location = [
                    (label_coord[0] - current_image_corner_loc[0]) / current_image_scale,
                    (label_coord[1] - current_image_corner_loc[1]) / current_image_scale
                ]
                current_slice = self.label.slice_num
                current_file = self.imgpath

                if current_location[0] < 0 or current_location[0] > self.raw_img.shape[2] or current_location[1] < 0 or current_location[1] > self.raw_img.shape[1]:
                    return
                self.data.append([current_time, 
                                format(current_location[0], f".{4}f"), 
                                format(current_location[1], f".{4}f"), 
                                current_slice,
                                duration, 
                                current_file])

    def save_incremental(self):
        """Append current gaze data to CSV and save measurement JSON incrementally."""
        if self.result_dir is None:
            print("**WARNING: result_dir not set, skipping save. Please run calibration first.**")
            return
        if self.data:
            file_path = os.path.join(self.result_dir, "Trial{}.csv".format(self.trial))
            df = pd.DataFrame(self.data, columns=["Time", "Gaze_Location_X", "Gaze_Location_Y", "Slice Number", "Duration", "File Name"])
            write_header = not os.path.exists(file_path)
            df.to_csv(file_path, mode='a', index=False, header=write_header)
            print(f"----------------------append {len(self.data)} rows to csv----------------------")
            self.data = []

        if self.measurement:
            file_path = os.path.join(self.result_dir, "Trial{}.json".format(self.trial))
            with open(file_path, "w") as file:
                json.dump(self.measurement, file)
            print("----------------------save measurement json----------------------")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        uifile = sys.argv[1]
    else:
        uifile = 'ui-nu.ui'
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow(uifile)
    window.show()
    sys.exit(app.exec_())
