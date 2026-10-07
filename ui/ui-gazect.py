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
from scipy import ndimage
from skimage import measure
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gaze_classifier import GazeClassifier, create_fixation_gazemap


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
        self.non_coverage_contours = {}
        self.missed_nodules = []
        self.show_nonseen_flag = False

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
                    x = min(x0, x1)*self.image_scale + self.image_position[0]
                    y = min(y0, y1)*self.image_scale + self.image_position[1]
                    w = abs(x1 - x0)*self.image_scale
                    h = abs(y1 - y0)*self.image_scale
                    painter.setPen(QColor("red"))
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

        if self.show_nonseen_flag:
            # Draw non-coverage contours (red)
            if self.slice_num in self.non_coverage_contours:
                painter.setPen(QPen(QColor("yellow"), 2))
                painter.setBrush(Qt.transparent)
                for contour in self.non_coverage_contours[self.slice_num]:
                    points = []
                    for row, col in contour:
                        px = col * self.image_scale + self.image_position[0]
                        py = row * self.image_scale + self.image_position[1]
                        points.append(QPoint(int(px), int(py)))
                    if len(points) > 1:
                        for i in range(len(points) - 1):
                            painter.drawLine(points[i], points[i + 1])

            # Draw missed nodules not annotated by doctor (cyan)
            painter.setPen(QPen(QColor("yellow"), 2))
            for box in self.missed_nodules:
                z0, x0, y0, z1, x1, y1 = map(int, box)
                if z0 <= self.slice_num <= z1:
                    x = (min(x0, x1)-5) * self.image_scale + self.image_position[0]
                    y = (min(y0, y1)-5) * self.image_scale + self.image_position[1]
                    w = (abs(x1 - x0)+10) * self.image_scale
                    h = (abs(y1 - y0)+10) * self.image_scale
                    painter.drawRect(int(x), int(y), int(w), int(h))

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
        from PyQt5.QtGui import QFont

        font = QFont()
        font.setPointSize(20)
        self.setFont(font)

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
        self.showlooked_button.clicked.connect(self.toggle_missed)
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
        self.measurement_firstread = {}
        self.measurement_secondlook = {}
        self.current_patient = 0
        self.trial = 1
        self.boxes = []
        self.missed_slices = {}
        self.showbox_flag = False
        self.endRecord_flag = True
        self.phase = "first_read"
        self.lung_mask = None
        self.lung_mask_dir = None
        self.result_dir = None
        self.patient_list = []
        self.nodules_id_dict = {}
        self.id = None
        self.json_data = None
        sys.excepthook = self.exceptionHandler

        self.timer = QTimer()
        self.timer.timeout.connect(self.record)

    def show_message(self, title, message):
        """Show a styled message box with larger font."""
        from PyQt5.QtGui import QFont

        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Information)

        # Set font directly
        font = QFont()
        font.setPointSize(20)
        msg.setFont(font)

        # Style buttons
        for button in msg.buttons():
            button.setFont(font)
            button.setMinimumSize(100, 40)

        msg.exec_()

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
        self.phase = "first_read"
        if self.data != []:
            self.counter.setStyleSheet("background-color: green; color: white;")
        self.endRecord_flag = False

    def stopthread(self):
        # Prevent clicking during wrong phase
        if self.phase != "first_read":
            self.show_message("Warning", "Already in Second Look phase.\nUse 'Next Case' to proceed.")
            return

        print("----------------------end first read----------------------")
        self.timer.stop()
        self.counter.setStyleSheet("background-color: black; color: white;")
        self.counter.repaint()

        # Save first read measurement for this case
        self.measurement_firstread[str(self.id)] = self.label.measurement
        self.label.measurement = []
        self.label.drawn_boxes = []
        self.label.box_start_end = []

        # Analyze current case BEFORE save (save clears self.data)
        self.analyze_current_case()

        # Incremental save (gaze CSV + firstread JSON)
        self.save_incremental()

        # Check if there are missed nodules
        n_missed = len(self.label.missed_nodules)
        n_missed_slices = len(self.missed_slices.get(self.id, []))

        if n_missed == 0 and n_missed_slices == 0:
            # No missed nodules - no need for second look
            self.show_message("Analysis Complete",
                "First read analysis done.\n\n"
                "There is no potential missed nodules!\n"
                "No second look needed.\n\n"
                "Click OK to continue to next case.")
            # Keep counter black, don't enter second look
            self.counter.setStyleSheet("background-color: black; color: white;")
            self.phase = "second_look"  # Mark as done, allow next case
            print("**No missed nodules, skipping second look**")
        else:
            # Has missed nodules - enter second look
            self.label.show_nonseen_flag = True
            self.showlooked_button.setStyleSheet("background-color: lightgreen")
            self.label.update()

            self.show_message("Analysis Complete",
                f"First read analysis done.\n"
                f"{n_missed} missed (unannotated) nodule(s).\n"
                f"{n_missed_slices} slice(s) to review.\n\n"
                f"We think these areas might contain missed nodules, please review them again.\n\n"
                f"Click OK to start second look.")

            # Jump to first missed slice (largest slice number, which is at the top)
            missing_slices = self.missed_slices.get(self.id, [])
            if missing_slices:
                first_missed_slice = max(missing_slices)  # Get the largest slice (top of scan)
                self.scrollbar.setValue(self.raw_img.shape[0] - 1 - first_missed_slice)
                self.pixmap = self.show3dimg()
                self.label.update_image(self.pixmap, self.raw_img.shape[0] - 1 - self.scrollbar.value())
                print(f"**Jumped to first missed slice: {first_missed_slice}**")

            # Switch to second look phase
            self.phase = "second_look"
            self.counter.setStyleSheet("background-color: orange; color: black;")
            self.timer.start(10)

    def pause_operation(self):
        print("----------------------pause eye tracking----------------------")
        self.timer.stop()

        # Save current phase measurement
        if self.phase == "second_look" and self.label.measurement:
            self.measurement_secondlook[str(self.id)] = self.label.measurement
            self.label.measurement = []
            self.label.drawn_boxes = []
            self.label.box_start_end = []
        elif self.phase == "first_read" and self.label.measurement:
            self.measurement_firstread[str(self.id)] = self.label.measurement
            self.label.measurement = []
            self.label.drawn_boxes = []
            self.label.box_start_end = []

        self.save_incremental()

        # Close eyetracker
        if self.eyetracker is not None:
            self.eyetracker.close()
            pylink.closeGraphics()
            self.eyetracker = None

        self.counter.setStyleSheet("background-color: black; color: white;")
        self.endRecord_flag = True

    def close_operation(self):
        print("----------------------close eye tracking----------------------")
        self.timer.stop()

        # Save current phase measurement before closing
        if self.phase == "second_look" and self.label.measurement:
            self.measurement_secondlook[str(self.id)] = self.label.measurement
            self.label.measurement = []
            self.label.drawn_boxes = []
            self.label.box_start_end = []
        elif self.phase == "first_read" and self.label.measurement:
            self.measurement_firstread[str(self.id)] = self.label.measurement
            self.label.measurement = []
            self.label.drawn_boxes = []
            self.label.box_start_end = []

        if self.data != []:
            self.save_incremental()
        if self.eyetracker is not None:
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

        # load lung mask
        self.lung_mask_dir = folder.replace("vol", "lung_mask")
        self._load_lung_mask()

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

    def _load_lung_mask(self):
        """Load lung mask for the current patient."""
        self.lung_mask = None
        if not os.path.exists(self.lung_mask_dir):
            print(f"**WARNING: lung_mask dir not found: {self.lung_mask_dir}**")
            return
        mask_file = os.path.join(self.lung_mask_dir, f"{self.id}_lungmask.nii.gz")
        if not os.path.exists(mask_file):
            # try without _lung_mask suffix
            mask_file = os.path.join(self.lung_mask_dir, f"{self.id}_mask.nii.gz")
        if os.path.exists(mask_file):
            mask_img = sitk.ReadImage(mask_file)
            self.lung_mask = sitk.GetArrayFromImage(mask_img).squeeze() > 0
            print(f"**lung mask loaded: {mask_file}, shape: {self.lung_mask.shape}**")
        else:
            print(f"**WARNING: lung mask not found for {self.id}**")

    def last_case(self):
        if self.current_patient == 0:
            print("**this is the first patient**")
            return

        # Block if in second look phase
        if self.phase == "second_look":
            self.show_message("Warning", "Please complete current case first.\nUse 'Next Case' to proceed.")
            return

        # Confirm if in first read phase (analysis not done)
        if self.phase == "first_read":
            reply = QMessageBox.question(self, "Confirm",
                "First read analysis not done!\n"
                "Data will be saved but analysis will be skipped.\n\n"
                "Continue to previous case?",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
            # Save current data before going back
            self.timer.stop()
            self.counter.setStyleSheet("background-color: black; color: white;")
            if self.label.measurement:
                self.measurement_firstread[str(self.id)] = self.label.measurement
                self.label.measurement = []
                self.label.drawn_boxes = []
                self.label.box_start_end = []
            self.save_incremental()

        # Reset state and switch to previous case
        self.label.box_start_end = []
        self.showbox_flag = False
        self.label.show_box_flag = False
        self.label.box_missed_state = None
        self.label.show_nonseen_flag = False
        self.label.non_coverage_contours = {}
        self.label.missed_nodules = []
        self.showbox_button.setStyleSheet("background-color: white; color: black;")
        self.showlooked_button.setStyleSheet("background-color: white; color: black;")
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

        self._load_lung_mask()
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

        # Auto start first read for previous case
        self.phase = "first_read"
        self.counter.setStyleSheet("background-color: green; color: white;")
        self.timer.start(10)

    def next_case(self):
        # Prevent skipping first read analysis
        if self.phase == "first_read":
            self.show_message("Warning", "Please click 'End First Read' to complete analysis first.")
            return

        # Stop recording
        self.timer.stop()
        self.counter.setStyleSheet("background-color: black; color: white;")


        # Save second look measurement for this case
        if self.label.measurement:
            self.measurement_secondlook[str(self.id)] = self.label.measurement
        self.label.measurement = []
        self.label.drawn_boxes = []
        self.label.box_start_end = []

        # Incremental save (gaze CSV + both JSONs)
        self.save_incremental()

        # Clear visual state
        self.showbox_flag = False
        self.label.show_box_flag = False
        self.label.box_missed_state = None
        self.label.show_nonseen_flag = False
        self.label.non_coverage_contours = {}
        self.label.missed_nodules = []
        self.showbox_button.setStyleSheet("background-color: white; color: black;")
        self.showlooked_button.setStyleSheet("background-color: white; color: black;")
        self.label.reset_image(self.label.image_scale)

        self.current_patient += 1
        if self.current_patient >= len(self.patient_list):
            print("**no more patient**")
            self.show_message("Information", "All cases completed!")
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

            # load lung mask for new case
            self._load_lung_mask()

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

            # Auto start first read for next case
            self.phase = "first_read"
            self.counter.setStyleSheet("background-color: green; color: white;")
            self.timer.start(10)

    def next_miss(self):
        if self.id not in self.missed_slices:
            print("**no analysis yet**")
            self.show_message("Information", "No analysis has been run for this case yet.")
            return
        missing_slices = self.missed_slices[self.id]
        if missing_slices == []:
            print("**no missing slices**")
            self.show_message("Information", "There is no missing nodules in this case")
            return

        current_slice = self.raw_img.shape[0] - 1 - self.scrollbar.value()
        next_bigger = None
        for x in missing_slices:
            if x < current_slice and (next_bigger is None or x > next_bigger):
                next_bigger = x

        print("current slice: {}, missing slices: {}, next bigger slice: {}".format(current_slice, missing_slices, next_bigger))

        if next_bigger is None:
            print("**no next missing slice**")
            self.show_message("Information", "There is no next missing slice in this case")
            return

        self.scrollbar.setValue(self.raw_img.shape[0] - 1 - next_bigger)
        self.pixmap = self.show3dimg()
        self.label.ellipses = []
        self.label.mean_density = []
        self.label.line1 = []; self.label.line2 = []; self.label.line2_slope = []
        self.label.draw_first_line = True; self.label.draw_second_line = False
        self.label.update_image(self.pixmap, self.raw_img.shape[0] - 1 - self.scrollbar.value())

    def last_miss(self):
        if self.id not in self.missed_slices:
            print("**no analysis yet**")
            self.show_message("Information", "No analysis has been run for this case yet.")
            return
        missing_slices = self.missed_slices[self.id]
        if missing_slices == []:
            print("**no missing slices**")
            self.show_message("Information", "There is no missing nodules in this case")
            return

        current_slice = self.raw_img.shape[0] - 1 - self.scrollbar.value()
        last_smaller = None
        for x in missing_slices:
            if x > current_slice and (last_smaller is None or x < last_smaller):
                last_smaller = x

        print("current slice: {}, missing slices: {}, last smaller slice: {}".format(current_slice, missing_slices, last_smaller))

        if last_smaller is None:
            print("**no last missing slice**")
            self.show_message("Information", "There is no last missing slice in this case")
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

    def toggle_missed(self):
        """Toggle display of missed nodules and non-coverage contours during second look."""
        if self.pixmap is None:
            return

        # Check if we have any missed data to show
        if not self.label.missed_nodules and not self.label.non_coverage_contours:
            print("**no missed nodules data yet**")
            self.show_message("Information",
                "No missed nodules data available.\nRun first read analysis first.")
            return

        # Toggle the flag
        self.label.show_nonseen_flag = not self.label.show_nonseen_flag

        if self.label.show_nonseen_flag:
            self.showlooked_button.setStyleSheet("background-color: lightgreen")
            print("*******Show missed regions*******")
        else:
            self.showlooked_button.setStyleSheet("background-color: white; color: black;")
            print("*******Hide missed regions*******")

        self.label.update()

    def compute_iou_3d(self, box1, box2, method='iomin'):
        z0_1, x0_1, y0_1, z1_1, x1_1, y1_1 = box1
        z0_2, x0_2, y0_2, z1_2, x1_2, y1_2 = box2

        z0_1 -=5
        z1_1 +=5
        z0_2 -=5
        z1_2 +=5

        # Compute intersection box
        iz0 = max(z0_1, z0_2)
        iy0 = max(y0_1, y0_2)
        ix0 = max(x0_1, x0_2)
        iz1 = min(z1_1, z1_2)
        iy1 = min(y1_1, y1_2)
        ix1 = min(x1_1, x1_2)

        inter_depth = max(0, iz1 - iz0 + 1)
        inter_height = max(0, iy1 - iy0 + 1)
        inter_width  = max(0, ix1 - ix0 + 1)

        inter_volume = inter_depth * inter_height * inter_width

        # Compute volumes
        vol1 = (z1_1 - z0_1 + 1) * (y1_1 - y0_1 + 1) * (x1_1 - x0_1 + 1)
        vol2 = (z1_2 - z0_2 + 1) * (y1_2 - y0_2 + 1) * (x1_2 - x0_2 + 1)
        
        if method == 'iomin':
            denom = min(vol1, vol2)
        else:  # standard iou
            denom = vol1 + vol2 - inter_volume
    
        return inter_volume / denom if denom > 0 else 0



    def analyze_current_case(self):
        """Analyze gaze data using 3D attention map + non-coverage pipeline."""
        pid = self.id
        boxes = self.boxes
        iou_threshold = 0.01

        # Step 1: Build gaze DataFrame from in-memory data
        first_read_data = [row for row in self.data if row[6] == "first_read"]
        if not first_read_data:
            print("**no first_read gaze data for current case**")
            self.missed_slices[pid] = []
            return

        df = pd.DataFrame(first_read_data, columns=["Time", "Gaze_Location_X", "Gaze_Location_Y", "Slice Number", "Duration", "File Name", "Phase"])
        df["Gaze_Location_X"] = df["Gaze_Location_X"].astype(float)
        df["Gaze_Location_Y"] = df["Gaze_Location_Y"].astype(float)

        # Step 2: Fixation classification + gazemap
        classifier = GazeClassifier(velocity_threshold=400.0, min_fixation_duration=6, sampling_rate=None)
        classified_data = classifier.classify_events(df)
        fixations = classifier.extract_fixations(classified_data)
        img_shape = self.raw_img.shape
        print(f"**fixations: {len(fixations)}, image shape: {img_shape}**")

        if len(fixations) > 0:
            fixation_gazemap = create_fixation_gazemap(fixations, img_shape)
        else:
            # Fallback to raw gazemap
            print("**no fixations detected, using raw gaze points for gazemap**")
            fixation_gazemap = np.zeros(img_shape, dtype=np.float32)
            for _, row in df.iterrows():
                s = int(row["Slice Number"])
                x = int(float(row["Gaze_Location_X"]))
                y = int(float(row["Gaze_Location_Y"]))
                if 0 <= s < img_shape[0] and 0 <= y < img_shape[1] and 0 <= x < img_shape[2]:
                    fixation_gazemap[s, y, x] += 1

        # Step 3: 3D Gaussian smoothing
        attention_3d = gaussian_filter(fixation_gazemap, sigma=(1, 20, 20))

        # Step 4: Threshold with lung mask
        if self.lung_mask is not None and self.lung_mask.shape == img_shape:
            lung_attention = attention_3d[self.lung_mask]
            if lung_attention.max() > 0:
                threshold = lung_attention.max() * 0.1
            else:
                threshold = 0
            coverage_map = attention_3d > threshold
            non_coverage = self.lung_mask & (~coverage_map)
        else:
            # No lung mask: use global threshold
            print("**WARNING: lung mask not available, using global threshold**")
            max_att = attention_3d.max()
            threshold = max_att * 0.1 if max_att > 0 else 0
            coverage_map = attention_3d > threshold
            non_coverage = ~coverage_map
        print(f"**non-coverage voxels: {non_coverage.sum()}**")

        # Step 5: Morphological cleanup
        struct_3d = ndimage.generate_binary_structure(3, 1)
        non_coverage = ndimage.binary_opening(non_coverage, structure=struct_3d, iterations=10)
        non_coverage = ndimage.binary_closing(non_coverage, structure=struct_3d, iterations=1)

        struct_2d = ndimage.generate_binary_structure(2, 1)
        min_area_2d = 2000
        non_coverage_cleaned = np.zeros_like(non_coverage, dtype=bool)
        for z in range(non_coverage.shape[0]):
            sl = non_coverage[z]
            sl = ndimage.binary_opening(sl, structure=struct_2d, iterations=10)
            sl = ndimage.binary_fill_holes(sl)
            labeled_2d, num_2d = ndimage.label(sl)
            areas = ndimage.sum(sl, labeled_2d, range(num_2d + 1))
            for rid in range(1, num_2d + 1):
                if areas[rid] >= min_area_2d:
                    non_coverage_cleaned[z][labeled_2d == rid] = True
        print(f"**non-coverage after cleanup: {non_coverage_cleaned.sum()}**")

        # Step 7: Find missed nodules (not annotated by doctor)
        annotated_boxes = self.measurement_firstread.get(pid, [])
        missed_nodules = []
        for box in boxes:
            box_annotated = False
            for anno_box in annotated_boxes:
                iou = self.compute_iou_3d(box, anno_box)
                if iou >= iou_threshold:
                    box_annotated = True
                    break
            if not box_annotated:
                missed_nodules.append(box)

        # Step 8: Extract contours only for non-coverage regions that contain AI nodules
        # First, create a mask of only those non-coverage regions with AI nodules
        non_coverage_with_nodules = np.zeros_like(non_coverage_cleaned, dtype=bool)
        for box in missed_nodules:
            z0, x0, y0, z1, x1, y1 = map(int, box)
            for z in range(max(0, z0), min(img_shape[0], z1 + 1)):
                # Find connected component that contains this nodule
                labeled_slice, num_labels = ndimage.label(non_coverage_cleaned[z])
                for yy in range(max(0, y0), min(img_shape[1], y1 + 1)):
                    for xx in range(max(0, x0), min(img_shape[2], x1 + 1)):
                        if labeled_slice[yy, xx] > 0:
                            # Mark the entire connected component
                            non_coverage_with_nodules[z][labeled_slice == labeled_slice[yy, xx]] = True

        # Smooth for contour extraction
        non_coverage_smoothed = gaussian_filter(non_coverage_with_nodules.astype(float), sigma=(0, 3, 3))
        contours_per_slice = {}
        for z in range(img_shape[0]):
            if non_coverage_with_nodules[z].sum() == 0:
                continue
            contours = measure.find_contours(non_coverage_smoothed[z], 0.5)
            if contours:
                contours_per_slice[z] = contours

        # Step 9: Store results
        self.label.non_coverage_contours = contours_per_slice
        self.label.missed_nodules = missed_nodules

        # Collect missed slices for navigation (all slices spanned by non-seen nodules + missed nodules)
        missed_slices = set()
        for box in missed_nodules:
            z0, _, _, z1, _, _ = map(int, box)
            missed_slices.update(range(z0, z1 + 1))
        self.missed_slices[pid] = sorted(missed_slices)

        print(f"**Analysis: {len(missed_nodules)} missed, {len(contours_per_slice)} slices with contours**")

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
            print("**label top left position not set, cannot record gaze**")
            return
        if self.eyetracker:
            gaze_x = None; gaze_y = None; duration = None

            dt = self.eyetracker.getNewestSample()
            if dt is None:
                return  # Skip invalid samples instead of recording fake data
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
                                current_file,
                                self.phase])

    def save_incremental(self):
        """Append current gaze data to CSV and save measurement JSONs incrementally."""
        if self.result_dir is None:
            print("**WARNING: result_dir not set, skipping save. Please run calibration first.**")
            return
        if self.data:
            file_path = os.path.join(self.result_dir, "Trial{}.csv".format(self.trial))
            df = pd.DataFrame(self.data, columns=["Time", "Gaze_Location_X", "Gaze_Location_Y", "Slice Number", "Duration", "File Name", "Phase"])
            write_header = not os.path.exists(file_path)
            df.to_csv(file_path, mode='a', index=False, header=write_header)
            print(f"----------------------append {len(self.data)} rows to csv----------------------")
            self.data = []

        if self.measurement_firstread:
            file_path = os.path.join(self.result_dir, "Trial{}_firstread.json".format(self.trial))
            with open(file_path, "w") as f:
                json.dump(self.measurement_firstread, f)
            print("----------------------save firstread json----------------------")

        if self.measurement_secondlook:
            file_path = os.path.join(self.result_dir, "Trial{}_secondlook.json".format(self.trial))
            with open(file_path, "w") as f:
                json.dump(self.measurement_secondlook, f)
            print("----------------------save secondlook json----------------------")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        uifile = sys.argv[1]
    else:
        uifile = 'ui-nu.ui'
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow(uifile)
    window.show()
    sys.exit(app.exec_())
