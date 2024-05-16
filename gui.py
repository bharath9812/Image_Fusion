

import sys
from PIL import Image, ImageEnhance, ImageQt
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
                             QComboBox, QFileDialog, QMessageBox, QFrame)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMenu, QAction
import os


# from scripts import dwt_script as dwt
# from scripts import laplace_region_based_multi_resolution as laplace

scripts_path ='/Users/chintubharath/Documents/sdp/scripts'
sys.path.insert(1, scripts_path)

# import dwt_script as dwt
import laplace_region_based_multi_resolution as laplace
import vgg19 as vgg_cnn
import pca_ihs as pca_ihs
import dwt_adv_v2 as dwt_v2

class ImageFusionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Image Fusion Application'
        self.left = 10
        self.top = 10
        self.width = 1000
        self.height = 700
        self.image_path1 = None
        self.image_path2 = None
        self.fused_image = None  # Initialize the fused_image attribute
        self.vgg_fused_image = None  # Separate attribute for VGG fused image
        self.recentFiles = []  # List to store paths of recent files
        self.maxRecentFiles = 10  # Maximum number of recent files to track
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: 2px solid #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLabel {
                color: #FFF;
            }
        """)

        # Main layout is a horizontal layout
        main_layout = QHBoxLayout()
        self.input_panel = QWidget()
        input_layout = QVBoxLayout(self.input_panel)

        # Layout for model selection
        model_selection_layout = QHBoxLayout()
        
        # Label for model selection
        model_label = QLabel("Select the model:")
        model_label.setStyleSheet("color: white;")  # Setting the color of the label text

        # ComboBox for selecting the fusion method
        self.comboBox = QComboBox()
        self.comboBox.addItem("DWT Fusion")
        self.comboBox.addItem("Laplacian Fusion")
        self.comboBox.addItem("VGG 19")
        self.comboBox.addItem("PCA")
        self.comboBox.addItem("IHS")
        self.comboBox.setFixedWidth(150)  # Set the width of the ComboBox

        # Adding label and ComboBox to the model selection layout
        model_selection_layout.addWidget(model_label)
        model_selection_layout.addWidget(self.comboBox)
        model_selection_layout.addStretch(1)  # Add stretch to push the ComboBox closer to the label
        
        # Setup buttons
        self.btn_load1 = QPushButton('Load Image 1')
        self.btn_load2 = QPushButton('Load Image 2')
        self.btn_fuse = QPushButton('Fuse Images')
        self.btn_save = QPushButton('Save Image')
        self.btn_clear = QPushButton('Clear All')  # Clear button
        self.btn_upscale = QPushButton('Upscale Image')

        # Connect buttons to functions
        self.btn_load1.clicked.connect(lambda: self.load_image(1))
        self.btn_load2.clicked.connect(lambda: self.load_image(2))
        self.btn_fuse.clicked.connect(self.process_images)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_clear.clicked.connect(self.clear_images)
        self.btn_upscale.clicked.connect(self.upscale_image)

        # Calculate and set a fixed width for the buttons
        button_width = max(self.btn_load1.sizeHint().width(), self.btn_load2.sizeHint().width(),
                           self.btn_fuse.sizeHint().width(), self.btn_upscale.sizeHint().width(), 
                           self.btn_save.sizeHint().width(), self.btn_clear.sizeHint().width()) + 20
        self.btn_load1.setFixedWidth(button_width)
        self.btn_load2.setFixedWidth(button_width)
        self.btn_fuse.setFixedWidth(button_width)
        self.btn_upscale.setFixedWidth(button_width)
        self.btn_save.setFixedWidth(button_width)
        self.btn_clear.setFixedWidth(button_width)

        # Layout for buttons
        load_button_layout = QHBoxLayout()
        load_button_layout.addWidget(self.btn_load1)
        load_button_layout.addWidget(self.btn_load2)

        # Image labels setup
        self.img_label1 = QLabel()
        self.img_label2 = QLabel()
        self.img_label1.setFixedSize(350, 350)
        self.img_label2.setFixedSize(350, 350)

        # Image display layout
        image_display_layout = QHBoxLayout()
        image_display_layout.addWidget(self.img_label1)
        image_display_layout.addWidget(self.img_label2)

        # Input layout
        input_layout.addLayout(model_selection_layout)
        input_layout.addLayout(load_button_layout)
        input_layout.addLayout(image_display_layout)
        input_layout.addWidget(self.btn_fuse, alignment=Qt.AlignCenter)
        input_layout.addWidget(self.btn_clear, alignment=Qt.AlignCenter)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setFixedWidth(2)

        # Output panel needs to be defined before adding to layout
        self.output_panel = QWidget()
        output_layout = QVBoxLayout(self.output_panel)
        self.output_label = QLabel()
        self.output_label.setFixedSize(500, 500)
        output_layout.addWidget(self.output_label, alignment=Qt.AlignCenter)
        output_layout.addWidget(self.btn_upscale, alignment=Qt.AlignCenter)  # Add upscale button
        output_layout.addWidget(self.btn_save, alignment=Qt.AlignCenter)

        # Add panels to main layout
        main_layout.addWidget(self.input_panel)
        main_layout.addWidget(separator)  # Add separator to the main layout
        main_layout.addWidget(self.output_panel)

        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Menu for recent files
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        self.recentFileActs = [QAction(self, visible=False) for _ in range(self.maxRecentFiles)]
        for action in self.recentFileActs:
            action.triggered.connect(self.openRecentFile)
            fileMenu.addAction(action)
        self.updateRecentFileActions()

    def clear_images(self):
        """Clear all images from input and output."""
        self.img_label1.clear()
        self.img_label2.clear()
        self.output_label.clear()
        self.image_path1 = None
        self.image_path2 = None
        self.fused_image = None  # Clear the fused_image attribute
        self.vgg_fused_image = None  # Clear the VGG fused image attribute

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event, img_number):
        urls = event.mimeData().urls()
        if urls and len(urls) > 0:
            filepath = str(urls[0].toLocalFile())
            self.load_image(img_number, filepath)

    def load_image(self, img_number, filepath=None):
        if not filepath:
            filepath, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png)")
        if filepath:
            pixmap = QPixmap(filepath)
            label = self.img_label1 if img_number == 1 else self.img_label2
            label.setPixmap(pixmap.scaled(350, 350, Qt.KeepAspectRatio))
            if img_number == 1:
                self.image_path1 = filepath
            else:
                self.image_path2 = filepath
            self.updateRecentFiles(filepath)

    def updateRecentFiles(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        self.recentFiles.insert(0, filePath)
        if len(self.recentFiles) > self.maxRecentFiles:
            self.recentFiles.pop()
        self.updateRecentFileActions()

    def updateRecentFileActions(self):
        for i, filePath in enumerate(self.recentFiles):
            text = os.path.basename(filePath)
            self.recentFileActs[i].setText(text)
            self.recentFileActs[i].setData(filePath)
            self.recentFileActs[i].setVisible(True)

    def openRecentFile(self):
        action = self.sender()
        if action:
            filepath = action.data()
            if os.path.exists(filepath):
                self.load_image(1 if self.image_path1 is None else 2, filepath)

    def process_images(self):
        if self.image_path1 is None or self.image_path2 is None:
            QMessageBox.warning(self, 'Error', 'Please load both images before fusing.')
            return

        img1 = cv2.imread(self.image_path1)
        img2 = cv2.imread(self.image_path2)

        if img1 is None or img2 is None:
            QMessageBox.warning(self, 'Error', 'Failed to load images. Check file paths.')
            return

        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))

        method = self.comboBox.currentIndex()
        fused_image = None
        if method == 0:
            fused_image = dwt_v2.fusion_process(img1, img2, wavelet='db5', level=2)
        elif method == 1:
            input_images = [img1, img2]
            fusion_instance = laplace.Fusion(input_images)
            fused_image = fusion_instance.fuse()
            fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)
        elif method == 2:
            fused_image = vgg_cnn.predict_fused_image(self.image_path1, self.image_path2)
            self.vgg_fused_image = fused_image  # Store the VGG fused image separately
            self.display_vgg_image(fused_image)  # Call separate display function for VGG19
            return
        elif method == 3:
            fused_image = pca_ihs.image_fusion_pca(img1, img2, num_components=3)
        elif method == 4:
            fused_image = pca_ihs.image_fusion_ihs(img1, img2)

        if fused_image is not None:
            self.fused_image = fused_image  # Store the fused image
            self.display_image(fused_image)

    def upscale_image(self):
        method = self.comboBox.currentIndex()
        if method == 2:  # VGG19
            self.upscale_vgg_image()
        else:
            self.upscale_general_image()

    def upscale_general_image(self):
        if self.fused_image is None:
            QMessageBox.warning(self, 'Error', 'No fused image to upscale.')
            return

        try:
            # Ensure the image has three channels
            if len(self.fused_image.shape) == 2:
                self.fused_image = cv2.cvtColor(self.fused_image, cv2.COLOR_GRAY2RGB)
            elif self.fused_image.shape[2] == 1:
                self.fused_image = cv2.cvtColor(self.fused_image, cv2.COLOR_GRAY2RGB)

            # Upscale image using super-resolution
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            path = "EDSR_x4.pb"  # Path to the super-resolution model
            if not os.path.isfile(path):
                QMessageBox.warning(self, 'Error', f'Model file not found: {path}')
                return

            sr.readModel(path)
            sr.setModel("edsr", 4)  # Set the model and scale

            upscaled_image = sr.upsample(self.fused_image)
        
            # Enhance contrast
            pil_image = Image.fromarray(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced_image = enhancer.enhance(2.0)  # Enhance contrast

            # Convert back to numpy array
            enhanced_image = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)
            self.display_image(enhanced_image)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to upscale image: {str(e)}')

    def upscale_vgg_image(self):
        if self.vgg_fused_image is None:
            QMessageBox.warning(self, 'Error', 'No VGG fused image to upscale.')
            return

        try:
            # Print shape of vgg_fused_image for debugging
            print(f"Shape of vgg_fused_image before processing: {self.vgg_fused_image.shape}")

            # Ensure the image has a single channel (grayscale)
            if len(self.vgg_fused_image.shape) == 3 and self.vgg_fused_image.shape[2] == 3:
                self.vgg_fused_image = cv2.cvtColor(self.vgg_fused_image, cv2.COLOR_RGB2GRAY)

            # Print shape after conversion to grayscale
            print(f"Shape of vgg_fused_image after grayscale conversion: {self.vgg_fused_image.shape}")

            # Ensure the image is 2-dimensional by squeezing the extra dimension
            if len(self.vgg_fused_image.shape) == 3 and self.vgg_fused_image.shape[2] == 1:
                self.vgg_fused_image = np.squeeze(self.vgg_fused_image, axis=2)

            # Print shape after squeezing
            print(f"Shape of vgg_fused_image after squeezing: {self.vgg_fused_image.shape}")

            # Ensure the image is 2-dimensional
            if len(self.vgg_fused_image.shape) != 2:
                raise ValueError("VGG fused image is not 2-dimensional")

            # Convert the fused image to a 3-channel image by stacking
            fused_image_3c = np.stack([self.vgg_fused_image] * 3, axis=-1)

            # Print shape after stacking
            print(f"Shape of fused_image_3c after stacking: {fused_image_3c.shape}")

            # Upscale image using super-resolution
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            path = "EDSR_x4.pb"  # Path to the super-resolution model
            sr.readModel(path)
            sr.setModel("edsr", 4)  # Set the model and scale
            upscaled_image = sr.upsample(fused_image_3c)

            # Print shape after upscaling
            print(f"Shape of upscaled_image: {upscaled_image.shape}")

            # Convert back to grayscale
            upscaled_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)

            # Normalize the image to the full range [0, 255]
            upscaled_image = (upscaled_image - upscaled_image.min()) / (upscaled_image.max() - upscaled_image.min()) * 255.0
            upscaled_image = np.clip(upscaled_image, 0, 255).astype(np.uint8)

            # Ensure the upscaled image is contiguous in memory
            upscaled_image = np.ascontiguousarray(upscaled_image)

            # Create QImage from the numpy array
            height, width = upscaled_image.shape
            qImg = QImage(upscaled_image.data, width, height, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qImg)
            self.output_label.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))
        except Exception as e:
            print(f"Failed to upscale VGG image: {str(e)}")
            QMessageBox.critical(self, 'Error', f'Failed to upscale VGG image: {str(e)}')

    def display_image(self, fused_image):
        self.output_label.clear()
        if len(fused_image.shape) == 2:
            fused_image = cv2.cvtColor(fused_image, cv2.COLOR_GRAY2RGB)

        height, width, channel = fused_image.shape
        bytes_per_line = 3 * width
        qImg = QImage(fused_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        self.output_label.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))

    def display_vgg_image(self, fused_image):
        self.output_label.clear()  # Clear the output label

        # Check if the shape has an extra dimension
        if len(fused_image.shape) == 4 and fused_image.shape[0] == 1:
            fused_image = np.squeeze(fused_image, axis=0)
        elif len(fused_image.shape) == 3 and fused_image.shape[2] == 1:
            fused_image = np.squeeze(fused_image, axis=2)

        # Normalize the image to the full range [0, 255]
        fused_image = (fused_image - fused_image.min()) / (fused_image.max() - fused_image.min()) * 255.0
        fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)

        height, width = fused_image.shape

        # Ensure the fused image is contiguous in memory
        fused_image = np.ascontiguousarray(fused_image)

        # Create QImage from the numpy array
        qImg = QImage(fused_image.data, width, height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qImg)
        self.output_label.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))
        
    def save_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg)", options=options)
        if file_name:
            self.output_label.pixmap().save(file_name)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageFusionApp()
    ex.show()
    sys.exit(app.exec_())
