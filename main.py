import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QFrame, QGraphicsDropShadowEffect, 
                             QFileDialog, QMessageBox, QDialog, QListWidget, QListWidgetItem, QAbstractItemView)
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtGui import  QColor, QImage, QPixmap, QIcon

OUTPUT_DIR = 'tu'
KEY_RECYCLABLE = 'recyclable'
KEY_HAZARDOUS = 'hazardous'
KEY_KITCHEN = 'kitchen'
KEY_OTHER = 'other'

class BinGalleryDialog(QDialog):
    def __init__(self, title, color_hex, image_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(600, 400)
        self.setStyleSheet(f"background-color: white;")
        layout = QVBoxLayout(self)
        header = QLabel(f"{title} - 已收集 {len(image_paths)} 件物品")
        header.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color_hex}; margin-bottom: 10px;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(100, 100))
        self.list_widget.setSpacing(10)
        self.list_widget.setViewMode(QListWidget.IconMode)
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.setSelectionMode(QAbstractItemView.NoSelection)
        self.list_widget.setStyleSheet("""
            QListWidget { border: none; background: #F9FAFB; }
            QListWidget::item { background: white; border-radius: 8px; border: 1px solid #E5E7EB; }
        """)
        layout.addWidget(self.list_widget)
        if not image_paths:
            empty_label = QLabel("🗑️ 这个桶是空的")
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setStyleSheet("color: #9CA3AF; font-size: 16px;")
            layout.addWidget(empty_label)
            self.list_widget.hide()
        else:
            for img_path in image_paths:
                self.add_image_item(img_path)

    def add_image_item(self, path):
        item = QListWidgetItem()
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            q_img = QImage(img.data, w, h, w * ch, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            item.setIcon(QIcon(pixmap))
            item.setText(os.path.basename(path))
            self.list_widget.addItem(item)

class TrashBinWidget(QPushButton):
    def __init__(self, name, emoji, color_hex, key_type, parent=None):
        super().__init__(parent)
        self.name = name
        self.key_type = key_type
        self.color = color_hex
        self.count = 0
        self.setFixedSize(160, 180)
        self.setCursor(Qt.PointingHandCursor)
        self.layout = QVBoxLayout(self)
        self.icon_lbl = QLabel(emoji)
        self.icon_lbl.setStyleSheet("font-size: 50px; background: transparent;")
        self.icon_lbl.setAlignment(Qt.AlignCenter)
        self.name_lbl = QLabel(name)
        self.name_lbl.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {self.color};")
        self.name_lbl.setAlignment(Qt.AlignCenter)
        self.count_lbl = QLabel("0")
        self.count_lbl.setStyleSheet("font-size: 24px; font-weight: 800; color: #374151;")
        self.count_lbl.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.icon_lbl)
        self.layout.addWidget(self.name_lbl)
        self.layout.addWidget(self.count_lbl)
        self.update_style(False)

    def update_count(self, new_count):
        self.count = new_count
        self.count_lbl.setText(str(self.count))

    def enterEvent(self, event):
        self.update_style(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update_style(False)
        super().leaveEvent(event)

    def update_style(self, hover):
        bg_color = "#FFFFFF" if not hover else "#FFFFFF" # 保持白色背景
        # 悬停时边框变色，并加深阴影模拟浮起
        border = f"2px solid {self.color}" if hover else "1px solid #E5E7EB"
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                border: {border};
                border-radius: 16px;
                text-align: center;
            }}
        """)

class ModernGarbageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.bin_data = { KEY_RECYCLABLE: [], KEY_HAZARDOUS: [], KEY_KITCHEN: [], KEY_OTHER: [] }
        self.initUI()
        self.oldPos = self.pos()
        try:
            self.model = tf.keras.models.load_model("./mobilenetv2_laji.h5")
        except Exception as e:
            print(f"Error: {e}")
        self.to_predict_images = []
        self.class_names = ['一次性杯子', '卫生纸', '口罩', '指甲油','易拉罐','杀虫剂','果皮','水果','瓶子','纸袋','过期药物','食物']
        self.current_bin_key = None

    def initUI(self):
        self.setWindowTitle('智能垃圾分类系统')
        self.resize(1000, 880) # 稍微调高一点
        self.setAttribute(Qt.WA_TranslucentBackground) # 保持透明，为了圆角
        self.setWindowFlags(Qt.FramelessWindowHint)    # 无边框
        
        
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(50, 50, 50, 50) # 给阴影留出外部空间
        self.setLayout(root_layout)

        self.main_background = QFrame()
        self.main_background.setObjectName("MainBackground") # 用于设置样式
        
        content_layout = QVBoxLayout(self.main_background)
        content_layout.setContentsMargins(30, 40, 30, 40) # 内部边距
        content_layout.setSpacing(30) # 上下部分的间距

        self.center_card = QFrame()
        self.center_card.setObjectName("CenterCard")
        card_layout = QHBoxLayout(self.center_card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        # Left Panel
        self.left_panel = QFrame()
        self.left_panel.setObjectName("LeftPanel")
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setAlignment(Qt.AlignCenter)
        
        self.placeholder_widget = QWidget()
        ph_layout = QVBoxLayout(self.placeholder_widget)
        ph_layout.addWidget(QLabel("🖼️", objectName="IconLabel"))
        ph_layout.addWidget(QLabel("预览区域", objectName="GrayText"))
        ph_layout.addWidget(QLabel("等待上传图片...", objectName="LightGrayText"))
        self.img_display = QLabel()
        self.img_display.setAlignment(Qt.AlignCenter)
        self.img_display.hide()
        left_layout.addWidget(self.placeholder_widget)
        left_layout.addWidget(self.img_display)

        # Right Panel
        self.right_panel = QFrame()
        self.right_panel.setObjectName("RightPanel")
        right_main_layout = QVBoxLayout(self.right_panel)
        right_main_layout.setContentsMargins(40, 30, 40, 40)
        right_main_layout.setSpacing(20)

        # Header
        header_row = QHBoxLayout()
        title_section = QVBoxLayout()
        title_section.addWidget(QLabel("♻️ 垃圾分类助手", objectName="Title"))
        title_section.addWidget(QLabel("AI 智能识别，让环保更简单", objectName="Subtitle"))
        self.close_btn = QPushButton("×")
        self.close_btn.setFixedSize(40, 40)
        self.close_btn.setObjectName("CloseButton")
        self.close_btn.clicked.connect(self.close)
        header_row.addLayout(title_section)
        header_row.addStretch(1)
        header_row.addWidget(self.close_btn, 0, Qt.AlignTop)

        # Result Box
        self.result_box = QFrame()
        self.result_box.setObjectName("ResultBox")
        res_layout = QVBoxLayout(self.result_box)
        self.res_value = QLabel("待识别...")
        self.res_value.setObjectName("ResValue")
        self.res_value.setAlignment(Qt.AlignCenter)
        self.res_tag = QLabel("AI Ready")
        self.res_tag.setObjectName("ResTag")
        self.res_tag.setAlignment(Qt.AlignCenter)
        res_layout.addStretch()
        res_layout.addWidget(QLabel("识别结果", objectName="ResLabel"), 0, Qt.AlignCenter)
        res_layout.addWidget(self.res_value, 0, Qt.AlignCenter)
        res_layout.addWidget(self.res_tag, 0, Qt.AlignCenter)
        res_layout.addStretch()

        # Buttons
        self.btn_select = QPushButton("📤 选择图像")
        self.btn_select.setObjectName("BtnPrimary")
        self.btn_select.clicked.connect(self.change_images)
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶ 开始识别")
        self.btn_start.setObjectName("BtnSecondary")
        self.btn_start.clicked.connect(self.predict_images)
        self.btn_save = QPushButton("💾 保存到本地")
        self.btn_save.setObjectName("BtnSecondary")
        self.btn_save.clicked.connect(self.save_image_to_disk)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_save)

        right_main_layout.addLayout(header_row)
        right_main_layout.addWidget(self.result_box)
        right_main_layout.addWidget(self.btn_select)
        right_main_layout.addLayout(btn_row)

        card_layout.addWidget(self.left_panel, 6)
        card_layout.addWidget(self.right_panel, 4)

        # --- 下半部分：垃圾桶 ---
        bins_layout = QHBoxLayout()
        bins_layout.setSpacing(20)
        self.bin_widgets = {}
        bin_configs = [
            ("可回收物", "♻️", "#1E40AF", KEY_RECYCLABLE),
            ("有害垃圾", "☠️", "#991B1B", KEY_HAZARDOUS),
            ("厨余垃圾", "🍏", "#065F46", KEY_KITCHEN),
            ("其他垃圾", "🌪️", "#92400E", KEY_OTHER)
        ]
        for name, icon, color, key in bin_configs:
            bin_btn = TrashBinWidget(name, icon, color, key)
            bin_btn.clicked.connect(lambda checked, k=key: self.open_bin_dialog(k))
            bins_layout.addWidget(bin_btn)
            self.bin_widgets[key] = bin_btn

        content_layout.addWidget(self.center_card)
        content_layout.addLayout(bins_layout)
        root_layout.addWidget(self.main_background)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(0, 0, 0, 60)) 
        shadow.setOffset(0, 10)
        self.main_background.setGraphicsEffect(shadow)

        self.apply_styles()

    def change_images(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec_():
            files = file_dialog.selectedFiles()
            self.to_predict_images = files
            self.update_image_display()
            self.res_value.setText("待识别...")
            self.res_tag.setText("AI Ready")
            self.res_tag.setStyleSheet("background-color: #D1FAE5; color: #065F46;")
            self.current_bin_key = None

    def update_image_display(self):
        if not self.to_predict_images: return
        img_path = self.to_predict_images[0]
        self.placeholder_widget.hide()
        self.img_display.show()
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        q_img = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.left_panel.size() * 0.9, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_display.setPixmap(scaled_pixmap)

    def predict_images(self):
        if not self.to_predict_images:
             QMessageBox.warning(self, '提示', '请先选择图像！')
             return
        img_path = self.to_predict_images[0]
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (224, 224))
        outputs = self.model.predict(np.expand_dims(img_resized, axis=0))
        result_name = self.class_names[np.argmax(outputs)]
        
        category_text = ""
        tag_style = ""
        bin_key = KEY_OTHER

        if result_name in ['果皮', '水果', '食物']:
            category_text = "厨余垃圾"
            bin_key = KEY_KITCHEN
            tag_style = "background-color: #D1FAE5; color: #065F46;"
        elif result_name in ['瓶子', '纸袋', '易拉罐']:
            category_text = "可回收物"
            bin_key = KEY_RECYCLABLE
            tag_style = "background-color: #DBEAFE; color: #1E40AF;"
        elif result_name in ['杀虫剂', '过期药物', '指甲油']:
            category_text = "有害垃圾"
            bin_key = KEY_HAZARDOUS
            tag_style = "background-color: #FEE2E2; color: #991B1B;"
        else:
            category_text = "其他垃圾"
            bin_key = KEY_OTHER
            tag_style = "background-color: #FEF3C7; color: #92400E;"

        self.res_value.setText(result_name)
        self.res_tag.setText(category_text)
        self.res_tag.setStyleSheet(f"QLabel {{ {tag_style} border-radius: 10px; padding: 4px 12px; font-weight: bold; }}")
        self.current_bin_key = bin_key
        self.throw_trash_into_bin(bin_key, img_path)

    def throw_trash_into_bin(self, key, img_path):
        self.bin_data[key].append(img_path)
        current_count = len(self.bin_data[key])
        self.bin_widgets[key].update_count(current_count)

    def open_bin_dialog(self, key):
        data = self.bin_data[key]
        widget = self.bin_widgets[key]
        dialog = BinGalleryDialog(widget.name, widget.color, data, self)
        dialog.exec_()

    def save_image_to_disk(self):
        if not self.to_predict_images or not self.current_bin_key: return
        directory = os.path.join(OUTPUT_DIR, self.current_bin_key)
        if not os.path.exists(directory): os.makedirs(directory)
        img_path = self.to_predict_images[0]
        save_path = os.path.join(directory, os.path.basename(img_path))
        cv2.imwrite(save_path, cv2.imread(img_path))
        QMessageBox.information(self, "保存成功", f"文件已保存至: {directory}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: self.oldPos = event.globalPos()
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            delta = QPoint(event.globalPos() - self.oldPos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPos = event.globalPos()

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget { font-family: 'Microsoft YaHei UI', 'Microsoft YaHei', sans-serif; }
            
            #MainBackground {
                background-color: #F3F4F6; 
                border-radius: 24px;       
                border: 1px solid #E5E7EB; 
            }

            /* CenterCard 保持白色 */
            #CenterCard { 
                background-color: #FFFFFF; 
                border-radius: 20px; 
                
            }
            
            #LeftPanel { background-color: #E5E7EB; border-top-left-radius: 20px; border-bottom-left-radius: 20px; }
            #RightPanel { background-color: #FFFFFF; border-top-right-radius: 20px; border-bottom-right-radius: 20px; }
            #ResultBox { background-color: #F3F4F6; border: 2px dashed #9CA3AF; border-radius: 12px; min-height: 180px; }
            
            QPushButton { border-radius: 8px; font-weight: 600; font-size: 14px; padding: 10px; }
            #BtnPrimary { background-color: #10B981; color: white; border: none; }
            #BtnPrimary:hover { background-color: #059669; }
            #BtnSecondary { background-color: white; color: #374151; border: 1px solid #D1D5DB; }
            #BtnSecondary:hover { border-color: #10B981; color: #10B981; }
            
            #Title { font-size: 24px; font-weight: bold; color: #111827; }
            #Subtitle { font-size: 13px; color: #6B7280; }
            #ResValue { font-size: 32px; font-weight: 800; color: #1F2937; margin: 10px 0; }
            #CloseButton { background: transparent; color: #9CA3AF; font-size: 24px; border: none; }
            #CloseButton:hover { background: #FEE2E2; color: #EF4444; border-radius: 20px; }
            
            QLabel[objectName="IconLabel"] { font-size: 60px; background: transparent; }
            QLabel[objectName="GrayText"] { color: #6B7280; font-weight: bold; font-size: 16px; margin-top:10px; }
            QLabel[objectName="LightGrayText"] { color: #9CA3AF; font-size: 12px; margin-top:5px; }
        """)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ModernGarbageApp()
    ex.show()
    sys.exit(app.exec_())