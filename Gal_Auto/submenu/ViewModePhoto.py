from glob import glob
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QDesktopWidget
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt  

class ViewModePhoto(QMainWindow):

    def __init__(self):
        super().__init__()
        self.images = glob(r'mode_photo\*') 
        self.current_index = 0 
        self.initUI()

    def center(self):
        """将窗口移动至屏幕中央"""
        qr = self.frameGeometry()  # 获取窗口的框架几何信息
        cp = QDesktopWidget().availableGeometry().center()  # 获取屏幕中心点
        qr.moveCenter(cp)  # 将窗口框架中心移动到屏幕中心
        self.move(qr.topLeft())  # 窗口移动到新的左上角坐标

    def initUI(self):
        """主窗口初始化"""
        self.resize(800, 650)
        self.center()
        self.setWindowTitle('模版图片查看器')

        # 中央控件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()

        # 图片展示区域
        self.label = QLabel()
        self.label.setFixedSize(500, 400)  # 固定图片显示区域大小
        self.label.setStyleSheet("border: 1px solid black;")  # 添加边框
        main_layout.addWidget(self.label, alignment=Qt.AlignCenter)  # 图片居中显示

        # 文件名显示
        self.filename_label = QLabel()
        self.filename_label.setStyleSheet("font-size: 32px; color: black;")
        main_layout.addWidget(self.filename_label, alignment=Qt.AlignCenter)

        # 按钮布局
        button_layout = QHBoxLayout()

        # 上一张按钮
        previous_button = QPushButton('上一张')
        previous_button.clicked.connect(self.previous_image)
        button_layout.addWidget(previous_button)

        # 下一张按钮
        next_button = QPushButton('下一张')
        next_button.clicked.connect(self.next_image)
        button_layout.addWidget(next_button)

        # 将按钮布局添加到主布局
        main_layout.addLayout(button_layout)

        # 设置中央控件的布局
        central_widget.setLayout(main_layout)

        # 显示第一张图片
        self.show_image()

    def show_image(self):
        """显示当前图片"""
        if self.images:  # 如果有图片
            pixmap = QPixmap(self.images[self.current_index])
            self.label.setPixmap(pixmap.scaled(self.label.size(), aspectRatioMode=1))  # 调整图片大小适应显示区域
            self.filename_label.setText(f'文件名：{self.images[self.current_index]}')
        else:  # 如果没有图片
            self.label.clear()
            self.filename_label.setText('未找到图片')

    def next_image(self):
        """显示下一张图片"""
        if self.images:
            self.current_index = (self.current_index + 1) % len(self.images)  # 循环切换到下一张图片
            self.show_image()

    def previous_image(self):
        """显示上一张图片"""
        if self.images:
            self.current_index = (self.current_index - 1) % len(self.images)  # 循环切换到上一张图片
            self.show_image()


if __name__ == '__main__':
    app = QApplication([])
    viewer = ViewModePhoto()
    viewer.show()
    app.exec_()
