'''查看已注册参数'''

import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, 
                    QTextEdit, QPushButton, QHBoxLayout, QMessageBox, QDesktopWidget)

class viewregistered(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        try:
            with open(r"config\注册参数.json", 'r', encoding='utf-8') as f: self.registered = json.load(f)
        except:
            QMessageBox.warning(self, "警告", "未找到 注册参数.json 文件")

    def center(self):
        '''将窗口移动至屏幕中央'''
        qr = self.frameGeometry()  # 获取窗口的框架几何信息
        cp = QDesktopWidget().availableGeometry().center()  # 获取屏幕中心点
        qr.moveCenter(cp)  # 将窗口框架中心移动到屏幕中心
        self.move(qr.topLeft())  # 窗口移动到新的左上角坐标

    def initUI(self):
        '''主窗口'''
        self.resize(800, 650)
        self.center()
        self.setWindowTitle('注册参数管理')

        # 主布局
        main_widget = QWidget(self)
        self.layout = QVBoxLayout(main_widget)

        self.registeredlabel = QLabel("已注册参数:", self)
        self.registeredBox = QTextEdit(self)
        self.registeredBox.setReadOnly(True)
        self.registeredBox.setPlaceholderText("展示已注册的参数，若想修改，请于 注册参数 处以相同图片ID修改")
        self.layout.addWidget(self.registeredlabel)
        self.layout.addWidget(self.registeredBox)

        # 按钮布局
        self.button_layout = QHBoxLayout()
        self.show_button = QPushButton("展示", self)
        self.show_button.clicked.connect(self.show_registered_data)
        self.cancel_button = QPushButton("退出", self)
        self.cancel_button.clicked.connect(self.close)
        self.button_layout.addWidget(self.show_button)
        self.button_layout.addWidget(self.cancel_button)
        self.layout.addLayout(self.button_layout)

        # 应用布局
        self.setCentralWidget(main_widget)

    def show_registered_data(self):
        '''展示已注册数据'''
        self.registeredBox.clear()
        for img_id, registered_data in self.registered.items():
            self.registeredBox.append(f'注册参数：{img_id} -> {registered_data}')

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    main_window = viewregistered()
    main_window.show()
    app.exec_()    














