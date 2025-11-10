from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QRadioButton, QPushButton
from PyQt5.QtCore import pyqtSignal

class SelectFiletype(QMainWindow):
    format_selected = pyqtSignal(str)  # 定义一个信号

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('选择图片格式')
        self.resize(500, 300)

        # 创建中央部件和栅格布局
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        grid = QGridLayout()

        # 添加单选按钮到栅格布局
        self.radio_buttons = {}
        formats = ['jpg', 'png', 'bmp', 'tif']
        for i, fmt in enumerate(formats):
            radio = QRadioButton(fmt, self)
            grid.addWidget(radio, i // 2, i % 2)  # 每行 2 个按钮
            self.radio_buttons[fmt] = radio

        # 添加确定按钮
        self.confirm_button = QPushButton('确定', self)
        self.confirm_button.setEnabled(False)
        grid.addWidget(self.confirm_button, len(formats) // 2 + 1, 0, 1, 2)

        # 信号连接
        for btn in self.radio_buttons.values():
            btn.toggled.connect(self.update_confirm_button)
        self.confirm_button.clicked.connect(self.confirm_selection)  # 按钮点击事件

        # 设置中央部件
        central_widget.setLayout(grid)

    def update_confirm_button(self):
        """更新确定按钮状态"""
        self.confirm_button.setEnabled(any(btn.isChecked() for btn in self.radio_buttons.values()))

    def confirm_selection(self):
        """发射选中的文件格式信号"""
        for fmt, btn in self.radio_buttons.items():
            if btn.isChecked():
                self.format_selected.emit(fmt)  # 发射信号
                break
        self.close()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    main_window = SelectFiletype()
    main_window.show()
    app.exec_()    
 










