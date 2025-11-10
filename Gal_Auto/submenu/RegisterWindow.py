'''
注册窗口，实现让用户输入图片id，注册参数
以id为键，注册参数为值，存入json文件中
'''
import json
import os
import ast
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLineEdit, QTextEdit,
    QPushButton, QLabel, QMessageBox, QWidget, QDesktopWidget
)


class register(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.img_id = None
        self.params = None
        # 加载已有的注册参数
        try:
            with open(r'config\注册参数.json', 'r', encoding='utf-8') as f:
                self.register_dic = json.load(f)
               
        except FileNotFoundError:
            self.register_dic = {}
            QMessageBox.warning(self, "警告", "注册参数文件未找到，将创建新的文件。")
        except json.JSONDecodeError:
            self.register_dic = {}
            QMessageBox.warning(self, "警告", "注册参数文件格式错误，将重置文件内容。")

    def center(self):
        '''将窗口移动至屏幕中央'''
        qr = self.frameGeometry()  # 获取窗口的框架几何信息
        cp = QDesktopWidget().availableGeometry().center()  # 获取屏幕中心点
        qr.moveCenter(cp)  # 将窗口框架中心移动到屏幕中心
        self.move(qr.topLeft())  # 窗口移动到新的左上角坐标

    def initUI(self):
        '''主窗口'''
        self.resize(1000, 700)
        self.center()
        self.setWindowTitle('注册参数管理')

        # 主布局
        main_widget = QWidget(self)
        self.layout = QVBoxLayout(main_widget)

        # 图片ID输入
        self.id_layout = QHBoxLayout()
        self.id_label = QLabel("图片ID:", self)
        self.id_input = QLineEdit(self)
        self.id_input.setPlaceholderText("请输入图片ID")
        self.id_layout.addWidget(self.id_label)
        self.id_layout.addWidget(self.id_input)
        self.layout.addLayout(self.id_layout)

        # 注册参数输入
        self.params_label = QLabel("注册参数:", self)
        self.params_input = QTextEdit(self)
        self.params_input.setPlaceholderText("请输入注册参数，格式为： [[1893, 2609], [2022, 545], 0] \n注：必须使用英文字符，[]中的内容分别为Initial Point 和 Final Point，0是默认的")
        self.layout.addWidget(self.params_label)
        self.layout.addWidget(self.params_input)

        # 操作结果输出框
        self.resultBox = QTextEdit(self)
        self.resultBox.setReadOnly(True)
        self.resultBox.setPlaceholderText('参数注册状况...')
        self.layout.addWidget(self.resultBox)

        # 按钮布局
        self.button_layout = QHBoxLayout()
        self.save_button = QPushButton("保存", self)
        self.save_button.clicked.connect(self.save_register_data)
        self.cancel_button = QPushButton("退出", self)
        self.cancel_button.clicked.connect(self.close)
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.cancel_button)
        self.layout.addLayout(self.button_layout)

        # 应用布局
        self.setCentralWidget(main_widget)

    def save_register_data(self):
        img_id = self.id_input.text().strip()
        params_str = self.params_input.toPlainText().strip()
        
        # 输入非空校验
        if not img_id or not params_str:
            QMessageBox.warning(self, "错误", "图片ID或注册参数不能为空！")
            return
        
        # 解析参数
        try:
            params = ast.literal_eval(params_str)
            if not isinstance(params, list) or len(params) != 3:
                raise ValueError("参数必须为 [[x1,y1], [x2,y2], threshold] 格式")
        except Exception as e:
            QMessageBox.critical(self, "参数错误", f"格式校验失败: {str(e)}")
            return

        # 目录检查
        os.makedirs("config", exist_ok=True)

        try:
            # 存储参数
            self.register_dic[img_id] = params

            with open(r"config\注册参数.json", 'w', encoding='utf-8') as f:
                json.dump(self.register_dic, f, indent=4, ensure_ascii=False)

            self.resultBox.append(f"成功写入: {img_id} -> {params}")
        except Exception as e:
            self.resultBox.append(f"保存失败: {img_id} -> {params}, 错误: {str(e)}")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    main_window = register()
    main_window.show()
    app.exec_()    











