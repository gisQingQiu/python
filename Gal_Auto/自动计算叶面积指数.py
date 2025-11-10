
import os
import sys
import json
import time
import pyautogui
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QFileDialog, QMessageBox, QDesktopWidget,
    QTextEdit, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QAction
)
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon
from glob import glob
from submenu.SelectFiletype import SelectFiletype
from submenu.RegisterWindow import register
from submenu.ViewRegistered import viewregistered
from submenu.ViewModePhoto import ViewModePhoto
from submenu.HelpWindow import HelpWindow
from Pyauto_calculation.auto_control import auto_click, register_image, wait_result


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.selected_format = None
        self.select_work_space = None
        self.output_file_path = None
    
    def closeEvent(self, event):
        """处理窗口关闭事件"""
        reply = QMessageBox.question(self, '确认退出', '确定要退出吗？', 
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    
    def center(self):
        '''将窗口移动至屏幕中央'''
        qr = self.frameGeometry()  # 获取窗口的框架几何信息
        cp = QDesktopWidget().availableGeometry().center()  # 获取屏幕中心点
        qr.moveCenter(cp)  # 将窗口框架中心移动到屏幕中心
        self.move(qr.topLeft())  # 窗口移动到新的左上角坐标

    def initUI(self):
        '''设计程序主界面'''
        # 主菜单
        menubar = self.menuBar()    # 实例化主窗口的QMenuBar对象
        fileMenu = menubar.addMenu('文件')    # 向菜单栏中添加新的QMenu对象，父菜单       
        register = menubar.addMenu('注册')
        viewMenu = menubar.addMenu('查看')
        start = menubar.addMenu('启动')
        helptext = menubar.addMenu('帮助')

        # 文件菜单中的操作
        action_select_format = QAction('工作图片格式', self)
        action_select_format.triggered.connect(self.open_select_filetype)  # 关联到槽函数
        fileMenu.addAction(action_select_format)

        work_space = QAction('工作文件夹', self)
        work_space.triggered.connect(self.choose_work_space)  # 关联到槽函数
        fileMenu.addAction(work_space)

        action_select_output = QAction('输出文件', self)
        action_select_output.triggered.connect(self.select_output_file)  # 关联槽函数
        fileMenu.addAction(action_select_output)

        register_infomation = QAction('输入图片注册参数', self)
        register_infomation.triggered.connect(self.input_register_infomation)  # 关联槽函数
        register.addAction(register_infomation)

        view_registered = QAction('查看已注册参数', self)
        view_registered.triggered.connect(self.view_registered_infomation)  # 关联槽函数
        viewMenu.addAction(view_registered)

        view_mode_photo = QAction('查看模版图', self)
        view_mode_photo.triggered.connect(self.show_mode_photo)
        viewMenu.addAction(view_mode_photo)

        start_auto_control = QAction('启动软件计算', self)
        start_auto_control.triggered.connect(self.start_calculation)
        start.addAction(start_auto_control)

        help_window = QAction('帮助文档', self)
        helptext.triggered.connect(self.open_helpwindow)
        helptext.addAction(help_window)

        # 中央小部件
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        # 运行输出框
        self.outputBox = QTextEdit()
        self.outputBox.setReadOnly(True)
        self.outputBox.setPlaceholderText('程序运行状况...')
        # 图片处理状况
        self.returnBox = QTextEdit()
        self.returnBox.setReadOnly(True)
        self.returnBox.setPlaceholderText('图片处理结果...')
        # 布局
        hbox = QHBoxLayout()
        hbox.addWidget(self.outputBox)
        hbox.addWidget(self.returnBox)

        # 退出按钮
        qbtn = QPushButton('退出', self)
        qbtn.clicked.connect(QCoreApplication.instance().quit)

        # 垂直布局，右下角按钮
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(qbtn, alignment=Qt.AlignRight | Qt.AlignBottom)

        centralWidget.setLayout(vbox)

        self.resize(1400, 900)
        self.center()
        self.setWindowTitle('基于CV2判别和自动化模块控制的自动计算叶面积指数系统')
        self.setWindowIcon(QIcon(r"photo\图标.png"))
        self.show()

    def open_select_filetype(self):
        """打开选择文件格式窗口"""
        self.filetype_window = SelectFiletype()
        self.filetype_window.format_selected.connect(self.update_selected_format)
        self.filetype_window.show()

    def update_selected_format(self, fmt):
        self.selected_format = fmt
        self.outputBox.append(f"\n您选择的图片格式是: {self.selected_format}")

    def choose_work_space(self):
        self.select_work_space = QFileDialog.getExistingDirectory()
        self.outputBox.append(f"\n您选择的工作路径是: {self.select_work_space}")

    def select_output_file(self):
        """选择输出文件的路径和文件名，默认后缀为 .sum"""

        # 打开保存文件对话框
        output_file, _ = QFileDialog.getSaveFileName(
            self,
            caption="选择保存文件路径",
            directory=' ',
            filter="SUM 文件 (*.sum)")
        
        if output_file:
            # 检查文件名是否带有后缀，如果没有，默认加上 .sum
            if not output_file.endswith('.sum'):
                output_file += '.sum'

            self.outputBox.append(f"\n您选择的文件保存路径是: {output_file}")
            self.output_file_path = output_file
    
    def input_register_infomation(self):
        """打开 输入图片注册参数 窗口"""
        self.registerWindow = register()
        self.registerWindow.show()

    def view_registered_infomation(self):
        '''打开 查看已注册参数 的窗口'''
        self.Viewregistered = viewregistered()
        self.Viewregistered.show()

    def open_helpwindow(self):
        '''打开帮助文档'''
        self.helpwindow = HelpWindow()
        self.helpwindow.show()
    
    def show_mode_photo(self):
        '''查看模版图'''
        QMessageBox.information(self, "提示", "模版图涉及自动控制实现原理部分\n\n不得随意更改，详情见 help")
        self.mode_photo = ViewModePhoto()
        self.mode_photo.show()

    def start_calculation(self):
        '''开始计算'''
        if self.selected_format == None:
            QMessageBox.warning(self, "警告", "图片格式不得为空！")
            self.outputBox.append("\n图片格式不得为空")
            return
        if self.select_work_space == None:
            QMessageBox.warning(self, "警告", "工作文件夹不得为空！")
            self.outputBox.append('\n工作文件夹不得为空')
            return
        if self.output_file_path == None:
            QMessageBox.warning(self, "警告", "输出路径不得为空！")
            self.outputBox.append('\n输出路径不得为空')
            return
        
        self.returnBox.clear()
        with open(r'config\注册参数.json', 'r', encoding='utf-8') as f: self.register_dic = json.load(f)
        self.returnBox.append(f'\n已读取 {len(self.register_dic)} 组注册参数')
        self.paths = glob(os.path.normpath(os.path.join(self.select_work_space, '*.' + self.selected_format)))
        
        self.returnBox.append(f'\n工作空间中有 {len(self.paths)} 张图片')
        self.returnBox.append('\n参数检查通过，开始计算')

        reply = QMessageBox.question(
        self, 
        "打开方式", 
        "是否通过软件路径进行打开GLA？\n\n如果不需要，请确保GLA快捷方式在任务栏中固定", 
        QMessageBox.Yes | QMessageBox.No, 
        QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            file_path, _ = QFileDialog.getOpenFileName(self, "选择启动程序文件", '', "GAL启动程序 (*.exe)")
            if not file_path:  # 用户未选择文件
                QMessageBox.warning(self, "警告", "未选择文件，跳过此步骤。")
                self.outputBox.append("\n未选择文件，退出计算。")
                return

            if os.path.basename(file_path) != "GLA_v2.exe":  # 检查文件名是否正确
                QMessageBox.warning(self, "警告", "选择的文件并非 GLA_v2.exe 启动程序，退出计算")
                self.outputBox.append("\n用户选择的路径不是GAL启动程序，退出计算。")
                return

            self.outputBox.append(f"\n已选择的文件路径为：{file_path}")
            os.startfile(file_path)
            self.outputBox.append('\n软件启动成功...')

        else: 
            self.outputBox.append("\n用户选择直接启动软件。")

            if not wait_result(r'mode_photo\Software.png', 2, 20):  # 等待软件响应
                QMessageBox.warning(self, "警告", "软件打开失败，请确保GLA快捷方式在任务栏中固定")
                self.outputBox.append("\n软件打开失败，请确保GLA快捷方式在任务栏中固定")
                return
            auto_click(img_mode_path=r"mode_photo\Software.png")
            self.outputBox.append('\n软件启动成功...')
        time.sleep(7)  # 等待软件启动
        auto_click(img_mode_path=r"mode_photo\Full_Screen_Mode.png")
        time.sleep(1)

        unfinish = []
        for i, path in enumerate(self.paths):
            self.returnBox.append('\n' + '-'*55)
            self.returnBox.append(f"\n进程{i + 1}/{len(self.paths)}, 正在处理 {path} ...")
            # 获取注册参数
            key = os.path.splitext(os.path.basename(path))[0]
            if key in self.register_dic.keys():
                ip, fp, times = self.register_dic[key]
                self.outputBox.append(f'\n图片 {path} 的注册参数已找到: ip={ip}, fp={fp}, times={times}')
            else:
                self.returnBox.append(f'\n进程{i + 1}/{len(self.paths)}, 图片 {path} 未注册')
                unfinish.append(f'{path} 未注册')
                continue
            time.sleep(1)
            auto_click(img_mode_path=r"mode_photo\open_file.png")
            time.sleep(1)
            # 检查当前是否为第一个图片
            if i == 0:
                time.sleep(1)
                pyautogui.write(path, interval=0.025)
            else:
                time.sleep(1)
                auto_click(img_mode_path=r"mode_photo/open_new_photo.png")
                time.sleep(1)
                pyautogui.write(path, interval=0.025)
                
            pyautogui.press("enter")
            time.sleep(1)
            auto_click(img_mode_path=r"mode_photo\open_photo.png")
            time.sleep(1)
            if not wait_result(r'mode_photo/opened_image.png', 3):
                self.returnBox.append(f'\n{path} 打开图片失败')
                unfinish.append(f'{path} 打开图片失败')
                continue

            self.outputBox.append(f'\n{path} 正在注册')
            time.sleep(1)
            register_image(ip, fp)
            time.sleep(1)
            if not wait_result(r'mode_photo/finished_register.png'):
                self.returnBox.append(f'\n{path} 注册图片失败')
                unfinish.append(f'{path} 注册图片失败')
                continue
            time.sleep(1)
            auto_click(img_mode_path=r'mode_photo/threshold_image.png')
            time.sleep(1)
            if not wait_result(r'mode_photo/open_threshold.png', 2, 60):
                self.returnBox.append(f'\n{path} 计算阈值失败')
                unfinish.append(f'{path} 计算阈值失败')
                continue
            
            time.sleep(1)
            # 调整阈值
            if times > 0:
                auto_click(img_mode_path=r'mode_photo/upper_threshold.png', clicks=times, interval=1.5)
            elif times < 0:
                auto_click(img_mode_path=r'mode_photo/reduce_threshild.png', clicks=-times, interval=1.5)
            else:
                pass
            time.sleep(1)
            pyautogui.moveTo(1954, 1502, duration=0.5)
            pyautogui.click()
            time.sleep(1)
            auto_click(img_mode_path=r'mode_photo/calculation_image.png')
            time.sleep(1)
            auto_click(img_mode_path=r'mode_photo/calculation_enter.png')
            self.outputBox.append(f'\n{path} 正在计算')

            if not wait_result(r'mode_photo/ending_cal.png', 6, 600):
                self.returnBox.append(f'\n{path} 计算图片失败')
                unfinish.append(f'{path} 计算图片失败')
                continue
            time.sleep(1)
            # 保存结果
            auto_click(img_mode_path=r'mode_photo/write_info.png')
            pyautogui.write(f'{key}', interval=0.02)
            pyautogui.press("enter")
            time.sleep(1)
            auto_click(img_mode_path=r'mode_photo/save_result.png')
            self.returnBox.append(f"\n进程{i + 1}/{len(self.paths)}, 完成处理 {path} ...")
        # 保存所有结果
        auto_click(img_mode_path=r'mode_photo\close_window.png')
        time.sleep(0.5)
        auto_click(r'mode_photo\close.png', clicks=2, interval=1)
        time.sleep(1)
        auto_click(r'mode_photo\result_full_screen.png')    # 打开保存结果窗口
        time.sleep(1)
        auto_click(r'mode_photo\save_as.png')
        time.sleep(1)
        auto_click(r'mode_photo\write_path.png')
        time.sleep(0.5)
        pyautogui.write(os.path.split(self.output_file_path)[0], interval=0.025)
        pyautogui.press("enter")
        time.sleep(0.1)
        pyautogui.press("enter")

        auto_click(r'mode_photo\write_name.png', clicks=2, interval=1)
        time.sleep(0.5)
        pyautogui.write(os.path.split(self.output_file_path)[1], interval=0.025)
        pyautogui.press("enter")
        time.sleep(0.1)
        pyautogui.press("enter")
        auto_click(r'mode_photo\save_data.png')
        self.outputBox.append('\n全部图片计算完成')
        self.returnBox.append('\n' + '-'*55)
        self.returnBox.append('\n计算失败图片:')
        for i in unfinish:
            self.returnBox.append('\n' + f'{i}')

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = MainWindow()
    app.exec_()

    
    




