import sys
from glob import glob
from data.scripts.config import Config
from data.scripts.tray_manager import TrayManager
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMenu, QAction
from PyQt5.QtCore import QTimer, Qt, QPoint
from PyQt5.QtGui import QPixmap

class DesktopPet(QWidget):
    def __init__(self):
        super().__init__()
        self.n = 0
        self.default_count = 0
        self.is_hidden = False    # 窗口是否隐藏
        self.dragging = False    # 是否正在拖动窗口
        self.offset = QPoint(0, 0)    # 鼠标点击位置与窗口左上角的偏移量

        # 设置缩放
        self.scale = 1.0
        self.base_size = (498, 707)
        self.current_size = self.base_size

        self.default_pictures = glob(r'data\pets\loop\*.gif')
        with open(r"data\scripts\menu_style.qss", "r", encoding="utf-8") as f:
            qss = f.read()
        self.setStyleSheet(qss)
        self.initUI()

        # 系统托盘管理
        self.tray = TrayManager(self)
        self.tray.show_signal.connect(self.show_pet)
        self.tray.hide_signal.connect(self.hide_pet)

    def initUI(self):
        self.setWindowTitle('Desktop Pet')
        self.setWindowIcon(QIcon(r'data\icon\tw_icon_chieri_sd.png'))
        self.resize(498, 707)

        # 隐藏窗口边框和标题栏
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)    # 设置窗口透明

        # 添加标签用于显示图片
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, self.base_size[0], self.base_size[1])

        self.pm = QPixmap(self.default_pictures[0]).scaled(
            int(self.base_size[0] * self.scale), int(self.base_size[1] * self.scale),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label.setPixmap(self.pm)

        self.timer = QTimer(self)    # 创建定时器
        self.timer.setInterval(30)    # 50 ms
        self.timer.timeout.connect(self.update_picture)
        self.timer.start()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.offset = event.pos()    # 计算偏移量
        elif event.button() == Qt.RightButton:
            self.showContextMenu(event.pos())
    
    def mouseMoveEvent(self, event):
        if self.dragging:
            self.move(event.globalPos() - self.offset)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def showContextMenu(self, pos):
        menu = QMenu(self)

        scale_menu = menu.addMenu('改变大小')

        # 放大（+0.25）
        bigger = QAction('放大', self)
        bigger.triggered.connect(lambda: self.set_scale(self.scale + 0.25))
        scale_menu.addAction(bigger)

        # 缩小（-0.25）
        smaller = QAction('缩小', self)
        smaller.triggered.connect(lambda: self.set_scale(self.scale - 0.25))
        scale_menu.addAction(smaller)

        # 还原（1.0）
        reset = QAction('还原', self)
        reset.triggered.connect(lambda: self.set_scale(1.0))
        scale_menu.addAction(reset)

        # 切换图片选项
        switch_action = QAction('切换动作', self)
        menu.addAction(switch_action)

        # 创建子菜单
        sub_menu = QMenu('选择动作', self)
        switch_action.setMenu(sub_menu)    # 将子菜单关联到切换动作选项

        # 子菜单内容
        for action_name, (path, interval) in Config.switch_action_dict.items():
            action = QAction(action_name, self)

            def make_trigger(path=path, inter=interval):
                def trigger():
                    pics = Config.get_pictures(path)
                    self.default_pictures = pics
                    self.timer.setInterval(inter)
                    self.n = -1
                    self.timer.start()
                return trigger

            action.triggered.connect(make_trigger())
            sub_menu.addAction(action)

        hide_action = QAction('躲猫猫', self)
        hide_action.triggered.connect(self.hide_pet)
        menu.addAction(hide_action)

        # 退出选项
        quit_action = QAction('退出', self)
        quit_action.triggered.connect(QApplication.quit)
        menu.addAction(quit_action)

        menu.adjustSize()  # 必须先获取尺寸
        window_pos = self.mapToGlobal(QPoint(0, 0))
        screen = QApplication.primaryScreen().availableGeometry()

        # 右侧弹出，10px 间距
        menu_x = window_pos.x() + int((self.width() // 2) * 1.8)
        menu_y = window_pos.y()+ (self.height() - menu.height()) // 2

        # 防越界：右边 + 上下
        menu_x = min(menu_x, screen.width() - menu.width())
        menu_y = max(screen.y(), min(menu_y, screen.height() - menu.height()))

        menu.popup(QPoint(menu_x, menu_y))

    def random_action(self):
        import random
        action_name = random.choice(list(Config.switch_action_dict.keys()))
        path, interval = Config.switch_action_dict[action_name]
        pics = Config.get_pictures(path)
        self.default_pictures = pics
        self.timer.setInterval(interval)
        self.n = -1
        self.timer.start()

    def hide_pet(self):
        if not self.is_hidden:
            self.hide()
            self.is_hidden = True

    def show_pet(self):
        if self.is_hidden:
            self.show()
            self.is_hidden = False
            self.timer.start()
            # 恢复默认动作
            self.default_pictures = Config.get_pictures(r'data\pets\loop\*.gif')
            self.n = -1
            self.default_count = 0

    def update_geometry(self):
        '''更新窗口和标签的几何尺寸'''
        new_width = int(self.base_size[0] * self.scale)
        new_height = int(self.base_size[1] * self.scale)
        self.current_size = (new_width, new_height)
        self.resize(new_width, new_height)
        self.label.setGeometry(0, 0, new_width, new_height)

    def set_scale(self, factor):
        '''设置缩放比例'''
        if factor == self.scale:
            return
        
        if factor < 0.25 or factor > 2.0:
            return
        
        self.scale = factor
        self.update_geometry()

        if self.default_pictures:
            current_frame = self.default_pictures[self.n % len(self.default_pictures)]
            pm = QPixmap(current_frame).scaled(
                self.current_size[0], self.current_size[1],
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.label.setPixmap(pm)

    def update_picture(self):
        if self.is_hidden:
            return
        
        pics = self.default_pictures
        if not pics:
            return
        
        self.n = self.n + 1 if self.n < len(pics) - 1 else 0
        current_frame = pics[self.n]

        pm = QPixmap(current_frame).scaled(
            self.current_size[0], self.current_size[1],
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label.setPixmap(pm)

        if not self.is_hidden and self.n == len(pics) - 1:
            self.default_count += 1
            self.default_pictures = Config.get_pictures(r'data\pets\loop\*.gif')
            if self.default_count >= 6:    # 连续播放 6 次默认动作后切换随机动作
                self.random_action()
                self.default_count = 0

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pet = DesktopPet()
    pet.show()
    sys.exit(app.exec_())




















