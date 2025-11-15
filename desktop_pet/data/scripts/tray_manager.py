from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QSystemTrayIcon, QMenu as TrayMenu, QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QObject, pyqtSignal


class TrayManager(QObject):
    show_signal = pyqtSignal()
    hide_signal = pyqtSignal()

    def __init__(self, parent=None, icon_path=r"data\icon\tw_icon_chieri_sd.png"):
        super().__init__(parent)
        self.parent = parent
        self.icon_path = icon_path
        self.tray = QSystemTrayIcon(parent)    # 创建系统托盘图标
        self.setup_tray()    # 设置托盘图标和菜单

    def setup_tray(self):
        self.tray.setIcon(QIcon(self.icon_path))

        menu = TrayMenu()
        show_act = QAction("显示", self)
        show_act.triggered.connect(self.show_signal.emit)
        hide_act = QAction("隐藏", self)
        hide_act.triggered.connect(self.hide_signal.emit)
        quit_act = QAction("退出", self)
        quit_act.triggered.connect(QApplication.quit)

        menu.addAction(show_act)
        menu.addAction(hide_act)
        menu.addSeparator()
        menu.addAction(quit_act)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self.on_activated)
        self.tray.show()

    def on_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            if self.parent.isHidden():
                self.show_signal.emit()
            else:
                self.hide_signal.emit()

    def show_message(self, title, msg, icon=1, msecs=2000):
        self.tray.showMessage(title, msg, icon, msecs)
