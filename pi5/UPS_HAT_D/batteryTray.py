#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import smbus
import time
import logging
import signal
import INA219
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction, QMessageBox
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer

# Setup logging
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

# Configure signal handling
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Attempt to initialize the INA219 device with error handling
try:
    ina = INA219.INA219(addr=0x43)
except Exception as e:
    logging.error("Failed to initialize INA219 device: %s", e)
    sys.exit(1)

class Worker(QObject):
    trayMessage = pyqtSignal(float, float)

    def run(self):
        """Continuously read data from the INA219 sensor and emit the results."""
        while True:
            try:
                bus_voltage = self.get_with_retry(ina.getBusVoltage_V)
                current = -int(self.get_with_retry(ina.getCurrent_mA))
                self.trayMessage.emit(bus_voltage, current)
            except Exception as e:
                logging.error("Unexpected error in Worker.run(): %s", e)
            time.sleep(1)

    def get_with_retry(self, func, max_retries=5, delay=1):
        """Retries the given function on I2C errors up to max_retries times."""
        retries = 0
        while retries < max_retries:
            try:
                return func()
            except OSError as e:
                logging.warning("I2C error: %s. Retrying %d/%d", e, retries + 1, max_retries)
                retries += 1
                time.sleep(delay)
        logging.error("I2C communication failed after %d retries.", max_retries)
        time.sleep(delay * 2)  # Extra delay before retrying again indefinitely
        return self.get_with_retry(func, max_retries, delay)  # Retry indefinitely

class MainWindow(QMessageBox):
    def __init__(self):
        super().__init__()
        self.charge = 0
        self.tray_icon = None
        self.msgBox = None
        self.about = None
        self.counter = 0

        self.setWindowTitle("Status")
        self.setText("Battery Monitor Demo")

        # Initialize system tray icon
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon("images/battery.png"))

        # Set up tray menu with actions
        show_action = QAction("Status", self)
        quit_action = QAction("Exit", self)
        about_action = QAction("About", self)
        show_action.triggered.connect(self.show)
        about_action.triggered.connect(self.show_about)
        quit_action.triggered.connect(QApplication.instance().quit)
        tray_menu = QMenu()
        tray_menu.addAction(show_action)
        tray_menu.addAction(about_action)
        tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

        # Setup worker thread
        self._thread = QThread(self)
        self._worker = Worker()
        self._worker.moveToThread(self._thread)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.started.connect(self._worker.run)
        self._worker.trayMessage.connect(self.refresh)
        self._thread.start()

        # Timer for automatic shutdown
        self._timer = QTimer(self, timeout=self.on_timeout)
        self._timer.stop()

    def on_timeout(self):
        """Handle countdown and automatic shutdown."""
        self.counter -= 1
        if self.counter > 0:
            if self.charge == 1:
                if self.msgBox:
                    self.msgBox.close()
                    self._timer.stop()
                    self.msgBox = None
            else:
                self.msgBox.setInformativeText(f"Auto shutdown after {int(self.counter)} seconds")
                self.msgBox.show()
        else:
            address = os.popen("i2cdetect -y -r 1 0x2d 0x2d | grep '2d' | awk '{print $2}'").read().strip()
            if address == '2d':
                os.system("i2cset -y 1 0x2d 0x01 0x55")
            os.system("sudo poweroff")

    def refresh(self, v, c):
        """Update battery status, UI elements, and log information."""
        try:
            self.charge = 1 if c > 50 else 0
            p = int((v - 3) / 1.2 * 100)
            p = max(0, min(p, 100))
            img = f"images/battery.{int(p / 10 + self.charge * 11)}.png"
            self.tray_icon.setIcon(QIcon(img))
            self.setIconPixmap(QPixmap(img))

            status_text = f"{p}%  {v:.1f}V  {c}mA"
            self.tray_icon.setToolTip(status_text)
            info = f"Percent: {p}%\nVoltage: {v:.1f}V\nCurrent: {c}mA"
            self.setInformativeText(info)
            
            logging.info(status_text)

            if p <= 10 and self.charge == 0:
                if self.msgBox is None:
                    self.counter = 60
                    self._timer.start(1000)
                    self.msgBox = QMessageBox(
                        QMessageBox.NoIcon, 'Battery Warning',
                        "<p><strong>The battery level is low.<br>Please connect the power adapter.</strong>"
                    )
                    self.msgBox.setIconPixmap(QPixmap("images/batteryQ.png"))
                    self.msgBox.setInformativeText("Auto shutdown after 60 seconds")
                    self.msgBox.setStandardButtons(QMessageBox.NoButton)
                    self.msgBox.exec()
        except Exception as e:
            logging.error("Error in refresh method: %s", e)

    def show_about(self):
        """Display an 'About' message box."""
        if self.about is None:
            self.about = QMessageBox(
                QMessageBox.NoIcon, 'About',
                "<p><strong>Battery Monitor Demo</strong><p>Version: v1.0<p>Display by Waveshare"
            )
            self.about.setInformativeText("<a href=\"https://www.waveshare.com\">WaveShare Official Website</a>")
            self.about.setIconPixmap(QPixmap("images/logo.png"))
            self.about.setDefaultButton(None)
            self.about.exec()
            self.about = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # Exception handling in the main window
    try:
        mw = MainWindow()
        sys.exit(app.exec())
    except Exception as e:
        logging.error("Unexpected error in main window: %s", e)
