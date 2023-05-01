import sys
import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
from scipy import stats 

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib import rc

# pip install pyqt5

from PyQt5.QtWidgets import QApplication, QSizePolicy, QWidget, QMainWindow, QMenu, QHBoxLayout, QTableView
from PyQt5.QtCore import Qt, QSize, QAbstractTableModel
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5 import QtCore

plt.style.use('seaborn-whitegrid')
plt.minorticks_on()

plt.rc('font', family='serif',size=5)

class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=7, dpi=200):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.axes.plot()
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class LineCanvas(MyMplCanvas):
    line_changed = QtCore.pyqtSignal(int, float)

    def __init__(self, parent=None, width=10, height=7, dpi=200):
        super().__init__(parent, width, height, dpi)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect("motion_notify_event", self.onhover)

        self.COLS = np.array(["black", "red"])
        self.SIZE = np.array([4, 8])

        self.x, self.y = np.array([]), np.array([])
        self.m, self.c = 0., 0.

        self.focused = None
        self.update_figure()

    def onclick(self, event):
        self.update_vals(event.xdata, event.ydata)
        
    def onhover(self, event, atol = 0.02):
        x, y = event.xdata, event.ydata

        if x != None and y != None:
            inds = np.where(np.array((self.x - x)**2 + (self.y - y)**2) < atol**2)[0]
            self.focused = None if (len(inds) == 0) else inds[0]

            self.update_figure()

    def update_vals(self, x, y):
        if x != None and y != None:
            if not (self.focused is None):
                self.x = np.delete(self.x, self.focused)
                self.y = np.delete(self.y, self.focused)
                self.focused = None
            else:
                self.x = np.append(self.x, [x])
                self.y = np.append(self.y, [y])
                self.focused = len(self.x) - 1
            
            self.update_line()
            self.update_figure()

    def update_figure(self):
        mask = np.zeros(len(self.x), dtype = int)
        if not (self.focused is None):
            mask[self.focused] = 1

        self.axes.cla()

        xreg = np.linspace(0., 1., 100)
        self.axes.plot(xreg, self.m * xreg + self.c, zorder = 2, lw = 1, label = "Least square fit")
        self.axes.scatter([-1.], [-1.], c = self.COLS[0], s = self.SIZE[0],  label = "Data")
        self.axes.scatter(self.x, self.y, c = self.COLS[mask], s = self.SIZE[mask], zorder = 3)

        self.axes.set_ylim(0, 1)
        self.axes.set_xlim(0, 1)

        self.axes.xaxis.set_minor_locator(MultipleLocator(0.05))
        self.axes.yaxis.set_minor_locator(MultipleLocator(0.05))
        self.axes.set_title("Linear Regression")
        self.axes.set_xlabel("Independant variable (x)")
        self.axes.set_ylabel("Dependant variable (y)")
        self.axes.legend()

        self.axes.grid(True, zorder = 0, which='minor', alpha = 0.2)
        self.axes.grid(True, zorder = 0, which='major')

        self.draw()

    def update_line(self):
        if len(self.x) > 2:
            xmean, ymean = np.mean(self.x), np.mean(self.y)
            self.m =  np.sum((self.x - xmean) * (self.y - ymean))/np.sum((self.x - xmean) ** 2)
            self.c = ymean - self.m * xmean

            # hypothesis test
            yreg = self.m * self.x + self.c

            SSres, SSreg = np.sum((self.y - yreg) ** 2), np.sum((yreg - ymean) ** 2)
            dfres, dfreg = len(self.x) - 2, 1

            Fval = (SSreg/dfreg)/(SSres/dfres)
            self.line_changed.emit(dfres, Fval)

class FCanvas(MyMplCanvas):
    def __init__(self, parent=None, width=10, height=7, dpi=200):
        super().__init__(parent, width, height, dpi)

        self.COLS = np.array(["black", "red"])
        self.SIZE = np.array([4, 8])

        self.alpha = 0.05
        self.dfn, self.dfd = 1, 1
        self.Fval = 1.

        self.update_figure()

    @QtCore.pyqtSlot(int, float)
    def update_vals(self, dfd, Fval):
        self.dfd, self.Fval = dfd, Fval
        self.update_figure()

    def update_figure(self):
        self.axes.cla()
        
        xmax = np.max([self.Fval * 1.1, 5.])
        x = np.linspace(0., xmax, 100)
        
        rv = stats.f(self.dfn, self.dfd)
        self.axes.plot(x, rv.pdf(x), 'k-', label = "PDF")
        self.axes.axvline(self.Fval, lw = 1, label = "F-observed")
        self.axes.axvline(rv.pdf(self.alpha), lw = 1, ls = "dashed", label = "F-critical")

        self.axes.set_ylim(-0.1, 1)
        self.axes.xaxis.set_minor_locator(MultipleLocator(xmax/20))
        self.axes.yaxis.set_minor_locator(MultipleLocator(0.05))
        self.axes.set_title("Hypothesis testing")
        self.axes.set_xlabel("F-value")
        self.axes.set_ylabel("Probability density function")
        self.axes.legend()
        self.axes.grid(True, zorder = 0, which='minor', alpha = 0.2)
        self.axes.grid(True, zorder = 0, which='major')

        self.draw()

class ApplicationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setStyleSheet('font-size: 35px;')
        self.file_menu = QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.close, Qt.CTRL + Qt.Key_Q)
        # self.menuBar().addMenu(self.file_menu)

        self.main_widget = QWidget()
        self.setFixedSize(QSize(1000, 500))

        layout = QHBoxLayout(self.main_widget)

        self.line_plot = LineCanvas(self.main_widget)
        self.f_plot = FCanvas(self.main_widget)

        layout.addWidget(self.line_plot, 3)
        layout.addWidget(self.f_plot, 3)

        self.line_plot.line_changed.connect(self.f_plot.update_vals)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ApplicationWindow()
    win.setWindowTitle("Simple Linear Regression")
    win.show()
    sys.exit(app.exec_())