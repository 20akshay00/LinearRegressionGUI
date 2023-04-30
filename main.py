import sys
import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib import rc

# pip install pyqt5

from PyQt5.QtWidgets import QApplication, QSizePolicy, QWidget, QMainWindow, QMenu, QHBoxLayout, QTableView
from PyQt5.QtCore import Qt, QSize, QAbstractTableModel

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
        self.axes.plot(xreg, self.m * xreg + self.c, zorder = 2, lw = 1)
        self.axes.scatter(self.x, self.y, c = self.COLS[mask], s = self.SIZE[mask], zorder = 3)

        self.axes.set_ylim(0, 1)
        self.axes.set_xlim(0, 1)

        self.axes.xaxis.set_minor_locator(MultipleLocator(0.05))
        self.axes.yaxis.set_minor_locator(MultipleLocator(0.05))

        self.axes.grid(True, zorder = 0, which='minor', alpha = 0.2)
        self.axes.grid(True, zorder = 0, which='major')

        self.draw()

    def update_line(self):
        if len(self.x) > 1:
            xmean, ymean = np.mean(self.x), np.mean(self.y)
            self.m =  np.sum((self.x - xmean) * (self.y - ymean))/np.sum((self.x - xmean) ** 2)
            self.c = ymean - self.m * xmean

class FCanvas(MyMplCanvas):
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
        self.axes.plot(xreg, self.m * xreg + self.c, zorder = 2, lw = 1)
        self.axes.scatter(self.x, self.y, c = self.COLS[mask], s = self.SIZE[mask], zorder = 3)

        self.axes.set_ylim(0, 1)
        self.axes.set_xlim(0, 1)

        self.axes.xaxis.set_minor_locator(MultipleLocator(0.05))
        self.axes.yaxis.set_minor_locator(MultipleLocator(0.05))

        self.axes.grid(True, zorder = 0, which='minor', alpha = 0.2)
        self.axes.grid(True, zorder = 0, which='major')

        self.draw()


class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list
            return self._data[index.row()][index.column()]

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self._data[0])

class ApplicationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setStyleSheet('font-size: 35px;')
        self.file_menu = QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.close, Qt.CTRL + Qt.Key_Q)
        # self.menuBar().addMenu(self.file_menu)

        self.main_widget = QWidget()
        self.setFixedSize(QSize(1000, 400))

        layout = QHBoxLayout(self.main_widget)

        line_plot = LineCanvas(self.main_widget)
        f_plot = FCanvas(self.main_widget)

        table = QTableView()
        data = [[4, 9, 2], [1, 0, 0], [3, 5, 0], [3, 3, 2], [7, 8, 9]]
        model = TableModel(data)
        table.setModel(model)

        # layout.addWidget(table)
        layout.addWidget(line_plot)
        layout.addWidget(f_plot)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ApplicationWindow()
    win.setWindowTitle("PyQt5 Matplotlib App Demo")
    win.show()
    sys.exit(app.exec_())