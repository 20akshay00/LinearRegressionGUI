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

from PyQt5.QtWidgets import QApplication, QSizePolicy, QWidget, QMainWindow, QMenu, QHBoxLayout, QTableView, QSlider, QVBoxLayout, QPushButton, QFormLayout, QLabel, QTableWidget, QTableWidgetItem, QFrame, QAbstractItemView, QHeaderView
from PyQt5.QtCore import Qt, QSize, QAbstractTableModel
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5 import QtCore

plt.style.use('seaborn-whitegrid')
plt.minorticks_on()

plt.rc('font', family='serif',size=5)
plt.rcParams["legend.frameon"] = True

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
    intercept_changed = QtCore.pyqtSignal(float)
    slope_changed = QtCore.pyqtSignal(float)
    hypothesis_changed = QtCore.pyqtSignal(float, float, float, float)
    pearson_changed = QtCore.pyqtSignal(float)

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
            
            self.regress_line()

    def update_figure(self):
        mask = np.zeros(len(self.x), dtype = int)
        if not (self.focused is None) and len(self.x) > 0:
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
        self.axes.legend(loc='upper right', edgecolor="black", facecolor = "black", framealpha = 0.1)

        self.axes.grid(True, zorder = 0, which='minor', alpha = 0.2)
        self.axes.grid(True, zorder = 0, which='major')

        self.draw()

    def hypothesis_test(self):
        if len(self.x) > 2:
            xmean, ymean = np.mean(self.x), np.mean(self.y)

            # hypothesis test
            yreg = self.m * self.x + self.c

            SSres, SSreg = np.sum((self.y - yreg) ** 2), np.sum((yreg - ymean) ** 2)
            dfres, dfreg = len(self.x) - 2, 1

            Fval = (SSreg/dfreg)/(SSres/dfres)

            pearson = np.sum((self.x - xmean) * (self.y - ymean))/np.sqrt(np.sum((self.x - xmean)**2) * np.sum((self.y - ymean)**2))

            self.line_changed.emit(dfres, Fval)
            self.hypothesis_changed.emit(SSres, SSreg, dfres, dfreg)
            self.pearson_changed.emit(np.round(pearson, 4))

    def regress_line(self):
        xmean, ymean = np.mean(self.x), np.mean(self.y)
        self.m =  np.sum((self.x - xmean) * (self.y - ymean))/np.sum((self.x - xmean) ** 2)
        self.c = ymean - self.m * xmean

        self.slope_changed.emit(self.m)
        self.intercept_changed.emit(self.c)

        self.hypothesis_test()
        self.update_figure()

    def update_slope(self, m, lc = 10):
        self.m = m/lc
        self.hypothesis_test()
        self.update_figure()

    def update_intercept(self, c, lc = 100):
        self.c = c/lc 
        self.hypothesis_test()
        self.update_figure()

    def clear(self):
        self.x, self.y = np.array([]), np.array([])
        self.m, self.c = 0., 0.
        self.slope_changed.emit(self.m)
        self.intercept_changed.emit(self.c)
        self.line_changed.emit(0, 1.)
        self.hypothesis_changed.emit(np.nan, np.nan, np.nan, np.nan)
        self.pearson_changed.emit(0.)

        self.update_figure()

class FCanvas(MyMplCanvas):
    result_changed = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None, width=10, height=7, dpi=200):
        super().__init__(parent, width, height, dpi)

        self.COLS = np.array(["black", "red"])
        self.SIZE = np.array([4, 8])

        self.alpha = 0.05
        self.dfn, self.dfd = 1, 0
        self.Fval = 1.

        self.update_figure()

    @QtCore.pyqtSlot(int, float)
    def update_vals(self, dfd, Fval):
        self.dfd, self.Fval = dfd, Fval
        self.update_figure()

    def update_figure(self):
        self.axes.cla()
        
        if self.dfd > 0:
            Fcrit = stats.f.ppf(q = 1 - self.alpha, dfn = self.dfn, dfd = self.dfd)

            xmax = np.max([self.Fval * 1.1, 5., Fcrit * 1.1])
            x = np.linspace(0., xmax, 100)
            
            rv = stats.f(self.dfn, self.dfd)
            self.axes.plot(x, rv.pdf(x), 'k-', label = "PDF")
            self.axes.axvline(self.Fval, lw = 1, label = "F-observed")
            self.axes.axvline(Fcrit, lw = 1, ls = "dashed", label = "F-critical")

            self.axes.set_ylim(-0.1, 1)
            self.axes.xaxis.set_minor_locator(MultipleLocator(xmax/20))
            self.axes.yaxis.set_minor_locator(MultipleLocator(0.05))

            self.result_changed.emit(self.Fval > Fcrit)

        self.axes.set_title("Hypothesis testing")
        self.axes.set_xlabel("F-value")
        self.axes.set_ylabel("Probability density function")
        self.axes.legend(loc='upper right', edgecolor="black", facecolor = "black", framealpha = 0.1)

        self.axes.grid(True, zorder = 0, which='minor', alpha = 0.2)
        self.axes.grid(True, zorder = 0, which='major')

        self.draw()

    def update_alpha(self, val):
        self.alpha = val/100
        self.update_figure()

# class TableView(QTableWidget):
#     def __init__(self, data, *args):
#         QTableWidget.__init__(self, *args)
#         self.data = data
#         self.setData()
#         self.resizeColumnsToContents()
#         self.resizeRowsToContents()
 
#     def setData(self): 
#         horHeaders = []
#         for n, key in enumerate(sorted(self.data.keys())):
#             horHeaders.append(key)
#             for m, item in enumerate(self.data[key]):
#                 newitem = QTableWidgetItem(item)
#                 self.setItem(m, n, newitem)

#         self.setHorizontalHeaderLabels(horHeaders)

class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data = [[]], headers=[], parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent )
        self.__data = data
        self.__headers = headers

    def rowCount(self, parent):
        return len(self.__data)

    def columnCount(self, parent):
        return len(self.__data[0])

    def flags(self, index):
        return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def data(self, index, role):

        if role == QtCore.Qt.EditRole:
            row = index.row( )
            column = index.column( )
            return self.__data[row][column]

        if role == QtCore.Qt.ToolTipRole:
            row = index.row( )
            column = index.column( )
            return self.__data[row][column]

        if role == QtCore.Qt.DisplayRole:
            row = index.row( )
            column = index.column( )
            return self.__data[row][column]

    def setData(self, index, value, role = QtCore.Qt.EditRole):
        if role ==QtCore.Qt.EditRole:

            row = index.row()
            column = index.column()
            self.dataChanged.emit(index, index)
            return self.__data[row][column]

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return ["SS", "df", "MS"][section]
            if orientation == QtCore.Qt.Vertical:
                return ["Regression", "Residual", "Total"][section]

class ApplicationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.setStyleSheet('font-size: 35px;')
        self.file_menu = QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.close, Qt.CTRL + Qt.Key_Q)
        # self.menuBar().addMenu(self.file_menu)

        self.main_widget = QWidget()
        self.setFixedSize(QSize(1100, 720))

        layout = QVBoxLayout(self.main_widget)
        
        inner1 = QHBoxLayout()
        inner2 = QHBoxLayout()

        self.line_plot = LineCanvas(self.main_widget)
        self.f_plot = FCanvas(self.main_widget)

        self.mSlider = QSlider(Qt.Orientation.Horizontal); self.mSlider.setRange(-100, 100); self.mSlider.setSingleStep(1); self.mSlider.setValue(10); 
        self.mSlider.setTickPosition(QSlider.TickPosition.TicksAbove)
        self.mSliderLabel = QLabel('Slope: 1.0', self)
        self.mSliderLayout = QVBoxLayout()
        self.mSliderLayout.addWidget(self.mSlider)
        self.mSliderLayout.addWidget(self.mSliderLabel)
        self.mSliderLabel.setFont(QFont("Serif", 15))
        self.mSliderLabel.setAlignment(Qt.AlignCenter)

        self.cSlider = QSlider(Qt.Orientation.Horizontal);  self.cSlider.setRange(-100, 100); self.cSlider.setSingleStep(1); self.cSlider.setValue(0)
        self.cSlider.setTickPosition(QSlider.TickPosition.TicksAbove)
        self.cSliderLabel = QLabel('Intercept: 0.0', self)
        self.cSliderLayout = QVBoxLayout()
        self.cSliderLayout.addWidget(self.cSlider)
        self.cSliderLayout.addWidget(self.cSliderLabel)
        self.cSliderLabel.setFont(QFont("Serif", 15))
        self.cSliderLabel.setAlignment(Qt.AlignCenter)

        self.aSlider = QSlider(Qt.Orientation.Horizontal);  self.aSlider.setRange(1, 100); self.aSlider.setSingleStep(1); self.aSlider.setValue(5)
        self.aSlider.setTickPosition(QSlider.TickPosition.TicksAbove)
        self.aSliderLabel = QLabel('Alpha: 0.05', self)
        self.aSliderLayout = QVBoxLayout()
        self.aSliderLayout.addWidget(self.aSlider)
        self.aSliderLayout.addWidget(self.aSliderLabel)
        self.aSliderLabel.setFont(QFont("Serif", 15))
        self.aSliderLabel.setAlignment(Qt.AlignCenter)

        self.lsqfitButton = QPushButton(); self.lsqfitButton.setText("Regression fit"); self.lsqfitButton.setFont(QFont("Serif", 15))
        self.clearButton = QPushButton(); self.clearButton.setText("Clear data"); self.clearButton.setFont(QFont("Serif", 15))

        self.textInfoLayout = QVBoxLayout()
        self.pearsonLabel = QLabel("Pearson's coefficient: 0.", self); self.pearsonLabel.setFont(QFont("Serif", 15)); self.pearsonLabel.setAlignment(Qt.AlignCenter)
        self.resultLabel = QLabel("Result: -", self); self.resultLabel.setFont(QFont("Serif", 15)); self.resultLabel.setAlignment(Qt.AlignCenter)
        self.textInfoLayout.addWidget(self.pearsonLabel)
        self.textInfoLayout.addWidget(self.resultLabel)


        data = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
        model = TableModel(data)
        self.table = QTableView()
        self.table.setModel(model)

        header = self.table.horizontalHeader()       
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setFont(QFont("Serif", 15))

        self.infoLayout = QHBoxLayout()
        self.infoLayout.addWidget(self.table, 5)
        self.infoLayout.addLayout(self.textInfoLayout, 3)

        spacer = QFrame()
        spacer.setFrameShape(QFrame.HLine)
        spacer.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Expanding)
        spacer.setLineWidth(1.5)

        inner1.addWidget(self.line_plot)
        inner1.addWidget(self.f_plot)
        inner2.addLayout(self.mSliderLayout)
        inner2.addLayout(self.cSliderLayout)
        inner2.addLayout(self.aSliderLayout)
        inner2.addWidget(self.lsqfitButton)
        inner2.addWidget(self.clearButton)

        layout.addLayout(inner1, 4)
        layout.addLayout(inner2, 4)
        layout.addWidget(spacer)
        layout.addLayout(self.infoLayout, 1)

        self.line_plot.line_changed.connect(self.f_plot.update_vals)
        self.lsqfitButton.clicked.connect(self.line_plot.regress_line)
        self.clearButton.clicked.connect(self.line_plot.clear)

        self.aSlider.valueChanged.connect(self.update_alpha_label)
        self.aSlider.valueChanged.connect(self.f_plot.update_alpha)

        self.mSlider.valueChanged.connect(self.update_slope_label)
        self.mSlider.valueChanged.connect(self.line_plot.update_slope)

        self.cSlider.valueChanged.connect(self.update_intercept_label)
        self.cSlider.valueChanged.connect(self.line_plot.update_intercept)

        self.line_plot.slope_changed.connect(self.set_slope_slider)
        self.line_plot.intercept_changed.connect(self.set_intercept_slider)
        
        self.line_plot.hypothesis_changed.connect(self.updateTable)
        self.line_plot.pearson_changed.connect(self.set_pearson_label)
        self.f_plot.result_changed.connect(self.set_result_label)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def update_alpha_label(self, val):
        self.aSliderLabel.setText(f'Alpha: {val/100}')

    def update_slope_label(self, val):
        self.mSliderLabel.setText(f'Slope: {val/10}')

    def update_intercept_label(self, val):
        self.cSliderLabel.setText(f'Intercept: {val/100}')

    @QtCore.pyqtSlot(float)
    def set_slope_slider(self, val):
        if not np.isnan(val):    
            self.mSlider.setValue(int(np.round(val, 2) * 10))
            self.update_slope_label(np.round(val, 2) * 10)

    @QtCore.pyqtSlot(float)
    def set_intercept_slider(self, val):
        if not np.isnan(val):    
            self.cSlider.setValue(int(np.round(val, 2) * 100))
            self.update_intercept_label(np.round(val, 2) * 100)

    @QtCore.pyqtSlot(float)
    def set_pearson_label(self, val):
        self.pearsonLabel.setText(f"Pearson's coefficient: {val}")

    @QtCore.pyqtSlot(bool)
    def set_result_label(self, val):
        res = "Correlated!" if val else "Uncorrelated!"
        self.resultLabel.setText(f"Result: {res}")

    @QtCore.pyqtSlot(float, float, float, float)
    def updateTable(self, SSreg, SSres, dfreg, dfres):
        if np.isnan(SSreg):
            data = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
        else:
            data = [[SSreg, dfreg, SSreg/dfreg], [SSres, dfres, SSres/dfres], [SSreg + SSres, dfreg + dfres, SSreg/dfreg + SSres/dfres]]

        model = TableModel(data)
        self.table.setModel(model)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    win = ApplicationWindow()
    win.setWindowTitle("Simple Linear Regression")
    win.show()
    sys.exit(app.exec_())