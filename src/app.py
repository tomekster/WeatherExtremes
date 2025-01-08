import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class TemperatureViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Global Temperature Viewer")
        self.setGeometry(100, 100, 1200, 600)
        
        # Load temperature data
        self.temp_data = np.load('/home/tsternal/WeatherExtremes/daily_mean_2m_temperature_1990_1994_AGG.MEAN_aggrwindow_5_percboost_5/percentiles_0_95.npy')  # [365, 721, 1440]
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Create figures for map and plot
        self.map_figure = Figure(figsize=(8, 6))
        self.plot_figure = Figure(figsize=(4, 6))
        
        # Create canvases
        self.map_canvas = FigureCanvas(self.map_figure)
        self.plot_canvas = FigureCanvas(self.plot_figure)
        
        # Add canvases to layout
        layout.addWidget(self.map_canvas)
        layout.addWidget(self.plot_canvas)
        
        # Setup initial map
        self.setup_map()
        
        # Connect mouse events
        self.map_canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def setup_map(self):
        ax = self.map_figure.add_subplot(111)
        # Display average temperature across the year
        mean_temp = np.mean(self.temp_data, axis=0)
        im = ax.imshow(mean_temp, cmap='RdBu_r', aspect='auto')
        self.map_figure.colorbar(im, label='Temperature (°C)')
        ax.set_title('Global Temperature Map')
        self.map_figure.tight_layout()
        self.map_canvas.draw()

    def update_plot(self, lat_idx, lon_idx):
        self.plot_figure.clear()
        ax = self.plot_figure.add_subplot(111)
        
        # Get temperature data for selected location
        temp_series = self.temp_data[:, lat_idx, lon_idx]
        days = np.arange(365)
        
        ax.plot(days, temp_series)
        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Temperature at ({lat_idx}, {lon_idx})')
        self.plot_figure.tight_layout()
        self.plot_canvas.draw()

    def on_mouse_move(self, event):
        if event.inaxes:
            lat_idx = int(event.ydata)
            lon_idx = int(event.xdata)
            
            if 0 <= lat_idx < 721 and 0 <= lon_idx < 1440:
                self.update_plot(lat_idx, lon_idx)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = TemperatureViewer()
    viewer.show()
    sys.exit(app.exec())