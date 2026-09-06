from src.synthetic_vienna.temperature_model import TemperatureModel
import numpy as np
from scripts.load_vienna_data import load_vienna_subset

vienna_subset = load_vienna_subset()
y = vienna_subset['daily_max_2m_temperature'].values

model = TemperatureModel()
model.fit(y, with_autocorrelation=True)

for param in model.model_params:
    print(f"{param:.3f}")

y_hat = model.generate(std=0)
residuals = y - y_hat
variance = np.var(residuals)
std = np.sqrt(variance)
amplitude = np.sqrt(model.model_params.a**2 + model.model_params.b**2)

# print(f"beta0: {beta0:.3f}")
# print(f"Trend (beta1): {beta1:.3f} °C/year")
# print(f"a: {a:.3f}")
# print(f"b: {b:.3f}")

# # print(f"Trend: {10 * beta1:.3f} °C/decade")
# print(f"Seasonal amplitude: {amplitude:.2f} °C")
# print("Noise variance:", variance)
# print("Noise std:", std)

"""
For vienna = Coord(lat=48.2082, lon=16.3738)

Trend (beta1): 0.027 °C/year
a: -2.896
b: -11.591
Seasonal amplitude: 11.95 °C
Noise variance: 20.834055382995427
Noise std: 4.564433741768569

ar_innovation_std 2.9136834469177537
ar_rho for vienna: 0.8289066046135836
"""

