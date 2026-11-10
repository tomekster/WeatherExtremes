from dataclasses import asdict
from src.synthetic_vienna.temperature_model import TemperatureModel
import numpy as np
from scripts.load_vienna_data import load_vienna_subset

vienna_subset = load_vienna_subset()
y = vienna_subset['daily_max_2m_temperature'].values

model = TemperatureModel()
model.fit(y, with_autocorrelation=True)

for name, value in asdict(model.model_params).items():
    print(f"{name}: {np.round(value, 3) if value is not None else None}")

y_hat = model.generate(std=0)
residuals = y - y_hat
variance = np.var(residuals)
std = np.sqrt(variance)
# With K > 1 harmonics the seasonal cycle is no longer a single sinusoid: report each harmonic's amplitude
# and the half peak-to-peak range of the full fitted seasonal cycle
amplitudes = model.model_params.harmonic_amplitudes
seasonal = model.X[:365, 2:] @ model.model_params.linear_coeffs()[2:]
amplitude = (seasonal.max() - seasonal.min()) / 2
print("Harmonic amplitudes:", np.round(amplitudes, 2), "°C")

# print(f"beta0: {beta0:.3f}")
# print(f"Trend (beta1): {beta1:.3f} °C/year")
# print(f"a: {a:.3f}")
# print(f"b: {b:.3f}")

# print(f"Trend: {10 * beta1:.3f} °C/decade")
print(f"Seasonal amplitude: {amplitude:.2f} °C")
print("Noise variance:", variance)
print("Noise std:", std)

for name, value in asdict(model.model_params).items():
    print(f"{name}: {value}")

"""
NEW MODEL: K=3 annual harmonics, AR(1) residuals, time in no-leap years (days / 365), data 1961-2020:

For vienna = Coord(lat=48.2082, lon=16.3738)

intercept: 12.979997580120292
Trend: 0.03565184363520805 °C/year (0.357 °C/decade)
sin_coeffs: [-2.57209822  0.54580331 -0.22939042]
cos_coeffs: [-11.60846955  -0.8860334    0.1174735 ]

Harmonic amplitudes (k=1..3): [11.89  1.04  0.26] °C
Seasonal amplitude (half peak-to-peak of the fitted cycle): 12.09 °C
Noise variance: 20.364367955593575
Noise std: 4.512689658684007

ar_coeffs: [0.76251422]   (true AR(1) fit; the old ar_rho 0.829 was the lag-1 coefficient of an AR(2) fit)
innovation_std: 2.9196876410539185


OLD MODEL (1 harmonic, AR(2) fit of which only lag-1 was used, time in 365.2425-day years):

USING OLD VALUES OF 60-89 and 90-2019:

For vienna = Coord(lat=48.2082, lon=16.3738)

Trend (beta1): 0.027 °C/year
a: -2.896
b: -11.591

Seasonal amplitude: 11.95 °C
Noise variance: 20.834055382995427
Noise std: 4.564433741768569

ar_innovation_std 2.9136834469177537
ar_rho for vienna: 0.8289066046135836

AFTER UPDATING TO 61-90 and 91-2020:

a: -1.1205430618111016
b: -11.796190887241833

Seasonal amplitude: 11.85 °C
Noise variance: 21.41291532446
Noise std: 4.6274091373532125

ar_innovation_std: 2.9136834469177537
ar_rho: 0.8289066046135836


# beta0: 12.98622599886928
# beta1: 0.03404644274344148
"""