"""
Generate should start with a model fitted to Vienna and then overwrite:
 - the warming trend (alpha)
 - the noise variance (sigma)
 - autocorrelation
 
And then compute the exceedance counts for different:
- REF period
- AGG windows
- PERC values
- POOL (boosting window)

a) similar to the figure below, beta as a function of sigma for different alpha and REF —> this mainly shows how strongly beta depends on the combination of (alpha, sigma), and of course on REF

b) beta as a function of autocorrelation for different alpha and AGG —> this should show whether beta depends on the autocorrelation and AGG (my assumption is that beta reacts differently on autocorrelation for different values of AGG, but let’s see)

c) beta as a function of PERC for different AGG and REF —> this goes to the core of our paper and relates to the examples shown with real data

d) beta as a function of POOL for different AGG and PERC —> should show that the sensitivity to POOL is not that dramatic

"""

from src.synthetic_vienna.temperature_model import TemperatureModel
from scripts.load_vienna_data import load_vienna_subset
import numpy as np
import matplotlib.pyplot as plt

vienna_subset = load_vienna_subset()
y = vienna_subset['daily_max_2m_temperature'].values

model = TemperatureModel()
model.fit(y, with_autocorrelation=True)

print('ar_innovation_std', model.model_params.ar_innovation_std)
print('ar_rho', model.model_params.ar_rho)

y_hat = model.generate(with_autocorrelation=True)

def plot_y(data, path):
    plt.figure(figsize=(15, 5))
    plt.plot(y, label='Original Vienna Data', color='blue', alpha=0.7)
    plt.plot(y_hat, label='Generated Data (Model)', color='orange', alpha=0.7)

    plt.title('Vienna Daily Max 2m Temperature: Original vs Generated')
    plt.xlabel('Day')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    
plot_y(y_hat, 'scripts/test_temperature_model.png')

# ar_rho=0.5*(1+model.model_params.ar_rho)
# print(ar_rho)
y_hat = model.generate(with_autocorrelation=True, ar_rho=0.5)

plot_y(y_hat, 'scripts/test_temperature_model_change_ar_rho.png')