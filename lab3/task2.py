!pip install -q scikit-fuzzy plotly

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import plotly.graph_objects as go

temp = ctrl.Antecedent(np.linspace(0, 50, 51), 'temp')
delta = ctrl.Antecedent(np.linspace(-5, 5, 11), 'delta')
angle = ctrl.Consequent(np.linspace(-90, 90, 181), 'angle')

temp['very_cold'] = fuzz.trimf(temp.universe, [0, 5, 15])
temp['cold'] = fuzz.trimf(temp.universe, [10, 15, 20])
temp['normal'] = fuzz.trimf(temp.universe, [18, 22, 26])
temp['warm'] = fuzz.trimf(temp.universe, [25, 30, 35])
temp['very_warm'] = fuzz.trimf(temp.universe, [35, 45, 50])

delta['decreasing'] = fuzz.trimf(delta.universe, [-5, -2.5, 0])
delta['steady'] = fuzz.trimf(delta.universe, [-1, 0, 1])
delta['increasing'] = fuzz.trimf(delta.universe, [0, 2.5, 5])

angle['strong_left'] = fuzz.trimf(angle.universe, [-90, -70, -50])
angle['mild_left'] = fuzz.trimf(angle.universe, [-30, -10, 0])
angle['neutral'] = fuzz.trimf(angle.universe, [-5, 0, 5])
angle['mild_right'] = fuzz.trimf(angle.universe, [0, 10, 30])
angle['strong_right'] = fuzz.trimf(angle.universe, [50, 70, 90])

rule_set = [
    ctrl.Rule(temp['very_warm'] & delta['increasing'], angle['strong_left']),
    ctrl.Rule(temp['very_warm'] & delta['decreasing'], angle['mild_left']),
    ctrl.Rule(temp['very_warm'] & delta['steady'], angle['strong_left']),

    ctrl.Rule(temp['warm'] & delta['increasing'], angle['strong_left']),
    ctrl.Rule(temp['warm'] & delta['steady'], angle['mild_left']),
    ctrl.Rule(temp['warm'] & delta['decreasing'], angle['neutral']),

    ctrl.Rule(temp['normal'] & delta['increasing'], angle['mild_left']),
    ctrl.Rule(temp['normal'] & delta['steady'], angle['neutral']),
    ctrl.Rule(temp['normal'] & delta['decreasing'], angle['mild_right']),

    ctrl.Rule(temp['cold'] & delta['increasing'], angle['neutral']),
    ctrl.Rule(temp['cold'] & delta['steady'], angle['mild_right']),
    ctrl.Rule(temp['cold'] & delta['decreasing'], angle['strong_right']),

    ctrl.Rule(temp['very_cold'] & delta['increasing'], angle['mild_right']),
    ctrl.Rule(temp['very_cold'] & delta['steady'], angle['strong_right']),
    ctrl.Rule(temp['very_cold'] & delta['decreasing'], angle['strong_right']),
]

ac_system = ctrl.ControlSystem(rule_set)
ac_sim = ctrl.ControlSystemSimulation(ac_system)

T_in = 30
D_in = 1.5

ac_sim.input['temp'] = T_in
ac_sim.input['delta'] = D_in
ac_sim.compute()

print("Simulation Example:")
print(f"Input → Temperature: {T_in} °C | Change rate: {D_in} °C/min")
print(f"Output → Control angle: {ac_sim.output['angle']:.2f}°")

for variable in [temp, delta, angle]:
    plt.figure(figsize=(7, 4))
    variable.view(sim=ac_sim)
    plt.tight_layout()
    plt.show()

T_vals = np.arange(0, 51, 1)
D_vals = np.arange(-5, 6, 1)
T_grid, D_grid = np.meshgrid(T_vals, D_vals)
Z_grid = np.zeros_like(T_grid, dtype=float)

for row in range(D_grid.shape[0]):
    for col in range(T_grid.shape[1]):
        sim = ctrl.ControlSystemSimulation(ac_system)
        sim.input['temp'] = T_grid[row, col]
        sim.input['delta'] = D_grid[row, col]
        try:
            sim.compute()
            Z_grid[row, col] = sim.output['angle']
        except:
            Z_grid[row, col] = np.nan

surface_plot = go.Figure(
    data=[go.Surface(z=Z_grid, x=T_grid, y=D_grid, colorscale='RdBu', reversescale=True)]
)

surface_plot.update_layout(
    title="Fuzzy Air Conditioner Control Surface",
    scene=dict(
        xaxis_title="Temperature (°C)",
        yaxis_title="Temperature Change (°C/min)",
        zaxis_title="Control Angle (°)"
    ),
    width=900,
    height=700
)
surface_plot.show()
