!pip install scikit-fuzzy plotly

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import plotly.graph_objects as go

temp = ctrl.Antecedent(np.linspace(0, 100, 101), 'temp')
press = ctrl.Antecedent(np.linspace(0, 10, 11), 'press')
valve_hot = ctrl.Consequent(np.linspace(-90, 90, 181), 'valve_hot')
valve_cold = ctrl.Consequent(np.linspace(-90, 90, 181), 'valve_cold')

temp.automf(names=['cold', 'cool', 'warm', 'hot', 'very_hot'])
temp['cold'] = fuzz.trimf(temp.universe, [0, 0, 25])
temp['cool'] = fuzz.trimf(temp.universe, [10, 25, 40])
temp['warm'] = fuzz.trimf(temp.universe, [30, 50, 70])
temp['hot'] = fuzz.trimf(temp.universe, [60, 75, 90])
temp['very_hot'] = fuzz.trimf(temp.universe, [80, 100, 100])

press['weak'] = fuzz.trimf(press.universe, [0, 0, 3])
press['medium'] = fuzz.trimf(press.universe, [2, 5, 8])
press['strong'] = fuzz.trimf(press.universe, [6, 10, 10])

def define_valve(v):
    v['L_large'] = fuzz.trimf(v.universe, [-90, -90, -45])
    v['L_medium'] = fuzz.trimf(v.universe, [-60, -30, 0])
    v['L_small'] = fuzz.trimf(v.universe, [-15, 0, 15])
    v['stay'] = fuzz.trimf(v.universe, [-5, 0, 5])
    v['R_small'] = fuzz.trimf(v.universe, [0, 30, 60])
    v['R_medium'] = fuzz.trimf(v.universe, [45, 90, 90])

for v in (valve_hot, valve_cold):
    define_valve(v)

rule_set = [
    ctrl.Rule(temp['hot'] & press['strong'], [valve_hot['L_medium'], valve_cold['R_medium']]),
    ctrl.Rule(temp['hot'] & press['medium'], valve_cold['R_medium']),
    ctrl.Rule(temp['hot'] & press['weak'], valve_cold['R_small']),
    ctrl.Rule(temp['warm'] & press['strong'], [valve_hot['L_small'], valve_cold['L_small']]),
    ctrl.Rule(temp['warm'] & press['medium'], [valve_hot['L_small'], valve_cold['L_small']]),
    ctrl.Rule(temp['cool'] & press['strong'], [valve_hot['R_medium'], valve_cold['L_medium']]),
    ctrl.Rule(temp['cool'] & press['medium'], [valve_hot['R_medium'], valve_cold['L_small']]),
    ctrl.Rule(temp['cool'] & press['weak'], [valve_hot['stay'], valve_cold['stay']]),
    ctrl.Rule(temp['cold'] & press['weak'], valve_hot['R_medium']),
    ctrl.Rule(temp['cold'] & press['medium'], valve_hot['R_medium']),
    ctrl.Rule(temp['cold'] & press['strong'], [valve_hot['L_medium'], valve_cold['R_medium']]),
    ctrl.Rule(temp['warm'] & press['weak'], [valve_hot['stay'], valve_cold['stay']]),
    ctrl.Rule(temp['very_hot'] & press['strong'], [valve_hot['L_large'], valve_cold['R_medium']]),
    ctrl.Rule(temp['very_hot'] & press['medium'], [valve_hot['L_large'], valve_cold['R_small']]),
    ctrl.Rule(temp['very_hot'] & press['weak'], valve_cold['stay'])
]

controller = ctrl.ControlSystem(rule_set)
sim = ctrl.ControlSystemSimulation(controller)

sim.input['temp'] = 70
sim.input['press'] = 7
sim.compute()

print(f"Hot valve rotation: {sim.output['valve_hot']:.2f}°")
print(f"Cold valve rotation: {sim.output['valve_cold']:.2f}°")

for var, name in [(temp, "Temperature"), (press, "Pressure"), (valve_hot, "Hot Valve"), (valve_cold, "Cold Valve")]:
    plt.figure(figsize=(8, 5))
    var.view()
    plt.title(f"Membership Functions for {name}")
    plt.show()

t_range, p_range = np.meshgrid(np.arange(0, 101, 10), np.arange(0, 11, 1))
hot_surface = np.zeros_like(t_range, dtype=float)

for r in range(t_range.shape[0]):
    for c in range(t_range.shape[1]):
        model = ctrl.ControlSystemSimulation(controller)
        model.input['temp'] = t_range[r, c]
        model.input['press'] = p_range[r, c]
        try:
            model.compute()
            hot_surface[r, c] = model.output['valve_hot']
        except Exception:
            hot_surface[r, c] = np.nan

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(t_range, p_range, hot_surface, cmap='plasma', edgecolor='none')
ax.set_xlabel('Temperature')
ax.set_ylabel('Pressure')
ax.set_zlabel('Hot Valve Angle')
ax.set_title('3D Control Surface (Matplotlib)')
plt.show()

fig = go.Figure(data=[go.Surface(z=hot_surface, x=t_range, y=p_range)])
fig.update_layout(
    title='3D Control Surface for Hot Valve (Plotly)',
    scene=dict(xaxis_title='Temperature', yaxis_title='Pressure', zaxis_title='Hot Valve Angle')
)
fig.show()
