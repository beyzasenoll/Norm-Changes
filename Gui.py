import os
import json
import tkinter as tk
from tkinter import messagebox, ttk
import subprocess
import sys

# --- Project paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(PROJECT_ROOT, 'config.json')
MAIN_FILE = os.path.join(PROJECT_ROOT, 'main.py')

# --- Static field definitions ---
# Each tuple: (key, label text, default, type, [optional choices])
FIELDS = [
    ('num_agents', 'Number of Agents', 100, int),
    ('num_steps', 'Number of Steps', 1000, int),
    ('topology_type', 'Topology Type', 'small_world', str, ['small_world', 'ring', 'random']),
    ('k', 'k (neighbors)', 4, int),
    ('p', 'p (rewire prob)', 0.2, float),
    ('beta', 'beta (learning rate)', 0.3, float),
    ('circle_degree', 'Circle Degrees (comma sep)', [1, 2, 3], list),
    ('trendsetter_percent', 'Trendsetter %', 10, int),
    ('epsilon', 'Epsilon', 0.2, float),
    ('weights', 'Weights (comma sep)', [0, 0, 1], list),
    ('distance_type', 'Distance Type', 'close', str, ['close', 'far']),
]

# --- Config load/save ---
def load_config():
    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
            # fill missing
            for key, label, default, typ, *rest in FIELDS:
                if key not in cfg:
                    cfg[key] = default
            return cfg
        except Exception:
            messagebox.showwarning('Config Error', 'config.json okunamadı. Varsayılan ayarlar yüklendi.')
    # write defaults
    cfg = {key: default for key, _, default, *_ in FIELDS}
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=4)
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(cfg, f, indent=4)
        messagebox.showinfo('Saved', f'Configuration saved to {CONFIG_FILE}')
    except Exception as e:
        messagebox.showerror('Error', f'Could not save config: {e}')

# --- Run simulation ---
def run_simulation():
    cfg = get_current_config()
    save_config(cfg)
    try:
        subprocess.Popen([sys.executable, MAIN_FILE], cwd=PROJECT_ROOT)
    except Exception as e:
        messagebox.showerror('Error', f'Could not start main.py: {e}')

# --- Build GUI ---
root = tk.Tk()
root.title('Norm-Changes Simulator Configuration')

config = load_config()
vars_map = {}

for idx, field in enumerate(FIELDS):
    key, label_text, default, typ, *rest = field
    value = config.get(key, default)
    # prepare initial text
    if typ is list:
        init = ','.join(map(str, value))
    else:
        init = str(value)

    # label
    lbl = tk.Label(root, text=f'{label_text}:', anchor='e', width=20)
    lbl.grid(row=idx, column=0, padx=5, pady=4, sticky='e')

    # widget
    if typ is str and rest:
        # choices provided
        choices = rest[0]
        var = tk.StringVar(value=init)
        widget = ttk.Combobox(root, textvariable=var, values=choices, state='readonly')
    else:
        var = tk.StringVar(value=init)
        widget = tk.Entry(root, textvariable=var, width=30)

    widget.grid(row=idx, column=1, padx=5, pady=4, sticky='w')
    vars_map[key] = (var, typ)

# buttons
btn_save = tk.Button(root, text='Save Configuration', command=lambda: save_config(get_current_config()), width=20)
btn_run  = tk.Button(root, text='Run Simulation',   command=run_simulation, width=20)
row = len(FIELDS)
btn_save.grid(row=row, column=0, pady=10)
btn_run.grid(row=row, column=1, pady=10)

# --- Read GUI values to config dict ---
def get_current_config():
    new_cfg = {}
    for key, (var, typ) in vars_map.items():
        text = var.get().strip()
        try:
            if typ is int:
                new_cfg[key] = int(text)
            elif typ is float:
                new_cfg[key] = float(text)
            elif typ is list:
                parts = [x.strip() for x in text.split(',') if x.strip()]
                # infer list item type from default
                default_list = next(d for k, _, d, t, *rest in FIELDS if k == key)
                if default_list and isinstance(default_list[0], int):
                    new_cfg[key] = [int(x) for x in parts]
                elif default_list and isinstance(default_list[0], float):
                    new_cfg[key] = [float(x) for x in parts]
                else:
                    new_cfg[key] = parts
            else:
                new_cfg[key] = text
        except Exception:
            messagebox.showerror('Error', f'Invalid value for {key}: {text}')
            return config
    return new_cfg

root.mainloop()