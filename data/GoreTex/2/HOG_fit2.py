import autograd.numpy as np
import math
from autograd import jacobian
from numpy import genfromtxt
import scipy
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from scipy.optimize import Bounds
from tkinter import messagebox

opt_params_global = None
file_names_global = []
selected_models_global = []
fixed_params_global = {}


def evalP(par, lam1, lam2, model='GOH', fixed_params={}):
    mu = par[0]
    k1 = par[1]
    k2 = par[2]

    if 'kappa' in fixed_params:
        kappa = fixed_params['kappa']
    else:
        kappa = par[3]

    if 'alpha' in fixed_params:
        alpha = fixed_params['alpha']
    else:
        if 'kappa' in fixed_params:
            alpha = par[3]
        else:
            alpha = par[4]

    a0 = np.array([np.cos(alpha), np.sin(alpha), 0])
    M = np.outer(a0, a0)

    lam3 = 1 / (lam1 * lam2)
    F = np.array([[lam1, 0., 0], [0., lam2, 0], [0., 0, lam3]])
    C = F.T @ F
    invF = np.linalg.inv(F)
    invC = np.linalg.inv(C)
    I = np.eye(3)

    I1 = np.trace(C)
    I4 = np.tensordot(C, M)

    if model == 'GOH':
        H = kappa * I1 + (1 - 3 * kappa) * I4
        E = H - 1
        exp_term = np.exp(k2 * E ** 2)
        S2 = mu * I + 2 * k1 * exp_term * E * (kappa * I + (1 - 3 * kappa) * M)
    elif model == 'HOG':
        E = (1 - kappa) * (I1 - 3)**2 + kappa * (I4 - 1)**2
        exp_term = np.exp(k2 * E)
        S2 = mu * I + 2 * k1 * exp_term * (
            2 * (1 - kappa) * (I1 - 3) * I + 2 * kappa * (I4 - 1) * M)
    else:
        raise ValueError('Unknown model type')

    p = S2[2, 2] / invC[2, 2]
    S = -p * invC + S2

    P = F @ S
    return P

def load_multiple_csv_files():
    filepaths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
    if filepaths:
        data_list = []
        file_names = []
        for filepath in filepaths:
            data = np.genfromtxt(filepath, delimiter=',')
            file_name = filepath.split('/')[-1]
            file_names.append(file_name)

            if data.ndim == 1:
                print(f"Warning: Data from {file_name} is 1D. Reshaping to 2D.")
                data = data.reshape(-1, 4)
            data_list.append(data)

        return data_list, file_names
    else:
        messagebox.showwarning("Warning", "No files selected")
        return None, []

# RMSE
def calculate_rmse(real_values, predicted_values):
    return np.sqrt(np.mean((real_values - predicted_values) ** 2))

# Fit
def fit_parameters(data, model='GOH', fixed_params={}):
    def ObjA(par):
        err = 0.0
        lam1 = data[:, 0]
        lam2 = data[:, 2]
        PE1 = data[:, 1]
        PE2 = data[:, 3]
        for i in range(len(lam1)):
            P = evalP(par, lam1[i], lam2[i], model=model, fixed_params=fixed_params)
            err += (P[0, 0] - PE1[i]) ** 2 + (P[1, 1] - PE2[i]) ** 2
        return err / len(lam1)

    jacA = jacobian(ObjA)

    initial_par = []
    bounds_lower = []
    bounds_upper = []

    initial_par.extend([0.0001, 1, 1])
    bounds_lower.extend([0.0, 0.0, 0.0])
    bounds_upper.extend([1e-2, 10., 1000.])

    if 'kappa' not in fixed_params:
        initial_par.append(1 / 10)
        bounds_lower.append(0.)
        bounds_upper.append(1 / 3)

    if 'alpha' not in fixed_params:
        initial_par.append(np.pi / 4)
        bounds_lower.append(0)
        bounds_upper.append(np.pi)

    bounds = Bounds(bounds_lower, bounds_upper)

    optA = scipy.optimize.minimize(ObjA, initial_par, jac=jacA, bounds=bounds)

    opt_par = []
    idx = 0  # Индекс для optA.x
    opt_par.extend(optA.x[:3])  # mu, k1, k2

    if 'kappa' in fixed_params:
        opt_par.append(fixed_params['kappa'])
    else:
        opt_par.append(optA.x[idx + 3])
        idx += 1

    if 'alpha' in fixed_params:
        opt_par.append(fixed_params['alpha'])
    else:
        opt_par.append(optA.x[idx + 3])

    lam1 = data[:, 0]
    lam2 = data[:, 2]
    PE1 = data[:, 1]
    PE2 = data[:, 3]
    predicted_values = np.zeros((len(lam1), 2))

    for i in range(len(lam1)):
        P = evalP(opt_par, lam1[i], lam2[i], model=model, fixed_params={})
        predicted_values[i, 0] = P[0, 0]
        predicted_values[i, 1] = P[1, 1]

    rmse1 = calculate_rmse(PE1, predicted_values[:, 0])
    rmse2 = calculate_rmse(PE2, predicted_values[:, 1])

    print(f"Optimized Parameters for {model} model: {opt_par}")
    print(f"RMSE for e1: {rmse1}")
    print(f"RMSE for e2: {rmse2}")

    return opt_par, rmse1, rmse2

def plot_results(data_list, opt_params_list, file_names, model='GOH'):
    for i, (data, opt_params, file_name) in enumerate(zip(data_list, opt_params_list, file_names)):
        lam1 = data[:, 0]
        lam2 = data[:, 2]
        PE1 = data[:, 1]
        PE2 = data[:, 3]
        Parr = np.zeros([len(lam1), 2])

        for j in range(len(lam1)):
            P = evalP(opt_params, lam1[j], lam2[j], model=model)
            Parr[j, 0] = P[0, 0]
            Parr[j, 1] = P[1, 1]

        rmse1 = calculate_rmse(PE1, Parr[:, 0])
        rmse2 = calculate_rmse(PE2, Parr[:, 1])

        plt.figure(figsize=(10, 5))
        plt.plot(lam1, PE1, 'ro', label=f'exp P11 ({file_name})')
        plt.plot(lam1, Parr[:, 0], 'r--', label=f'mod P11 ({model}, RMSE: {rmse1:.4f})')

        plt.plot(lam2, PE2, 'bo', label=f'exp P11 ({file_name})')
        plt.plot(lam2, Parr[:, 1], 'b--', label=f'mod P22 ({model}, RMSE: {rmse2:.4f})')

        plt.title(f"Fit Result for {file_name} using {model} model")
        plt.xlabel('λ')
        plt.ylabel('1st PK [MPa]')
        plt.legend()

        plt.tight_layout()
        plt.show()

        print(f"Metrics for {file_name} using {model} model:")
        print(f"  RMSE for P11: {rmse1:.4f}")
        print(f"  RMSE for P22: {rmse2:.4f}")

def evaluate_on_test_data(test_data, opt_params, model='GOH'):
    test_data = np.array(test_data)

    if test_data.ndim != 2:
        print("Error: Test data should be a 2D array.")
        return

    lam1 = test_data[:, 0]
    lam2 = test_data[:, 2]
    PE1 = test_data[:, 1]
    PE2 = test_data[:, 3]
    predicted_values = np.zeros((len(lam1), 2))

    for i in range(len(lam1)):
        P = evalP(opt_params, lam1[i], lam2[i], model=model)
        predicted_values[i, 0] = P[0, 0]
        predicted_values[i, 1] = P[1, 1]

    rmse1 = calculate_rmse(PE1, predicted_values[:, 0])
    rmse2 = calculate_rmse(PE2, predicted_values[:, 1])

    print(f"Test RMSE for e1 ({model} model): {rmse1}")
    print(f"Test RMSE for e2 ({model} model): {rmse2}")

    plot_test_results(test_data, predicted_values, rmse1, rmse2, model=model)

def plot_test_results(test_data, predicted_values, rmse1, rmse2, model='GOH'):
    lam1 = test_data[:, 0]
    lam2 = test_data[:, 2]
    PE1 = test_data[:, 1]
    PE2 = test_data[:, 3]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(lam1, PE1, 'ro', label='exp P11')
    ax[0].plot(lam1, predicted_values[:, 0], 'r-', label=f'{model} P11 (RMSE: {rmse1:.4f})')
    ax[0].set_title(f'{model} P11')
    ax[0].set_xlabel("λ")
    ax[0].set_ylabel("1st PK [MPa]")
    ax[0].legend()

    ax[1].plot(lam2, PE2, 'bo', label='exp P22 ')
    ax[1].plot(lam2, predicted_values[:, 1], 'b-', label=f'{model} P22 (RMSE: {rmse2:.4f})')
    ax[1].set_title(f'{model} P22')
    ax[1].set_xlabel("λ")
    ax[1].set_ylabel("1st PK [MPa]")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

def main():
    root = tk.Tk()
    root.title("Experiment Data Fitting")

    goh_var = tk.BooleanVar()
    hog_var = tk.BooleanVar()

    kappa_var = tk.StringVar()
    alpha_var = tk.StringVar()

    goh_checkbox = tk.Checkbutton(root, text='GOH Model', variable=goh_var)
    hog_checkbox = tk.Checkbutton(root, text='HOG Model', variable=hog_var)
    goh_checkbox.pack()
    hog_checkbox.pack()

    kappa_label = tk.Label(root, text='kappa (leave blank to optimize):')
    kappa_label.pack()
    kappa_entry = tk.Entry(root, textvariable=kappa_var)
    kappa_entry.pack()

    alpha_label = tk.Label(root, text='alpha (radians, leave blank to optimize):')
    alpha_label.pack()
    alpha_entry = tk.Entry(root, textvariable=alpha_var)
    alpha_entry.pack()

    load_button = tk.Button(root, text="Load Multiple CSV Data", command=lambda: load_data_and_fit(goh_var.get(), hog_var.get(), kappa_var.get(), alpha_var.get()))
    load_button.pack()

    load_test_button = tk.Button(root, text="Load Test Data", command=lambda: load_test_data_and_evaluate())
    load_test_button.pack()

    root.mainloop()

def load_data_and_fit(use_goh, use_hog, kappa_value, alpha_value):
    global opt_params_global, file_names_global, selected_models_global, fixed_params_global  # Declare global variables
    data_list, file_names = load_multiple_csv_files()  # Load all data

    selected_models = []
    if use_goh:
        selected_models.append('GOH')
    if use_hog:
        selected_models.append('HOG')

    if not selected_models:
        messagebox.showwarning("Warning", "No models selected.")
        return

    selected_models_global = selected_models

    fixed_params = {}
    if kappa_value != '':
        try:
            fixed_params['kappa'] = float(kappa_value)
            print(f"kappa is fixed at {fixed_params['kappa']}")
        except ValueError:
            messagebox.showerror("Error", "Invalid value for kappa.")
            return

    if alpha_value != '':
        try:
            fixed_params['alpha'] = float(alpha_value)
            print(f"alpha is fixed at {fixed_params['alpha']}")
        except ValueError:
            messagebox.showerror("Error", "Invalid value for alpha.")
            return

    fixed_params_global = fixed_params

    if data_list is not None and len(data_list) > 0:
        combined_data = np.vstack(data_list)

        opt_params_global = {}
        rmse1_global = {}
        rmse2_global = {}

        for model in selected_models:
            opt_params, rmse1, rmse2 = fit_parameters(combined_data, model=model, fixed_params=fixed_params)
            opt_params_global[model] = opt_params
            rmse1_global[model] = rmse1
            rmse2_global[model] = rmse2

        file_names_global = file_names

        for model in selected_models:
            plot_results(data_list, [opt_params_global[model]] * len(data_list), file_names, model=model)
    else:
        messagebox.showwarning("Warning", "No data loaded.")

def load_test_data_and_evaluate():
    global opt_params_global, selected_models_global, fixed_params_global

    test_data_list, test_file_names = load_multiple_csv_files()

    if test_data_list is not None and opt_params_global is not None:
        for test_data, test_file_name in zip(test_data_list, test_file_names):
            if test_data.ndim != 2:
                print(f"Error: Test data in file {test_file_name} is not 2D.")
            else:
                for model in selected_models_global:
                    opt_params = opt_params_global[model]
                    print(f"Evaluating test data {test_file_name} using {model} model")
                    evaluate_on_test_data(test_data, opt_params, model=model)
    else:
        messagebox.showwarning("Warning", "Test data not loaded or model not trained.")

if __name__ == "__main__":
    main()
