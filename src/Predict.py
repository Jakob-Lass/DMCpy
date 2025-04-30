# SPDX-License-Identifier: MPL-2.0
import tkinter as tk
from DMCpy import DataSet, TasUBlibDEG, DataFile, Sample
import numpy as np

def predictiveTool():

    # Define the function that generates the plot
    def plot_sample(a, b, c, alpha, beta, gamma,  h1, k1, l1, h2, k2, l2, wavelength, A4Start=0, A4Stop=None):
        
        A4Start = - np.abs(A4Start)

        p1 = np.array([h1,k1,l1])
        p2 = np.array([h2,k2,l2])
        
        s = Sample.Sample(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma,
                        projectionVector1=p1, projectionVector2=p2)

        s.setProjectionVectors(p1, p2, s.planeNormal)

        df = DataFile.DataFile()
        df.sample = s
        ds = DataSet.DataSet(df)

        ax = ds.createRLUAxes()
        ax.grid(True, zorder=20)

        Ei = Ef = np.power(2 * np.pi / (wavelength * TasUBlibDEG.factorsqrtEK), 2.0)

        A4 = [A4Start]
        if A4Stop is None:
            A4.append(A4Start - 130)
        else:
            A4.append(A4Stop)

        QLengths = np.linalg.norm([TasUBlibDEG.calcTasQH(np.eye(3), [0, a4], Ei, Ef)[0] for a4 in A4], axis=1)

        theta = np.linspace(0, np.pi * 2, 315)
        color = None
        for qlength in QLengths:
            qx = np.cos(theta) * qlength
            qy = np.sin(theta) * qlength
            p = ax.plot(qx, qy, c=color)
            color = p[0].get_color()

        return ax

    # Create the GUI
    root = tk.Tk()
    root.title("DMC predictive calculator")

    # Create input fields for a, b, c, al, be, ga, h1, k1, l1, h2, k2, l2, wavelength, and A4Start
    a_label = tk.Label(root, text="a [AA]:")
    a_label.grid(row=0, column=0)
    a_entry = tk.Entry(root)
    a_entry.grid(row=0, column=1)
    a_entry.insert(tk.END, "5")

    b_label = tk.Label(root, text="b [AA]:")
    b_label.grid(row=1, column=0)
    b_entry = tk.Entry(root)
    b_entry.grid(row=1, column=1)
    b_entry.insert(tk.END, "5")

    c_label = tk.Label(root, text="c [AA]:")
    c_label.grid(row=2, column=0)
    c_entry = tk.Entry(root)
    c_entry.grid(row=2, column=1)
    c_entry.insert(tk.END, "5")

    al_label = tk.Label(root, text="alpha [deg.]:")
    al_label.grid(row=3, column=0)
    al_entry = tk.Entry(root)
    al_entry.grid(row=3, column=1)
    al_entry.insert(tk.END, "90")


    be_label = tk.Label(root, text="beta [deg.]:")
    be_label.grid(row=4, column=0)
    be_entry = tk.Entry(root)
    be_entry.grid(row=4, column=1)
    be_entry.insert(tk.END, "90")

    ga_label = tk.Label(root, text="gamma [deg.]:")
    ga_label.grid(row=5, column=0)
    ga_entry = tk.Entry(root)
    ga_entry.grid(row=5, column=1)
    ga_entry.insert(tk.END, "90")

    wavelength_label = tk.Label(root, text="Wavelength [AA]:")
    wavelength_label.grid(row=6, column=0)
    wavelength_entry = tk.Entry(root)
    wavelength_entry.grid(row=6, column=1)
    wavelength_entry.insert(tk.END, "2.46")

    A4Start_label = tk.Label(root, text="A4 start angle [deg.]:")
    A4Start_label.grid(row=7, column=0)
    A4Start_entry = tk.Entry(root)
    A4Start_entry.grid(row=7, column=1)
    A4Start_entry.insert(tk.END, "5")

    hkl1_label = tk.Label(root, text="hkl first direction (x):")
    hkl1_label.grid(row=0, column=2, columnspan=2)

    h1_label = tk.Label(root, text="     h:")
    h1_label.grid(row=1, column=2)
    h1_entry = tk.Entry(root)
    h1_entry.grid(row=1, column=3)
    h1_entry.insert(tk.END, "1")

    k1_label = tk.Label(root, text="     k:")
    k1_label.grid(row=2, column=2)
    k1_entry = tk.Entry(root)
    k1_entry.grid(row=2, column=3)
    k1_entry.insert(tk.END, "0")

    l1_label = tk.Label(root, text="     l:")
    l1_label.grid(row=3, column=2)
    l1_entry = tk.Entry(root)
    l1_entry.grid(row=3, column=3)
    l1_entry.insert(tk.END, "0")

    hkl2_label = tk.Label(root, text="hkl second direction (y):")
    hkl2_label.grid(row=4, column=2, columnspan=2)

    h2_label = tk.Label(root, text="     h:")
    h2_label.grid(row=5, column=2)
    h2_entry = tk.Entry(root)
    h2_entry.grid(row=5, column=3)
    h2_entry.insert(tk.END, "0")
    
    k2_label = tk.Label(root, text="     k:")
    k2_label.grid(row=6, column=2)
    k2_entry = tk.Entry(root)
    k2_entry.grid(row=6, column=3)
    k2_entry.insert(tk.END, "1")

    l2_label = tk.Label(root, text="     l:")
    l2_label.grid(row=7, column=2)
    l2_entry = tk.Entry(root)
    l2_entry.grid(row=7, column=3)
    l2_entry.insert(tk.END, "0")

    # Create a button to generate the plot
    plot_button = tk.Button(root, text="Generate Plot", command=lambda: plot_sample(float(a_entry.get()), float(b_entry.get()), float(c_entry.get()), float(al_entry.get()), float(be_entry.get()), float(ga_entry.get()), float(h1_entry.get()), float(k1_entry.get()), float(l1_entry.get()), float(h2_entry.get()), float(k2_entry.get()), float(l2_entry.get()), float(wavelength_entry.get()), float(A4Start_entry.get())))
    plot_button.grid(row=8, column=0, columnspan=4 )

    root.resizable(width=False, height=False)

    # Start the gui
    root.mainloop()
