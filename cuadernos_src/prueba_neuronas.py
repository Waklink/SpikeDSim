from matplotlib import pyplot as plt

from spikedsim.Izhikevich import Neurona, RedDeNeuronas, Simulador
from spikedsim import Visualizar

# Crear las neuronas

def obtener_neurona_corriente(neurona):
    corriente_base = 10
    if neurona == "tc2":
        n = Neurona.predefinida("tc", v_inicial=-87)
        corriente = 0
    elif neurona == "red":
        tc2 = Neurona.predefinida("tc", v_inicial=-87)
        n = RedDeNeuronas(neuronas={"rs": 1, "ib": 1, "ch": 1, "fs": 1, "tc": 1, tc2: 1, "rz": 1, "lts": 1},
                          conexiones=0, backend="numpy", precision=64, sparse=False)
        corriente_rz = [-0.045] * (pasos // 2) + [1.5] * 10 + [0.5] * (pasos // 2 - 10)
        corriente = []
        for i in range(pasos):
            corriente.append([corriente_base] * 4 + [1] + [0] + [corriente_rz[i]] + [corriente_base])
    else:
        n = Neurona.predefinida(neurona)
        if neurona == "tc":
            corriente = 1
        elif neurona == "rz":
            corriente = [-0.045] * (pasos // 2) + [1.5] * 10 + [0.5] * (pasos // 2 - 10)
        else:
            corriente = corriente_base
    return n, corriente

# Paso temporal y número total de pasos
dt = 0.1
pasos = int(200/dt)

# Crear el simulador y el visualizador
sim = Simulador(paso_temporal=dt, mostrar_progreso=True, medir_rendimiento=True, intervalo_rendimiento=100)
vis = Visualizar()

print("rs: Regular Spiking\nib: Intrinsically bursting\nch: Chattering")
print("fs: Fast Spiking\ntc: Thalamo-cortical (tonic firing)")
print("tc2: Thalamo-cortical (rebound bursts of action potentials")
print("rz: Resonator\nlts: Low-threshold Spiking\nred: Todas a la vez\n")
print("Cualquier otra cosa hará que se finalice el programa.")
neurona = input("\nIntroduce la neurona que quieras comprobar: ")
neurona = neurona.strip().lower()

if neurona in ("rs", "ib", "ch", "fs", "tc", "tc2", "rz", "lts"):
    n, corriente = obtener_neurona_corriente(neurona)
    sim.cargar_red(n)

    print()
    _ = sim.simular(pasos, corriente)
    print()

    # Crear las gráficas originales sin mostrar
    fig_v, ax_v = vis.potencial_membrana(historial=sim.historial, titulo=sim.red.nombre, max_etiquetas_leyenda=0, mostrar=False)
    fig_I, ax_I = vis.corriente(historial=sim.historial, titulo="", max_etiquetas_leyenda=0, mostrar=False)

    # Figura final: potencial arriba, corriente abajo
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), gridspec_kw={"height_ratios": [9, 1]}, sharex=True)

    ax_v_final, ax_I_final = axes

    # Potencial
    for linea in ax_v.lines:
        ax_v_final.plot(linea.get_xdata(), linea.get_ydata(), color="black", linewidth=0.8)

    # Corriente
    for linea in ax_I.lines:
        ax_I_final.plot(linea.get_xdata(), linea.get_ydata(), color="black", linewidth=0.8)

    # Quitar bordes
    for ax in (ax_v_final, ax_I_final):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_v_final.spines["bottom"].set_visible(False)
    ax_v_final.tick_params(axis="x", bottom=False, labelbottom=False)

    ax_v_final.set_title(sim.red.nombre)
    ax_v_final.set_ylabel("v (mV)")
    ax_I_final.set_ylabel("I")
    ax_I_final.set_xlabel("t (ms)")

    plt.tight_layout()
    plt.show()

    plt.close(fig_v)
    plt.close(fig_I)
elif neurona == "red":
    n, corriente = obtener_neurona_corriente(neurona)
    sim.cargar_red(n)

    print()
    _ = sim.simular(pasos, corriente)
    print()

    figuras = []

    for i in range(8):
        fig_v, ax_v = vis.potencial_membrana(historial=sim.historial, neuronas=i, titulo=sim.red.nombre[i], max_etiquetas_leyenda=0, mostrar=False)
        fig_I, ax_I = vis.corriente(historial=sim.historial, neuronas=i, titulo="", max_etiquetas_leyenda=0, mostrar=False)

        figuras.append((fig_v, ax_v, fig_I, ax_I))

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(5, 4, height_ratios=[9, 1, 4, 9, 1], hspace=0.05, wspace=0.35)

    for i, (fig_v, ax_v, fig_I, ax_I) in enumerate(figuras):
        columna = i % 4
        if i < 4:
            fila_v = 0
            fila_I = 1
        else:
            fila_v = 3
            fila_I = 4

        ax_v_final = fig.add_subplot(gs[fila_v, columna])
        ax_I_final = fig.add_subplot(gs[fila_I, columna], sharex=ax_v_final)

        # Potencial de membrana
        for linea in ax_v.lines:
            ax_v_final.plot(linea.get_xdata(), linea.get_ydata(), color="black", linewidth=0.8)

        # Corriente
        for linea in ax_I.lines:
            ax_I_final.plot(linea.get_xdata(), linea.get_ydata(), color="black", linewidth=0.8)

        # Nombre de la neurona
        ax_v_final.set_title(f"{sim.red.nombre[i]} ({Neurona.alias()[sim.red.nombre[i].strip().lower()].upper()})", loc="left", fontsize=11, fontweight="bold")

        # Etiquetas
        ax_v_final.set_ylabel("v (mV)")
        ax_I_final.set_ylabel("I")
        ax_I_final.set_xlabel("t (ms)")


        # Quitar borde superior y derecho
        for ax in (ax_v_final, ax_I_final):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax_v_final.spines["bottom"].set_visible(False)
        ax_v_final.tick_params(axis="x", bottom=False, labelbottom=False)


        plt.close(fig_v)
        plt.close(fig_I)

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.96)
    plt.show()
