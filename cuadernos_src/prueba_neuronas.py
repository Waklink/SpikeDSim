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

# Pedir una neurona y mostrar la evolución de su potencial de membrana
fin = False

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
    vis.potencial_membrana(historial=sim.historial, titulo=sim.red.nombre, max_etiquetas_leyenda=0)
    sim.reiniciar()
    sim.limpiar_todo()
elif neurona == "red":
    n, corriente = obtener_neurona_corriente(neurona)
    sim.cargar_red(n)
    print()
    _ = sim.simular(pasos, corriente)
    print()

    figuras = []

    for i in range(8):
        fig, ax = vis.potencial_membrana(historial=sim.historial, neuronas=i, titulo=sim.red.nombre[i], mostrar=False)
        figuras.append((fig, ax))

    fig_final, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    # Copiar las líneas
    for i, (fig, ax_original) in enumerate(figuras):
        ax_nuevo = axes[i]

        for linea in ax_original.lines:
            ax_nuevo.plot(linea.get_xdata(), linea.get_ydata())

        ax_nuevo.set_title(ax_original.get_title())

        plt.close(fig)

    plt.tight_layout()
    plt.show()

    sim.reiniciar()
    sim.limpiar_todo()
else:
    fin = True


while not fin:
    neurona = input("Introduce la neurona que quieras comprobar: ").strip().lower()

    if neurona in ("rs", "ib", "ch", "fs", "tc", "tc2", "rz", "lts", "red"):
        n, corriente = obtener_neurona_corriente(neurona)
        sim.cargar_red(n)
        _ = sim.simular(pasos, corriente)
        vis.potencial_membrana(historial=sim.historial, titulo=sim.red.nombre, max_etiquetas_leyenda=0)
        sim.reiniciar()
        sim.limpiar_todo()
    else:
        fin = True
