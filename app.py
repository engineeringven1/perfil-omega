import matplotlib
matplotlib.use("Agg")

from flask import Flask, render_template, request, jsonify, send_file
from sectionproperties.pre.geometry import Geometry
from sectionproperties.analysis import Section
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
import base64

app = Flask(__name__)


def crear_perfil_omega_puntos(A, B, e, degree):
    theta = np.deg2rad(degree)
    puntos_contorno = [
        (0, A * np.cos(theta) + e - e * np.sin(theta)),
        (e * np.cos(theta), A * np.cos(theta) + e),
        (e * np.cos(theta) + A * np.sin(theta), e),
        (e * np.cos(theta) + A * np.sin(theta) + B, e),
        (e * np.cos(theta) + 2 * A * np.sin(theta) + B, A * np.cos(theta) + e),
        (
            2 * e * np.cos(theta) + 2 * A * np.sin(theta) + B,
            A * np.cos(theta) + e - e * np.sin(theta),
        ),
        (
            (2 * e * np.cos(theta) + 2 * A * np.sin(theta) + B)
            - (A * np.cos(theta) + e - e * np.sin(theta)) * np.tan(theta),
            0,
        ),
        (
            (A * np.cos(theta) + e - e * np.sin(theta)) * np.tan(theta),
            0,
        ),
    ]
    lineas_contorno = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0)]
    puntos_control = [(B, e / 2)]
    return puntos_contorno, lineas_contorno, puntos_control


def verificacion_clase_Q(L_val, e_val, fy_val):
    k = 0.43
    epsilon = np.sqrt(235.0 / fy_val)
    c = L_val - e_val
    if c <= 0:
        raise ValueError(f"Ancho libre c = {c:.4f} mm es <= 0. Sección no viable.")
    c_sobre_t = c / e_val
    esbeltez_limite_elastica = 28.4 * epsilon * np.sqrt(k)
    esbeltez_reducida = c_sobre_t / esbeltez_limite_elastica
    if esbeltez_reducida <= 0.673:
        clase = 3
        Q = 1.0
    else:
        clase = 4
        Q = (esbeltez_reducida - 0.22) / (esbeltez_reducida ** 2)
        Q = min(Q, 1.0)
    return Q, clase, esbeltez_reducida


def calcular_dimensiones_tnx(puntos):
    pts = np.array(puntos, dtype=float)
    xs, ys = pts[:, 0], pts[:, 1]
    height = ys.max() - ys.min()
    width = xs.max() - xs.min()
    perimeter = sum(
        np.hypot(pts[(i+1) % len(pts)][0] - pts[i][0], pts[(i+1) % len(pts)][1] - pts[i][1])
        for i in range(len(pts))
    )
    return height, width, width, perimeter


def calcular_propiedades(A_val, B_val, e_val, degree_val=30.0, fy_val=355.0):
    mesh_size_val = A_val / 50.0
    puntos, lineas, controles = crear_perfil_omega_puntos(A_val, B_val, e_val, degree_val)
    height, width, wind_proj, perimeter = calcular_dimensiones_tnx(puntos)

    geometry = Geometry.from_points(points=puntos, facets=lineas, control_points=controles)
    geometry.create_mesh(mesh_sizes=mesh_size_val)
    section = Section(geometry)
    section.calculate_geometric_properties()
    section.calculate_warping_properties()
    section.calculate_plastic_properties()

    area = section.get_area()
    ixx_c, iyy_c, ixy_c = section.get_ic()
    zxx_plus, zxx_minus, zyy_plus, zyy_minus = section.get_z()
    rx, ry = section.get_rc()
    cw = section.get_gamma()
    j = section.get_j()

    Q, clase, esbeltez = verificacion_clase_Q(A_val, e_val, fy_val)

    us_name = f"UV{int(A_val)}x{int(B_val)}x{int(A_val)}x{int(e_val)}"

    return {
        "US Name": us_name,
        "SI Name": us_name,
        "Height [mm]": round(height, 4),
        "Width [mm]": round(width, 4),
        "Wind Proj [mm]": round(wind_proj, 4),
        "Perimeter [mm]": round(perimeter, 4),
        "Modulus [kPa]": 199947910.894,
        "Density [kg/m3]": 7849.049267,
        "Area [mm²]": round(area, 4),
        "Ix [mm⁴]": round(ixx_c, 4),
        "Sx top [mm³]": round(zxx_plus, 4),
        "Sx bot [mm³]": round(zxx_minus, 4),
        "rx [mm]": round(rx, 4),
        "SFy": 1,
        "QaQs": 1,
        "Iy [mm⁴]": round(iyy_c, 4),
        "Sy top [mm³]": round(zyy_plus, 4),
        "Sy bot [mm³]": round(zyy_minus, 4),
        "ry [mm]": round(ry, 4),
        "SFx": 1,
        "Cw [mm⁶]": round(cw, 4),
        "J [mm⁴]": round(j, 4),
        "Clase EC3": clase,
        "Factor Q": round(Q, 6),
        "Esbeltez reducida": round(esbeltez, 6),
    }, puntos


def generar_imagen_perfil(puntos):
    fig, ax = plt.subplots(figsize=(5, 5))
    pts = list(puntos) + [puntos[0]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.fill(xs[:-1], ys[:-1], color="#4A90D9", alpha=0.4)
    ax.plot(xs, ys, color="#1A5C9E", linewidth=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Perfil Omega — Sección Transversal")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/calcular", methods=["POST"])
def calcular():
    try:
        A = float(request.form["A"])
        B = float(request.form["B"])
        e = float(request.form["e"])
        degree = float(request.form.get("degree", 30.0))
        fy = float(request.form.get("fy", 355.0))

        if A <= 0 or B <= 0 or e <= 0:
            raise ValueError("A, B y e deben ser mayores que cero.")
        if e >= A:
            raise ValueError("El espesor e debe ser menor que el ala A.")

        resultados, puntos = calcular_propiedades(A, B, e, degree, fy)
        imagen = generar_imagen_perfil(puntos)
        return render_template("index.html", resultados=resultados, imagen=imagen,
                               A=A, B=B, e=e, degree=degree, fy=fy)
    except Exception as ex:
        return render_template("index.html", error=str(ex),
                               A=request.form.get("A",""),
                               B=request.form.get("B",""),
                               e=request.form.get("e",""),
                               degree=request.form.get("degree","30"),
                               fy=request.form.get("fy","355"))


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
