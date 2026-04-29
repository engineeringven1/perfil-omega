import matplotlib
matplotlib.use("Agg")

from flask import Flask, render_template, request, jsonify, send_file
from sectionproperties.pre.geometry import Geometry
from sectionproperties.analysis import Section
from shapely.geometry import Polygon as ShapelyPolygon
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


def calcular_propiedades(A_val, B_val, e_val, degree_val=30.0):
    mesh_size_val = max(A_val / 15.0, 2.0)
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
    cx_abs, cy_abs = section.get_c()

    # Centroide relativo al centro del bounding box del perfil
    pts_arr = np.array(puntos, dtype=float)
    x_mid = (pts_arr[:, 0].max() + pts_arr[:, 0].min()) / 2
    y_mid = (pts_arr[:, 1].max() + pts_arr[:, 1].min()) / 2

    si_name = f"UV{int(A_val)}x{int(B_val)}x{int(A_val)}x{int(e_val)}"
    peso_n_m = round((area / 1e6) * 7850 * 9.81, 4)

    return {
        "SI Name": si_name,
        "Height [mm]": round(height, 4),
        "Width [mm]": round(width, 4),
        "Wind Proj [mm]": round(wind_proj, 4),
        "Perimeter [mm]": round(perimeter, 4),
        "Modulus [kPa]": 199947910.894,
        "Density [kg/m3]": 7849.049267,
        "Area [mm²]": round(area, 4),
        "Peso [N/m]": peso_n_m,
        "cx [mm]": round(cx_abs - x_mid, 4),
        "cy [mm]": round(cy_abs - y_mid, 4),
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
    }, puntos, (cx_abs, cy_abs)


def _dibujar_centroide(ax, cx, cy):
    ax.axhline(y=cy, color="#E53E3E", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvline(x=cx, color="#E53E3E", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.plot(cx, cy, "o", color="#E53E3E", markersize=7, zorder=5, label=f"CM ({cx:.1f}, {cy:.1f})")
    ax.legend(fontsize=7, loc="upper right")


def generar_imagen_perfil(puntos, centroid=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    pts = list(puntos) + [puntos[0]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.fill(xs[:-1], ys[:-1], color="#4A90D9", alpha=0.4)
    ax.plot(xs, ys, color="#1A5C9E", linewidth=2)
    if centroid:
        _dibujar_centroide(ax, *centroid)
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
    return render_template("index.html", u=_u_default(), u_doble=_u_default())


@app.route("/calcular", methods=["POST"])
def calcular():
    try:
        A = float(request.form["A"])
        B = float(request.form["B"])
        e = float(request.form["e"])
        unidad = request.form.get("unidad", "mm")

        if A <= 0 or B <= 0 or e <= 0:
            raise ValueError("A, B y e deben ser mayores que cero.")
        if e >= A:
            raise ValueError("El espesor e debe ser menor que el ala A.")

        resultados, puntos, centroid = calcular_propiedades(A, B, e)
        resultados, u = _aplicar_unidad(resultados, unidad)
        imagen = generar_imagen_perfil(puntos, centroid)
        return render_template("index.html", resultados=resultados, imagen=imagen,
                               A=A, B=B, e=e, unidad=unidad, u=u, u_doble=_u_default())
    except Exception as ex:
        return render_template("index.html", error=str(ex),
                               A=request.form.get("A",""),
                               B=request.form.get("B",""),
                               e=request.form.get("e",""),
                               unidad=request.form.get("unidad","mm"),
                               u=_u_default(), u_doble=_u_default())


_CONV = {
    "Height [mm]":    0.1,   "Width [mm]":     0.1,
    "Wind Proj [mm]": 0.1,   "Perimeter [mm]": 0.1,
    "Area [mm²]":     0.01,
    "rx [mm]":        0.1,   "ry [mm]":        0.1,
    "cx [mm]":        0.1,   "cy [mm]":        0.1,
    "Ix [mm⁴]":      1e-4,  "Iy [mm⁴]":      1e-4,
    "Sx top [mm³]":  1e-3,  "Sx bot [mm³]":  1e-3,
    "Sy top [mm³]":  1e-3,  "Sy bot [mm³]":  1e-3,
    "Cw [mm⁶]":      1e-6,  "J [mm⁴]":       1e-4,
}

def _aplicar_unidad(res, unidad):
    if unidad == "cm":
        nuevo = dict(res)
        for key, factor in _CONV.items():
            if key in nuevo and isinstance(nuevo[key], (int, float)):
                nuevo[key] = round(nuevo[key] * factor, 6)
        u = {"len": "cm", "area": "cm²", "mod": "cm³", "iner": "cm⁴", "warp": "cm⁶"}
        return nuevo, u
    u = {"len": "mm", "area": "mm²", "mod": "mm³", "iner": "mm⁴", "warp": "mm⁶"}
    return res, u


def _u_default():
    return {"len": "mm", "area": "mm²", "mod": "mm³", "iner": "mm⁴", "warp": "mm⁶"}


def crear_doble_omega_partes(A1, B1, e1, A2, B2, e2, degree=30.0):
    pts1, _, _ = crear_perfil_omega_puntos(A1, B1, e1, degree)
    pts2_up, _, _ = crear_perfil_omega_puntos(A2, B2, e2, degree)

    # Alinear centros horizontales de ambos perfiles
    xs1 = [p[0] for p in pts1]
    cx1 = (max(xs1) + min(xs1)) / 2
    xs2 = [p[0] for p in pts2_up]
    cx2 = (max(xs2) + min(xs2)) / 2
    dx = cx1 - cx2

    # Pequeño gap vertical para evitar bordes coincidentes en y=0
    GAP = 0.01
    pts2 = [(x + dx, -y - GAP) for x, y in pts2_up]
    return pts1, pts2


def calcular_propiedades_doble(A1_val, B1_val, e1_val, A2_val, B2_val, e2_val, degree_val=30.0):
    mesh_size_val = max(min(A1_val, A2_val) / 15.0, 2.0)
    pts1, pts2 = crear_doble_omega_partes(A1_val, B1_val, e1_val, A2_val, B2_val, e2_val, degree_val)

    geom1 = Geometry(ShapelyPolygon(pts1))
    geom2 = Geometry(ShapelyPolygon(pts2))
    compound = geom1 + geom2

    compound.create_mesh(mesh_sizes=mesh_size_val)
    section = Section(compound)
    section.calculate_geometric_properties()
    section.calculate_warping_properties()
    section.calculate_plastic_properties()

    area = section.get_area()
    ixx_c, iyy_c, ixy_c = section.get_ic()
    zxx_plus, zxx_minus, zyy_plus, zyy_minus = section.get_z()
    rx, ry = section.get_rc()
    cw = section.get_gamma()
    j = section.get_j()
    cx_abs, cy_abs = section.get_c()

    all_pts = np.array(pts1 + pts2, dtype=float)
    height = round(all_pts[:, 1].max() - all_pts[:, 1].min(), 4)
    width = round(all_pts[:, 0].max() - all_pts[:, 0].min(), 4)
    x_mid = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2
    y_mid = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2

    si_name = f"DUV{int(A1_val)}x{int(B1_val)}x{int(e1_val)}-{int(A2_val)}x{int(B2_val)}x{int(e2_val)}"
    peso_n_m = round((area / 1e6) * 7849.049267 * 9.81, 4)

    return {
        "SI Name": si_name,
        "Height [mm]": height,
        "Width [mm]": width,
        "Area [mm²]": round(area, 4),
        "Peso [N/m]": peso_n_m,
        "cx [mm]": round(cx_abs - x_mid, 4),
        "cy [mm]": round(cy_abs - y_mid, 4),
        "Ix [mm⁴]": round(ixx_c, 4),
        "Sx top [mm³]": round(zxx_plus, 4),
        "Sx bot [mm³]": round(zxx_minus, 4),
        "rx [mm]": round(rx, 4),
        "Iy [mm⁴]": round(iyy_c, 4),
        "Sy top [mm³]": round(zyy_plus, 4),
        "Sy bot [mm³]": round(zyy_minus, 4),
        "ry [mm]": round(ry, 4),
        "Cw [mm⁶]": round(cw, 4),
        "J [mm⁴]": round(j, 4),
    }, pts1, pts2, (cx_abs, cy_abs)


def generar_imagen_doble_omega(pts1, pts2, centroid=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    for pts in [pts1, pts2]:
        closed = list(pts) + [pts[0]]
        xs = [p[0] for p in closed]
        ys = [p[1] for p in closed]
        ax.fill(xs[:-1], ys[:-1], color="#4A90D9", alpha=0.4)
        ax.plot(xs, ys, color="#1A5C9E", linewidth=2)
    ax.axhline(y=0, color="#718096", linewidth=0.8, linestyle="--", alpha=0.5)
    if centroid:
        _dibujar_centroide(ax, *centroid)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Doble Omega — Sección Transversal")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.route("/calcular_doble", methods=["POST"])
def calcular_doble():
    try:
        A1 = float(request.form["A1"])
        B1 = float(request.form["B1"])
        e1 = float(request.form["e1"])
        A2 = float(request.form["A2"])
        B2 = float(request.form["B2"])
        e2 = float(request.form["e2"])
        unidad_doble = request.form.get("unidad_doble", "mm")

        if any(v <= 0 for v in [A1, B1, e1, A2, B2, e2]):
            raise ValueError("Todos los parámetros deben ser mayores que cero.")
        if e1 >= A1:
            raise ValueError("El espesor e1 debe ser menor que el ala A1.")
        if e2 >= A2:
            raise ValueError("El espesor e2 debe ser menor que el ala A2.")

        resultados_doble, pts1, pts2, centroid_d = calcular_propiedades_doble(A1, B1, e1, A2, B2, e2)
        resultados_doble, u_doble = _aplicar_unidad(resultados_doble, unidad_doble)
        imagen_doble = generar_imagen_doble_omega(pts1, pts2, centroid_d)
        return render_template("index.html",
                               resultados_doble=resultados_doble, imagen_doble=imagen_doble,
                               A1=A1, B1=B1, e1=e1, A2=A2, B2=B2, e2=e2,
                               unidad_doble=unidad_doble, u=_u_default(), u_doble=u_doble)
    except Exception as ex:
        return render_template("index.html", error_doble=str(ex),
                               A1=request.form.get("A1", ""),
                               B1=request.form.get("B1", ""),
                               e1=request.form.get("e1", ""),
                               A2=request.form.get("A2", ""),
                               B2=request.form.get("B2", ""),
                               e2=request.form.get("e2", ""),
                               unidad_doble=request.form.get("unidad_doble","mm"),
                               u=_u_default(), u_doble=_u_default())


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
