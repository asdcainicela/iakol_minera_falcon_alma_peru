"""
geometry.py

Funciones geométricas para detección y análisis espacial en visión computacional.
"""

from typing import List, Tuple

Point = Tuple[float, float]
Polygon = List[Point]
Box = Tuple[float, float, float, float]


def is_point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """
    Verifica si un punto está dentro de un polígono usando el algoritmo de Ray Casting.

    Parámetros
    ----------
    point : Tuple[float, float]
        Punto (x, y) a verificar.
    polygon : List[Tuple[float, float]]
        Lista de puntos que representan el polígono en orden.

    Retorna
    -------
    bool
        True si el punto está dentro del polígono, False en caso contrario.
    """
    x, y = point
    inside = False
    n = len(polygon)

    if n < 3:
        return False  # No es un polígono válido

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y + 1e-9) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def calculate_intersection(box: Box, mask_points: Polygon) -> bool:
    """
    Determina si un bounding box y una máscara tienen intersección.

    Parámetros
    ----------
    box : Tuple[float, float, float, float]
        Bounding box en formato (x1, y1, x2, y2).
    mask_points : List[Tuple[float, float]]
        Puntos que definen la máscara poligonal.

    Retorna
    -------
    bool
        True si hay intersección, False en caso contrario.
    """
    x1, y1, x2, y2 = box

    # Si cualquier punto de la máscara está dentro del bounding box
    if any(x1 <= px <= x2 and y1 <= py <= y2 for px, py in mask_points):
        return True

    # O si alguna esquina del box está dentro de la máscara
    corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
    if any(is_point_in_polygon(corner, mask_points) for corner in corners):
        return True

    return False


if __name__ == "__main__":  # Ejemplo de uso
    # Definición de un polígono y puntos para prueba
    poly = [(0, 0), (5, 0), (5, 5), (0, 5)]
    point_inside = (3, 3)
    point_outside = (6, 3)

    print(f"Punto {point_inside} dentro de polígono? {is_point_in_polygon(point_inside, poly)}")
    print(f"Punto {point_outside} dentro de polígono? {is_point_in_polygon(point_outside, poly)}")

    box = (2, 2, 4, 4)
    mask = [(3, 3), (6, 3), (6, 6), (3, 6)]

    print(f"Intersección entre caja {box} y máscara? {calculate_intersection(box, mask)}")

    box2 = (0, 0, 1, 1)
    print(f"Intersección entre caja {box2} y máscara? {calculate_intersection(box2, mask)}")
