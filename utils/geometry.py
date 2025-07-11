"""
Funciones geométricas para detección y análisis espacial.
"""

def is_point_in_polygon(point, polygon):
    """Verifica si un punto está dentro de un polígono usando el algoritmo de Ray Casting"""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def calculate_intersection(box, mask_points):
    """Calcula si hay intersección entre un bounding box y una máscara"""
    x1, y1, x2, y2 = box

    for point in mask_points:
        px, py = point
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True

    corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
    for corner in corners:
        if is_point_in_polygon(corner, mask_points):
            return True

    return False
