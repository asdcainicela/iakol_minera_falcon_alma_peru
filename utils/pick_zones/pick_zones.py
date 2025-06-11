import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import json
import os

class PolySelector:
    def __init__(self):
        self.image_path = None
        self.original_image = None  # Imagen original sin redimensionar
        self.display_image = None   # Imagen redimensionada para mostrar
        self.clone = None           # Copia de la imagen redimensionada
        self.scale_factor = 1.0     # Factor de escala entre original y visualización
        self.polygons = []          # Polígonos en coordenadas de la imagen original
        self.current_polygon = []   # Vértices actuales en coordenadas de la imagen original
        self.is_drawing = False
        self.window_name = "Selector de Poligonos"
        self.output_folder = "vertices_output"
        self.snap_threshold = 15    # Distancia en píxeles para atraer a vértices existentes
        self.snap_point = None      # Punto actual para atraer el cursor (en coords. de visualización)
        self.snap_orig_point = None # Punto actual para atraer el cursor (en coords. originales)
        
        # Crear carpeta de salida si no existe
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            
        # Obtener tamaño de la pantalla
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.destroy()
        
        # Definir tamaño máximo para la visualización (80% del tamaño de la pantalla)
        self.max_display_width = int(self.screen_width * 0.8)
        self.max_display_height = int(self.screen_height * 0.8)

    def select_image(self):
        # Configurar y mostrar el diálogo de selección de archivo
        root = tk.Tk()
        root.withdraw()  # Ocultar la ventana principal de tkinter
        self.image_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not self.image_path:
            print("No se selecciono ninguna imagen. Saliendo...")
            return False
            
        # Cargar la imagen original
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            print(f"No se pudo cargar la imagen: {self.image_path}")
            return False
        
        # Redimensionar para visualización manteniendo la relación de aspecto
        self.resize_image_for_display()
        
        return True
    
    def resize_image_for_display(self):
        """Redimensiona la imagen para ajustarse a la pantalla"""
        orig_height, orig_width = self.original_image.shape[:2]
        
        # Calcular el factor de escala para ajustar a la pantalla
        width_scale = self.max_display_width / orig_width
        height_scale = self.max_display_height / orig_height
        
        # Usar el factor más pequeño para asegurar que la imagen quepa en la pantalla
        self.scale_factor = min(width_scale, height_scale)
        
        # Calcular nuevas dimensiones
        new_width = int(orig_width * self.scale_factor)
        new_height = int(orig_height * self.scale_factor)
        
        # Redimensionar la imagen para visualización
        self.display_image = cv2.resize(self.original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        self.clone = self.display_image.copy()
        
        print(f"Imagen original: {orig_width}x{orig_height}")
        print(f"Imagen redimensionada: {new_width}x{new_height}")
        print(f"Factor de escala: {self.scale_factor:.4f}")

    def display_to_original_coords(self, x, y):
        """Convierte coordenadas de la imagen mostrada a la imagen original"""
        orig_x = int(x / self.scale_factor)
        orig_y = int(y / self.scale_factor)
        return orig_x, orig_y
    
    def original_to_display_coords(self, x, y):
        """Convierte coordenadas de la imagen original a la imagen mostrada"""
        disp_x = int(x * self.scale_factor)
        disp_y = int(y * self.scale_factor)
        return disp_x, disp_y
    
    def find_closest_vertex(self, x, y):
        """
        Busca el vértice más cercano en polígonos existentes.
        Devuelve el punto más cercano en coordenadas de visualización y coordenadas originales,
        o None si no hay puntos dentro del umbral.
        """
        closest_point = None
        closest_orig_point = None
        min_distance = float('inf')
        
        # Revisar todos los vértices en polígonos existentes
        for poly in self.polygons:
            for vertex in poly:
                # Convertir a coordenadas de visualización
                disp_x, disp_y = self.original_to_display_coords(vertex[0], vertex[1])
                
                # Calcular distancia
                dist = np.sqrt((x - disp_x)**2 + (y - disp_y)**2)
                
                # Actualizar si es el más cercano
                if dist < min_distance and dist < self.snap_threshold:
                    min_distance = dist
                    closest_point = (disp_x, disp_y)
                    closest_orig_point = (vertex[0], vertex[1])
        
        return closest_point, closest_orig_point

    def mouse_callback(self, event, x, y, flags, param):
        # Buscar si estamos cerca de un vértice existente para "snap"
        self.snap_point, self.snap_orig_point = self.find_closest_vertex(x, y)
        
        # Coordenadas actuales para dibujar (con snap si aplica)
        current_x, current_y = (self.snap_point[0], self.snap_point[1]) if self.snap_point else (x, y)
        
        # Copiar la imagen para mostrar los cambios
        display_image = self.clone.copy()
        
        # Mostrar vértices y polígonos ya completados
        for poly in self.polygons:
            # Convertir los puntos de coordenadas originales a coordenadas de visualización
            display_poly = [self.original_to_display_coords(pt[0], pt[1]) for pt in poly]
            
            for i in range(len(display_poly)):
                # Dibujar vértices
                cv2.circle(display_image, display_poly[i], 5, (0, 255, 0), -1)
                # Dibujar líneas entre vértices
                if i < len(display_poly) - 1:
                    cv2.line(display_image, display_poly[i], display_poly[i+1], (0, 255, 0), 2)
                else:  # Conectar el último punto con el primero
                    cv2.line(display_image, display_poly[i], display_poly[0], (0, 255, 0), 2)
        
        # Si estamos en "snap", resaltar el vértice al que nos acercamos
        if self.snap_point:
            cv2.circle(display_image, self.snap_point, 8, (255, 255, 0), 2)  # Círculo amarillo alrededor
        
        # Mostrar el polígono actual que se está dibujando
        if self.current_polygon:
            # Convertir los puntos de coordenadas originales a coordenadas de visualización
            display_current = [self.original_to_display_coords(pt[0], pt[1]) for pt in self.current_polygon]
            
            for i in range(len(display_current)):
                # Dibujar vértices
                cv2.circle(display_image, display_current[i], 5, (0, 0, 255), -1)
                # Dibujar líneas entre vértices
                if i < len(display_current) - 1:
                    cv2.line(display_image, display_current[i], display_current[i+1], (0, 0, 255), 2)
                
                # Si hay más de un punto, dibujar línea al cursor o al punto de snap
                if i == len(display_current) - 1:
                    # Dibujar línea desde el último punto hasta la posición actual del mouse/snap
                    cv2.line(display_image, display_current[i], (current_x, current_y), (0, 0, 255), 2)
                    
                    # Si estamos cerca del primer punto, mostrar una línea al primer punto
                    dist_to_first = np.sqrt((current_x - display_current[0][0])**2 + 
                                           (current_y - display_current[0][1])**2)
                    if dist_to_first < 20:  # 20 píxeles de tolerancia
                        cv2.line(display_image, display_current[i], display_current[0], (255, 0, 0), 2)
        
        # Manejar eventos del mouse
        if event == cv2.EVENT_LBUTTONDOWN:
            # Determinar las coordenadas a usar (originales)
            if self.snap_orig_point:
                # Usar coordenadas exactas del vértice existente
                orig_x, orig_y = self.snap_orig_point
                print(f"Vertice ajustado a un punto existente: ({orig_x}, {orig_y})")
            else:
                # Convertir coordenadas de visualización a coordenadas originales
                orig_x, orig_y = self.display_to_original_coords(x, y)
            
            # Si no estamos dibujando, iniciar un nuevo polígono
            if not self.is_drawing:
                self.current_polygon = []
                self.is_drawing = True
            
            # Revisar si estamos cerca del primer punto (para cerrar el polígono)
            if len(self.current_polygon) > 2:
                # Obtener el primer punto en coordenadas de visualización
                first_display_x, first_display_y = self.original_to_display_coords(
                    self.current_polygon[0][0], self.current_polygon[0][1]
                )
                
                dist_to_first = np.sqrt((current_x - first_display_x)**2 + (current_y - first_display_y)**2)
                if dist_to_first < 20:  # 20 píxeles de tolerancia
                    # Cerrar el polígono
                    self.polygons.append(self.current_polygon.copy())
                    self.current_polygon = []
                    self.is_drawing = False
                    print(f"Polígono {len(self.polygons)} completado con {len(self.polygons[-1])} vértices")
                    cv2.imshow(self.window_name, display_image)
                    return
            
            # Agregar el nuevo punto al polígono actual (en coordenadas originales)
            self.current_polygon.append([orig_x, orig_y])
            print(f"Vertice añadido en coordenadas originales: ({orig_x}, {orig_y})")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Botón derecho para cancelar el polígono actual
            if self.is_drawing:
                self.current_polygon = []
                self.is_drawing = False
                print("Poligono cancelado")
        
        cv2.imshow(self.window_name, display_image)
        
    def run(self):
        if not self.select_image():
            return
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Mostrar instrucciones
        print("\n=== INSTRUCCIONES ===")
        print("- Haz clic para seleccionar los vértices del polígono")
        print("- Acerca el cursor a vértices existentes para reutilizarlos (se mostrará un círculo amarillo)")
        print("- Acerca el cursor al primer vértice para cerrar el polígono")
        print("- Haz clic derecho para cancelar el polígono actual")
        print("- Presiona 'r' para reiniciar todos los polígonos")
        print("- Presiona 's' para guardar los polígonos")
        print("- Presiona 'q' para salir sin guardar")
        print("=====================\n")
        
        # Mostrar la imagen inicial
        cv2.imshow(self.window_name, self.display_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Tecla q para salir
            if key == ord('q'):
                break
                
            # Tecla r para reiniciar
            elif key == ord('r'):
                self.polygons = []
                self.current_polygon = []
                self.is_drawing = False
                print("Todos los polígonos han sido reiniciados")
                cv2.imshow(self.window_name, self.clone)
                
            # Tecla s para guardar
            elif key == ord('s'):
                self.save_polygons()
        
        cv2.destroyAllWindows()
    
    def save_polygons(self):
        if not self.polygons:
            print("No hay polígonos para guardar")
            return
            
        # Obtener el nombre base de la imagen
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        output_path = os.path.join(self.output_folder, f"{base_name}_vertices.json")
        
        # Guardar los polígonos en formato JSON con formato personalizado
        # (un polígono por línea, sin tabulación adicional)
        with open(output_path, 'w') as f:
            # Escribir el inicio del array JSON
            f.write("[\n")
            
            # Escribir cada polígono en una línea independiente
            for i, poly in enumerate(self.polygons):
                # Convertir el polígono a JSON string sin pretty print
                poly_json = json.dumps(poly, separators=(',', ':'))
                
                # Añadir coma si no es el último polígono
                if i < len(self.polygons) - 1:
                    f.write(f"{poly_json},\n")
                else:
                    f.write(f"{poly_json}\n")
            
            # Escribir el final del array JSON
            f.write("]")
            
        print(f"Se guardaron {len(self.polygons)} polígonos en {output_path}")
        
        # Guardar imagen con los polígonos marcados (usando una copia de la imagen original)
        vis_image = self.original_image.copy()
        for i, poly in enumerate(self.polygons):
            # Color diferente para cada polígono
            color = (
                (i * 50) % 255,
                (i * 100) % 255,
                (i * 150) % 255
            )
            
            # Dibujar el polígono en la imagen original
            pts = np.array(poly, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(vis_image, [pts], True, color, 3)
            
            # Numerar cada polígono
            centroid_x = int(sum(p[0] for p in poly) / len(poly))
            centroid_y = int(sum(p[1] for p in poly) / len(poly))
            cv2.putText(vis_image, str(i+1), (centroid_x, centroid_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            
        # Guardar la imagen con los polígonos
        vis_path = os.path.join(self.output_folder, f"{base_name}_visualizacion.jpg")
        cv2.imwrite(vis_path, vis_image)
        print(f"Se guardó la visualización en {vis_path}")

if __name__ == "__main__":
    selector = PolySelector()
    selector.run()