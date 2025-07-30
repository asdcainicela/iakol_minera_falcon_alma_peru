import cv2

def put_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                             font_scale=0.4, color=(255,255,255), thickness=1,
                             bg_color=(0,0,0), bg_alpha=0.6):
    """Coloca texto con fondo semitransparente para mejor legibilidad"""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position
    bg_img = img.copy()
    padding = 5

    cv2.rectangle(bg_img, (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + padding), bg_color, -1)

    overlay = cv2.addWeighted(bg_img, bg_alpha, img, 1 - bg_alpha, 0)
    cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness)

    return overlay


def draw_zone_and_status(frame, detection_zone, draw_object_in_zone,
                         object_interacting, draw_ruma_variation,
                         TEXT_COLOR_RED=(0, 0, 255), TEXT_COLOR_GREEN=(0, 255, 0)):
    """Dibuja la zona de detección y el estado de las alertas"""
    width = frame.shape[1]

    # Dibujar zona de detección
    pts = detection_zone.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    # Textos de estado
    text_y_start = 50

    zone_text = "Movimiento en la zona" if draw_object_in_zone else "Zona despejada"
    zone_color = TEXT_COLOR_RED if draw_object_in_zone else TEXT_COLOR_GREEN
    frame = put_text_with_background(
        frame, zone_text, (width - 650, text_y_start),
        color=zone_color, font_scale=1.5
    )

    interact_text = "Interaccion con las rumas" if object_interacting else "Sin interacciones"
    interact_color = TEXT_COLOR_RED if object_interacting else TEXT_COLOR_GREEN
    frame = put_text_with_background(
        frame, interact_text, (width - 650, text_y_start + 60),
        color=interact_color, font_scale=1.5
    )

    variation_text = "Variacion en las rumas" if draw_ruma_variation else "Rumas en reposo"
    variation_color = TEXT_COLOR_RED if draw_ruma_variation else TEXT_COLOR_GREEN
    frame = put_text_with_background(
        frame, variation_text, (width - 650, text_y_start + 120),
        color=variation_color, font_scale=1.5
    )

    return frame
