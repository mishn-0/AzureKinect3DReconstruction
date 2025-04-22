import numpy as np
import open3d as o3d
from pyk4a import PyK4A, Config, ColorResolution, DepthMode
import cv2
import time

# Configurar Kinect
k4a = PyK4A(
    Config(
        color_resolution=ColorResolution.RES_720P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True,
    )
)
k4a.start()

# Crear visualizador
vis = o3d.visualization.Visualizer()
vis.create_window("Nube de puntos en tiempo real")

# Esperar una primera captura válida
while True:
    capture = k4a.get_capture()
    if capture.color is not None and capture.transformed_depth is not None:
        break

# Flip vertical y horizontal
def process_images(color_img, depth_img):
    color_img = cv2.flip(color_img, -1)
    depth_img = cv2.flip(depth_img, -1)

    color_o3d = o3d.geometry.Image(color_img)
    depth_o3d = o3d.geometry.Image(depth_img)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        convert_rgb_to_intensity=False,
        depth_scale=2000.0,
        depth_trunc=4.0,
    )
    return rgbd

# Transformación para alinear con el eje correcto (suelo abajo)
flip_transform = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
])

# Configuración de la cámara
intrinsics = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
)

# Inicializar la nube
color = capture.color
depth = capture.transformed_depth
rgbd = process_images(color, depth)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
pcd.transform(flip_transform)
vis.add_geometry(pcd)

# Configurar la cámara en la vista
vis.poll_events()  # Esperar la inicialización de la ventana
vis.update_renderer()

# Configurar la vista inicial de la cámara
ctr = vis.get_view_control()
ctr.set_front([0.0, 0.0, -1.0])
ctr.set_up([0.0, -1.0, 0.0])
ctr.set_lookat([0.0, 0.0, 0.0])
ctr.set_zoom(0.5)

# Loop de visualización en tiempo real
try:
    while True:
        capture = k4a.get_capture()
        if capture.color is None or capture.transformed_depth is None:
            continue

        color = capture.color
        depth = capture.transformed_depth
        rgbd = process_images(color, depth)

        new_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
        new_pcd.transform(flip_transform)

        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.03)

except KeyboardInterrupt:
    print("Finalizando visualización.")
finally:
    k4a.stop()
    vis.destroy_window()
