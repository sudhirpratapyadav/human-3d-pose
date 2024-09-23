import numpy as np
from vispy import app, scene
from vispy.scene import visuals

from vispy.visuals.transforms import STTransform, MatrixTransform

class PoseVisualiser:
    def __init__(self, connections, show_grid=True, camera_pose=None, cam_fov_v=65, img_width=640, img_height=480, display_image=True, camera_intrinsics=None):
        # Create a canvas with a 3D viewport
        self.canvas = scene.SceneCanvas(keys='interactive', show=True, size=(800, 600))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(fov=60, elevation=30, azimuth=0)
        self.view.padding = 0
        self.view.camera.fov = 45
        self.view.camera.distance = 5

        # Create a scatter plot for vertices
        self.scatter = visuals.Markers()
        self.view.add(self.scatter)

        # Create lines for edges
        self.lines = visuals.Line(color='red', method='gl')
        self.view.add(self.lines)

        # Add axes
        self.axis = visuals.XYZAxis(parent=self.view.scene)

        # Add grid
        self.grid = scene.GridLines(parent=self.view.scene)
        self.grid.set_gl_state('translucent', depth_test=False)
        self.grid.visible = show_grid

        # Add camera visualization if camera_pose is provided
        self.display_image = display_image
        self.frame = None
        self.img_width = img_width
        self.img_height = img_height
        self.camera_intrinsics = camera_intrinsics
        aspect_ratio = self.img_width/self.img_height
        if camera_pose is not None:
            self.add_camera_visualization(camera_pose , cam_fov_v, aspect_ratio)

        # Set up a timer for updating
        self.timer = app.Timer(interval=0.05, connect=self.on_timer, start=True)
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_release.connect(self.on_mouse_release)
        self.canvas.events.key_press.connect(self.on_key_press)

        self.user_interacting = False
        self.auto_rotate = True
        self.rotation_speed = 1.0  # degrees per frame

        # Store the latest points and connections
        self.points = None
        self.connections = connections
    
    def on_mouse_press(self, event):
        self.user_interacting = True

    def on_mouse_release(self, event):
        self.user_interacting = False

    def on_key_press(self, event):
        if event.key == ' ':  # Spacebar toggles auto-rotation
            self.auto_rotate = not self.auto_rotate
            print(f"Auto-rotation {'enabled' if self.auto_rotate else 'disabled'}")

    def add_camera_visualization(self, camera_pose, fov_v=65, aspect_ratio=16/9):
        # Extract position and rotation from the homogeneous matrix
        position = camera_pose[:3, 3]
        rotation = camera_pose[:3, :3]

        # Create custom axes using lines
        axis_length = 0.5
        colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]  # RGB for XYZ
        
        self.camera_axes = []
        for i in range(3):
            axis_end = position + rotation[:, i] * axis_length
            axis_points = np.array([position, axis_end])
            axis_line = visuals.Line(pos=axis_points, color=colors[i], parent=self.view.scene)
            self.camera_axes.append(axis_line)

        # Create camera frustum
        frustum_length = 4.0

        # Calculate the width and height of the frustum at the far plane
        height = 2 * frustum_length * np.tan(np.radians(fov_v) / 2)
        width = height * aspect_ratio

        fov_h = 2 * np.arctan(np.tan(np.radians(fov_v) / 2) * aspect_ratio)
        print(f"fov_h: {np.degrees(fov_h)}")

        # Frustum corners in camera space (before transforming with the rotation/translation)
        frustum_corners = np.array([
            [-width / 2, -height / 2, frustum_length],  # Bottom-left
            [width / 2, -height / 2, frustum_length],   # Bottom-right
            [width / 2, height / 2, frustum_length],    # Top-right
            [-width / 2, height / 2, frustum_length],   # Top-left
        ])

        # Transform frustum corners to world space
        frustum_world = np.dot(rotation, frustum_corners.T).T + position

        # Draw lines from the camera position to each frustum corner
        for corner in frustum_world:
            frustum_line = np.array([position, corner])
            visuals.Line(pos=frustum_line, color=(0.8, 0.8, 0.8, 1), width=0.5, parent=self.view.scene)

        if self.display_image:
            # Create an image
            image_data = np.ones((2, 1, 4), dtype=np.uint8)  # Adding the alpha channel
            image_data[:, :, :3] = 255  # Set the RGB values to white (or any color)
            image_data[:, :, 3] = 128   # Set the alpha value (128 is 50% transparency)

            # image_data = np.ones((2, 1, 3), dtype=np.uint8)
            self.image = scene.visuals.Image(image_data, parent=self.view.scene, method='auto')

            frustum_length = 0.5

            # Calculate the width and height of the frustum at the far plane
            height = 2 * frustum_length * np.tan(np.radians(fov_v) / 2)
            width = height * aspect_ratio

            # Frustum corners in camera space (before transforming with the rotation/translation)
            frustum_corners = np.array([
                [-width / 2, -height / 2, frustum_length],  # Bottom-left
                [width / 2, -height / 2, frustum_length],   # Bottom-right
                [width / 2, height / 2, frustum_length],    # Top-right
                [-width / 2, height / 2, frustum_length],   # Top-left
            ])
            # Transform frustum corners to world space
            frustum_world = np.dot(rotation, frustum_corners.T).T + position

            for i in range(4):
                next_i = (i + 1) % 4
                visuals.Line(pos=np.array([frustum_world[i], frustum_world[next_i]]), color=(0.8, 0.8, 0.8, 1), parent=self.view.scene)

            far_plane_center = np.mean(frustum_world, axis=0)
            scale = frustum_length/self.camera_intrinsics[0,0]

            rot_transform = MatrixTransform()
            transform = np.eye(4)
            transform[:3,:3] = rotation
            transform[:3, 3] = np.dot(rotation, np.array([-self.img_width/2, -self.img_height/2, 0]))
            rot_transform.matrix = transform.T
            st_transform = STTransform(translate=far_plane_center, scale=(scale,scale,scale))

            self.image.transform = st_transform*rot_transform




    def update(self, points, frame=None):
        self.points = points
        self.frame = frame
        # alpha_channel = np.full((frame.shape[0], frame.shape[1], 1), 128, dtype=np.uint8)
        # self.frame = np.concatenate((frame, alpha_channel), axis=2)
        
        
    def on_timer(self, event):
        if self.points is not None and self.connections is not None:
            # Update scatter plot
            self.scatter.set_data(self.points, edge_color='blue', face_color='blue', size=10)

            # Update lines
            connects = np.array(self.connections)
            self.lines.set_data(pos=self.points, connect=connects)
        
        if self.auto_rotate and not self.user_interacting:
            self.view.camera.azimuth += self.rotation_speed
        
        if self.display_image and self.frame is not None:
            self.image.set_data(self.frame)

        # Ensure the canvas updates
        self.canvas.update()

    def close(self):
        # Stop the timer and close the canvas
        self.timer.stop()
        self.canvas.close()

    def run(self):
        app.run()

    def toggle_grid(self):
        self.grid.visible = not self.grid.visible
        self.canvas.update()

    def update_camera_pose(self, new_camera_pose):
        # Remove old camera axes
        if hasattr(self, 'camera_axes'):
            for axis in self.camera_axes:
                self.view.scene.remove(axis)

        # Add new camera visualization
        self.add_camera_visualization(new_camera_pose)
        self.canvas.update()