import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import re
import cv2

# --------------------------------------------------
# 1. Load traffic-light bounding boxes
# --------------------------------------------------

boxes = pd.read_csv("bbox_light.csv")

xyz_folder = Path("xyz")
xyz_files = sorted(xyz_folder.glob("*.npz"))

print("Depth files found:", len(xyz_files))


# --------------------------------------------------
# 2. Function to get a stable 3D traffic-light point
# --------------------------------------------------

def get_light_xyz(xyz, u, v, radius=5):
    """
    Look at a small square around the center of the traffic light
    and return the median X,Y,Z of all valid depth points.
    """

    height, width = xyz.shape[:2]

    x_start = max(0, u - radius)
    x_end = min(width, u + radius + 1)

    y_start = max(0, v - radius)
    y_end = min(height, v + radius + 1)

    patch = xyz[y_start:y_end, x_start:x_end, :3]

    # Flatten the patch into a list of XYZ points
    points = patch.reshape(-1, 3)

    # Keep only finite points
    valid = np.all(np.isfinite(points), axis=1)

    points = points[valid]

    if len(points) == 0:
        return None

    # Median is more resistant to noisy depth values
    return np.median(points, axis=0)


# --------------------------------------------------
# 3. Go through every available depth frame
# --------------------------------------------------

frame_numbers = []
light_positions = []

for file in xyz_files:

    # Example filename:
    # depth000049.npz
    #
    # Extract 49 from the name.
    match = re.search(r"(\d+)", file.stem)

    if match is None:
        continue

    frame_number = int(match.group(1))

    # Find matching bounding box row
    matching_rows = boxes[boxes["frame"] == frame_number]

    if len(matching_rows) == 0:
        continue

    row = matching_rows.iloc[0]

    x1 = int(row["x1"])
    y1 = int(row["y1"])
    x2 = int(row["x2"])
    y2 = int(row["y2"])

    # A box of all zeros means no usable traffic-light detection
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        continue

    # Center of bounding box
    u = int((x1 + x2) / 2)
    v = int((y1 + y2) / 2)

    # Load XYZ data
    data = np.load(file)
    xyz = data["xyz"]

    # Get stable XYZ estimate
    point = get_light_xyz(xyz, u, v)

    if point is None:
        continue

    X, Y, Z = point

    frame_numbers.append(frame_number)
    light_positions.append([X, Y, Z])


# Convert list to NumPy array
light_positions = np.array(light_positions)

print()
print("Valid traffic-light measurements:", len(light_positions))

if len(light_positions) == 0:
    raise RuntimeError("No valid traffic-light measurements were found.")


# --------------------------------------------------
# 4. Convert camera coordinates to our ground plane
# --------------------------------------------------

# Camera:
# +X = forward
# +Y = right
#
# World:
# +X = toward traffic light at the initial time
# +Y = left
#
# Therefore camera-right becomes negative world-left.

light_ground = np.column_stack((
    light_positions[:, 0],
    -light_positions[:, 1]
))


# --------------------------------------------------
# 5. Rotate so INITIAL car-to-light direction is +X
# --------------------------------------------------

initial_light = light_ground[0]

initial_angle = np.arctan2(
    initial_light[1],
    initial_light[0]
)

cos_a = np.cos(-initial_angle)
sin_a = np.sin(-initial_angle)

rotation = np.array([
    [cos_a, -sin_a],
    [sin_a,  cos_a]
])

light_world = (rotation @ light_ground.T).T


# --------------------------------------------------
# 6. Traffic light is world origin.
#
# light_world tells us:
# "where is the light relative to the car?"
#
# Therefore the CAR relative to the light is the
# negative of that vector.
# --------------------------------------------------

car_world = -light_world

car_x = car_world[:, 0]
car_y = car_world[:, 1]


# --------------------------------------------------
# 7. Print a few results
# --------------------------------------------------

print()
print("First five estimated car positions:")

for i in range(min(5, len(car_world))):
    print(
        "frame",
        frame_numbers[i],
        "X =",
        round(car_x[i], 2),
        "Y =",
        round(car_y[i], 2)
    )


# --------------------------------------------------
# 8. Plot trajectory
# --------------------------------------------------

plt.figure(figsize=(8, 8))

plt.plot(
    car_x,
    car_y,
    "o-",
    markersize=3,
    linewidth=1,
    label="Ego Vehicle"
)

# Traffic light is the origin
plt.scatter(
    0,
    0,
    marker="*",
    s=200,
    label="Traffic Light"
)

# Mark starting position
plt.scatter(
    car_x[0],
    car_y[0],
    s=80,
    label="Start"
)

# Mark ending position
plt.scatter(
    car_x[-1],
    car_y[-1],
    s=80,
    label="End"
)

plt.xlabel("World X (meters)")
plt.ylabel("World Y (meters)")
plt.title("Estimated Ego-Vehicle Trajectory")

plt.grid(True)
plt.axis("equal")
plt.legend()

plt.tight_layout()

plt.savefig(
    "trajectory.png",
    dpi=200
)

print()
print("Saved trajectory.png")

plt.close()


# --------------------------------------------------
# 9. Create animated trajectory video
# --------------------------------------------------

print("Creating trajectory.mp4...")

video_width = 800
video_height = 800

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video = cv2.VideoWriter(
    "trajectory.mp4",
    fourcc,
    20,
    (video_width, video_height)
)

# Keep the same limits for every frame
margin = 3

x_min = min(np.min(car_x), 0) - margin
x_max = max(np.max(car_x), 0) + margin

y_min = min(np.min(car_y), 0) - margin
y_max = max(np.max(car_y), 0) + margin


for i in range(len(car_x)):

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    # Draw trajectory up through the current frame
    ax.plot(
        car_x[:i+1],
        car_y[:i+1],
        "o-",
        markersize=3,
        linewidth=1,
        label="Ego Vehicle"
    )

    # Traffic light
    ax.scatter(
        0,
        0,
        marker="*",
        s=200,
        label="Traffic Light"
    )

    # Start
    ax.scatter(
        car_x[0],
        car_y[0],
        s=80,
        label="Start"
    )

    # Current vehicle position
    ax.scatter(
        car_x[i],
        car_y[i],
        s=100,
        label="Current Position"
    )

    ax.set_xlabel("World X (meters)")
    ax.set_ylabel("World Y (meters)")
    ax.set_title(
        f"Estimated Ego-Vehicle Trajectory - Frame {frame_numbers[i]}"
    )

    ax.grid(True)
    ax.legend()

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    # Convert matplotlib image into an OpenCV frame
    fig.canvas.draw()

    image = np.asarray(fig.canvas.buffer_rgba())

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGBA2BGR
    )

    # Guarantee correct video size
    image = cv2.resize(
        image,
        (video_width, video_height)
    )

    video.write(image)

    plt.close(fig)


video.release()

print("Saved trajectory.mp4")
print()
print("DONE!")