import py3r.behaviour as p3b
import py3r.behaviour.animation.animation_stream
from style_distances import style

TRACKING_FOLDER = "./tracking"
VIDEO_PATH = "./videos/OFT_left_2.avi"
OUTPUT_PATH =  "./output/animation.mp4"
ALL_POINTS = ("nose", "earl", "earr", "headcentre", "neck", "bcl", "bcr", "bodycentre", "hipl", "hipr", "tailbase", "tailcentre", "tailtip")
DISTANCES = (
    ("bodycentre", "earl"),
    ("bodycentre", "earr"),
    ("bodycentre", "bcl"),
    ("bodycentre", "bcr"),
    ("bodycentre", "hipl"),
    ("bodycentre", "hipr"),
    ("bodycentre", "nose"),
    ("bodycentre", "tailtip"),
    )

tc = p3b.TrackingCollection.from_yolo3r_folder(folder_path = TRACKING_FOLDER, fps = 30)
tc.each.filter_likelihood(0.9)
tc.each.interpolate(limit=5)
tc.each.smooth_all(window=3, method="mean")
tc.each.strip_column_names()
tc.each.rescale_by_known_distance(
    point1="tr",
    point2="tl",
    distance_in_metres=0.64)
fc = tc.to_features()

distance_names = []
for i, distance in enumerate(DISTANCES):
    name = f"distance_{i}"
    fc.each.distance_between(distance[0], distance[1]).store()
    distance_names.append(name)
    
f : p3b.Features = fc[0]
f.data.columns = distance_names
f.data = f.data * 1000
print(f.data.columns)

s = f.animation_stream(points = ALL_POINTS, lines = DISTANCES, features = distance_names,
                       pixel_coords = 1, canvas_size = (1280, 1024), undo_meta_scaling = True , style = style)
s._line_keys = distance_names

s.play(video_path = VIDEO_PATH)
#s.save(video_path = VIDEO_PATH, out_path = OUTPUT_PATH)

