import py3r.behaviour as p3b
import py3r.behaviour.animation.animation_stream
from style_area import style

TRACKING_FOLDER = "./tracking"
VIDEO_PATH = "./videos/OFT_left_2.avi"
OUTPUT_PATH =  "./output/animation.mp4"
ALL_POINTS = ("nose", "earl", "earr", "headcentre", "neck", "bcl", "bcr", "bodycentre", "hipl", "hipr", "tailbase", "tailcentre", "tailtip")
AREAS = (
    ("nose", "earl", "headcentre"),
    ("nose", "headcentre", "earr"),
    ("earl", "headcentre", "neck"),
    ("earr", "headcentre", "neck"),
    ("bcl", "neck", "bodycentre"),
    ("bcr", "neck", "bodycentre"),
    ("bcl", "bodycentre", "hipl"),
    ("bcr", "bodycentre", "hipr"),
    ("hipl", "bodycentre", "tailbase"),
    ("hipr", "bodycentre", "tailbase"),
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

area_names = []
for i, area in enumerate(AREAS):
    name = f"area_{i}"
    fc.each.define_dynamic_boundary(area, name = name)
    fc.each.area_of_boundary(name).store()
    area_names.append(name)
    
f : p3b.Features = fc[0]
f.data.columns = area_names
f.data = f.data * 100000

print(f.data.columns)

s = f.animation_stream(points = ALL_POINTS, boundaries = area_names, 
                       features = area_names,
                       pixel_coords = 1, canvas_size = (1280, 1024), undo_meta_scaling = True , style = style)
s.play(video_path = VIDEO_PATH)
s.save(video_path = VIDEO_PATH, out_path = OUTPUT_PATH)

