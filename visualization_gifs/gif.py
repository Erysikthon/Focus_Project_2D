import py3r.behaviour as p3b

TRACKING_FOLDER = "./tracking"

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
fc.each.area_of_boundary()
print(fc[0].data)

