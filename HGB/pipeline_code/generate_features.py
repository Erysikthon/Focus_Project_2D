import pandas as pd
from py3r.behaviour.tracking.tracking import LoadOptions as opt
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.tracking.tracking_mv import TrackingMV
from natsort import natsorted



def triangulate(collection_path : str, 
                fps : int, 
                rescale_points : tuple[str], 
                rescale_distance : float, 
                filter_threshold : int = 0.9,
                construction_points : dict[str : dict["between_points" : tuple[str], "mouse_or_oft" : str]] = None,
                smoothing = True,
                smoothing_mouse = 3,
                smoothing_oft = 20):
    

    options = opt(fps=fps)
    tracking_collection = TrackingCollection.from_yolo3r_folder(collection_path,options, TrackingMV)

    # Likelihood filter

    tracking_collection.filter_likelihood(filter_threshold)

    #Triangulation

    triangulated_tracking_collection = tracking_collection.stereo_triangulate()
    triangulated_tracking_collection.strip_column_names()
    triangulated_tracking_collection.rescale_by_known_distance(rescale_points[0],rescale_points[1], rescale_distance, dims = ("x","y","z"))

    # Initialize smoothing

    smoothing_dict = {
        # mouse
        "nose": {"window": smoothing_mouse, "type": "mean"},
        "headcentre": {"window": smoothing_mouse, "type": "mean"},
        "neck": {"window": smoothing_mouse, "type": "mean"},
        "earl": {"window": smoothing_mouse, "type": "mean"},
        "earr": {"window": smoothing_mouse, "type": "mean"},
        "bodycentre": {"window": smoothing_mouse, "type": "mean"},
        "bcl": {"window": smoothing_mouse, "type": "mean"},
        "bcr": {"window": smoothing_mouse, "type": "mean"},
        "hipl": {"window": smoothing_mouse, "type": "mean"},
        "hipr": {"window": smoothing_mouse, "type": "mean"},
        "tailbase": {"window": smoothing_mouse, "type": "mean"},
        "tailcentre": {"window": smoothing_mouse, "type": "mean"},
        "tailtip": {"window": smoothing_mouse, "type": "mean"},

        # oft
        "tr": {"window": smoothing_oft, "type": "median"},
        "tl": {"window": smoothing_oft, "type": "median"},
        "br": {"window": smoothing_oft, "type": "median"},
        "bl": {"window": smoothing_oft, "type": "median"},
        "top_tr": {"window": smoothing_oft, "type": "median"},
        "top_tl": {"window": smoothing_oft, "type": "median"},
        "top_br": {"window": smoothing_oft, "type": "median"},
        "top_bl": {"window": smoothing_oft, "type": "median"},
    }
    if not construction_points==None:
        for handle in construction_points:
            construction_infos = construction_points[handle]
            triangulated_tracking_collection.construction_point(handle,construction_infos["between_points"],dims=("x","y","z"))
            if construction_infos["mouse_or_oft"] == "mouse":
                smoothing_dict[handle] = {"window": smoothing_mouse, "type": "mean"}
            elif construction_infos["mouse_or_oft"] == "oft":
                smoothing_dict[handle] = {"window": smoothing_oft, "type": "median"}
            else:
                raise ValueError(f"{construction_infos['mouse_or_oft']} only accepts 'mouse' or 'oft' as values"  )
            print(f"Created construction point {handle} between {construction_infos['between_points']} as {construction_infos['mouse_or_oft']} point")
        
    if smoothing:
        triangulated_tracking_collection.smooth(smoothing_dict)

    features_collection = FeaturesCollection.from_tracking_collection(triangulated_tracking_collection)

    return features_collection

    # Distance

def features(features_collection : FeaturesCollection, 
                      distance : dict[tuple : str] = [],
                      angle : dict[tuple[str] : str] = [],
                      speed : tuple[str] = [],
                      distance_to_boundary : tuple[str] = [],
                      is_point_recognized : tuple[str] = [],
                      volume : dict[tuple : tuple[tuple]] = [],
                      standard_deviation : tuple[str] = [],
                      f_b_fill = True,
                      embedding_length = list(range(0,1))
                      ):

    # Distance
    print("calculating distance...")

    for handle in distance:
        for dim in distance[handle]:
            features_collection.distance_on_axis(handle[0], handle[1], dim).store()

    # Azimuth / Angles
    print("calculating angles...")

    for handle in angle:
        radians_or_sincos : str = angle[handle]
        if radians_or_sincos == "radians":
            features_collection.angle(handle[0],handle[1],handle[2],handle[3],plane=("x","y")).store()
            features_collection.angle(handle[0],handle[1],handle[2],handle[3],plane=("y","z")).store()

        elif radians_or_sincos == "sincos":
            features_collection.sin_of_angle(handle[0],handle[1],handle[2],handle[3],plane=("x","y")).store()
            features_collection.cos_of_angle(handle[0],handle[1],handle[2],handle[3],plane=("x","y")).store()
            features_collection.sin_of_angle(handle[0],handle[1],handle[2],handle[3],plane=("y","z")).store()
            features_collection.cos_of_angle(handle[0],handle[1],handle[2],handle[3],plane=("y","z")).store()

        else:
            raise KeyError(f"only sincos or radians are accepted as argument of angles. You typed: {radians_or_sincos}")

    # Speed
    print("calculating speed...")

    for point in speed:
        features_collection.speed(point, dims=("x","y","z")).store()
    
    # is it BALL?
    print("calculating ball...")

    for point in is_point_recognized:
        features_collection.is_recognized(point).store()

    #Distances to boundary
    print("calculating distance to boundary...")

    for point in distance_to_boundary:
        features_collection.distance_to_boundary_dynamic(point, ["tl", "tr", "bl", "br"], "oft").store()

    #Volume
    print("calculating volume...")

    for handle in volume:
        faces : tuple[tuple] = volume[handle]
        features_collection.volume(points = handle, faces = faces).store()

    #Standard deviation
    print("calculating standard deviation...")

    for thing in standard_deviation:
        features_collection.standard_dev(thing).store()

    ############################################### Missing data handling

    if f_b_fill:
        print("Missing data filling (forward/backward)...")

        # Forward fill then backward fill missing data
        for file in features_collection.keys():
            feature_obj = features_collection[file]
            df = feature_obj.data

            # Forward fill, then backward fill remaining NAs
            df = df.ffill().bfill()

            feature_obj.data = df

    print("Embedding...")

    embedding = {}
    for column in features_collection[0].data.columns:
        embedding[column] =  list(embedding_length)
    features_collection = features_collection.embedding_df(embedding)

    # Extract features
    feature_dict = {}
    for handle in natsorted(features_collection):
        feature_obj = features_collection[handle]
        feature_dict[handle] = feature_obj

    combined_features = pd.concat(feature_dict.values(), keys=feature_dict.keys(), names=['video_id', 'frame'])

    return combined_features


def features_2d(features_collection : FeaturesCollection,
                      distance : dict[tuple : str] = [],
                      angle : dict[tuple[str] : str] = [],
                      speed : tuple[str] = [],
                      distance_to_boundary : tuple[str] = [],
                      is_point_recognized : tuple[str] = [],
                      f_b_fill = True,
                      embedding_length = list(range(0,1))
                      ):
    """
    2D version of features function - only calculates features in x,y plane
    (no z-coordinate, volume, or standard deviation features)
    """

    # Distance
    print("calculating distance...")

    for handle in distance:
        for dim in distance[handle]:
            features_collection.distance_on_axis(handle[0], handle[1], dim).store()

    # Azimuth / Angles (only x,y plane for 2D)
    print("calculating angles...")

    for handle in angle:
        radians_or_sincos : str = angle[handle]
        if radians_or_sincos == "radians":
            features_collection.angle(handle[0],handle[1],handle[2],handle[3],plane=("x","y")).store()

        elif radians_or_sincos == "sincos":
            features_collection.sin_of_angle(handle[0],handle[1],handle[2],handle[3],plane=("x","y")).store()
            features_collection.cos_of_angle(handle[0],handle[1],handle[2],handle[3],plane=("x","y")).store()

        else:
            raise KeyError(f"only sincos or radians are accepted as argument of angles. You typed: {radians_or_sincos}")

    # Speed (only x,y dims for 2D)
    print("calculating speed...")

    for point in speed:
        features_collection.speed(point, dims=("x","y")).store()

    # is it BALL?
    print("calculating ball...")

    for point in is_point_recognized:
        features_collection.is_recognized(point).store()

    #Distances to boundary
    print("calculating distance to boundary...")

    for point in distance_to_boundary:
        features_collection.distance_to_boundary_dynamic(point, ["tl", "tr", "bl", "br"], "oft").store()

    ############################################### Missing data handling

    if f_b_fill:
        print("Missing data filling (forward/backward)...")

        # Forward fill then backward fill missing data
        for file in features_collection.keys():
            feature_obj = features_collection[file]
            df = feature_obj.data

            # Forward fill, then backward fill remaining NAs
            df = df.ffill().bfill()

            feature_obj.data = df

    print("Embedding...")

    embedding = {}
    for column in features_collection[0].data.columns:
        embedding[column] =  list(embedding_length)
    features_collection = features_collection.embedding_df(embedding)

    # Extract features
    feature_dict = {}
    for handle in natsorted(features_collection):
        feature_obj = features_collection[handle]
        feature_dict[handle] = feature_obj

    combined_features = pd.concat(feature_dict.values(), keys=feature_dict.keys(), names=['video_id', 'frame'])

    return combined_features

