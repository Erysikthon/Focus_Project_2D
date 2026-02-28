import pandas as pd
#from py3r.behaviour.tracking.tracking import LoadOptions as opt
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.tracking.tracking_mv import TrackingMV
from py3r.behaviour.features.boundary import DynamicBoundary
from natsort import natsorted


def features_2d(features_collection : FeaturesCollection,
                      distance : dict[tuple : str] = [],
                      azimuth_deviation : dict[str] = [],
                      azimuth : dict[str : str] = [],
                      speed : tuple[str] = [],
                      distance_change : dict[str] = [],
                      area_of_boundary : dict[str] = [],
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
        dims = distance[handle]
        features_collection.each.distance_between(handle[0], handle[1], dims=dims).store()

    # Azimuth / Angles (only x,y plane for 2D)
    print("calculating angles and azimuths...")

    for handle in azimuth_deviation:
        features_collection.each.azimuth_deviation(*handle).store() #pretty sure this uses x, y by default

    for handle in azimuth:
        features_collection.each.azimuth(*handle).store()



    # Speed (only x,y dims for 2D)
    print("calculating speed...")

    for point in speed:
        features_collection.each.speed(point, dims=("x","y")).store()

    # Absolute position change from last frame (also x, y)
    print("calculating movement...")

    for point in distance_change:
        features_collection.each.distance_change(point, dims=("x","y")).store()

    #Areas

    print("calculating areas...")

    for boundary_points in area_of_boundary:
        # Create a DynamicBoundary from the points tuple
        boundary_name = "_".join(boundary_points)
        boundary = DynamicBoundary(list(boundary_points))
        features_collection.each.area_of_boundary(boundary).store()

    #Distances to boundary
    print("calculating distance to boundary...")

    for point in distance_to_boundary:
        # Create a DynamicBoundary for the OFT corners
        boundary = DynamicBoundary(["tl", "tr", "bl", "br"])
        features_collection.each.distance_to_boundary(point, boundary).store()

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
    features_collection.each.embedding_df(embedding)

    # Extract features
    feature_dict = {}
    for handle in natsorted(features_collection):
        feature_obj = features_collection[handle]
        feature_dict[handle] = feature_obj.data

    combined_features = pd.concat(feature_dict.values(), keys=feature_dict.keys(), names=['video_id', 'frame'])

    return combined_features

