import numpy as np
import pandas as pd
import py3r.behaviour as p3b
from natsort import natsorted
import trimesh as tm

def triangulate(collection_path="./pipeline_inputs/collection"):
    tc = p3b.TrackingCollection.from_yolo3r_folder(collection_path, fps=30, tracking_cls=p3b.TrackingMV)
    tc = tc.stereo_triangulate()
    tc.each.filter_likelihood(0.9)
    tc.each.interpolate(limit=5)
    tc.each.smooth_all(window=3, method="mean")
    tc.each.strip_column_names()
    tc.each.rescale_by_known_distance(
        point1="tr",
        point2="tl",
        distance_in_metres=0.64)
    fc = tc.to_features()
    for handle in fc:
        feature : p3b.Features = fc[handle]
        get_mid_point_df(feature.tracking)
    return fc

def get_mid_point_df(tracking : p3b.Tracking):
    for dim in ("x", "y", "z"):
        temp_df = pd.DataFrame()
        for point in ("tl", "tr", "bl", "br"):
            temp_df = pd.concat([temp_df,tracking.data[point+"."+dim]],axis = 1)
        mean = temp_df.mean(axis=1,skipna=False)
        tracking.data["mid." + dim] = mean

def height(feature : p3b.Features, point: str) -> p3b.FeaturesResult:
    """
    returns distance from point1 to point2 along the given axis
    """
    
    p1 = feature.tracking.data["mid.z"]
    p2 = feature.tracking.data[point + ".z"]
    result=abs(p2-p1)
    name = f"Height_of_{point}"
    meta = {
        "function": "height",
        "point1": point
    }
    return p3b.FeaturesResult(result, feature, name, meta)

def aziz(feature, point1, point2, plane):
        _1x = feature.tracking.data[point1 + "." + plane[0]]
        _1y = feature.tracking.data[point1 + "." + plane[1]]
        _2x = feature.tracking.data[point2 + "." + plane[0]]
        _2y = feature.tracking.data[point2 + "." + plane[1]]
        result = np.arctan2((_2y - _1y), (_2x - _1x))
        return result

def angle(feature : p3b.Features, point1: str, point2: str, point3: str, point4: str, plane = ("x", "y")) -> p3b.FeaturesResult:
    """
    output between -pi and pi, measures angle from first line to second line
    this is the angle between the projections of the vectors in the given plane
    WARNING: be mindful of input order. vectors are oriented from first to second point
    """
    
    a1 = aziz(feature, point1, point2, plane)
    a2 = aziz(feature, point3, point4, plane)
    agl = (a1 - a2 + np.pi) % (2 * np.pi) - np.pi
    name = f"angle_from_{point1}_{point2}_to_{point3}_{point4}_in_{plane[0]}{plane[1]}"
    meta = {
        "function": "angle",
        "point1": point1,
        "point2": point2,
        "point3": point3,
        "point4": point4,
        "dims" : plane
    }
    return p3b.FeaturesResult(agl, feature, name, meta)

def is_recognized(feature, point: str) -> p3b.FeaturesResult:
    """
    because sometimes no information is the best information
    """

    len = feature.tracking.data[point + ".x"].shape[0]
    result = [None] * len
    for i in range(0, len):
        result[i] = not np.isnan(feature.tracking.data[point + ".x"][i])
    name = f"{point}_recognized"
    meta = {
        "function": "is_recognized",
        "point": point
    }
    return p3b.FeaturesResult(result, feature, name, meta)


def volume(feature,points : list[str], faces : list[list[int]]) -> p3b.FeaturesResult:
    """
    Points are a list of defined points. The order of the points matters.
    The first point entered will be vertex number 0, the second 1 and so on.
    Faces are a list of list of int which define a triangular face.
    The vertices that create a face must be called in counterclockwise sense in order to have a positive volume.
    ex.
    fc.volume(points = ["headcentre","nose","earl","earr"],faces = [[0,1,2],[0,2,3],[0,3,1],[1,3,2]])
    """
    volumedf = []
    frames = feature.tracking.data[points[0] + ".x"].shape[0]
    for frame in range(0,frames):
        vertices = []
        isna = False
        for point in points:

            x = feature.tracking.data[point + ".x"][frame]
            y = feature.tracking.data[point + ".y"][frame]
            z = feature.tracking.data[point + ".z"][frame]

            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                isna = True
            vertex = [x,y,z]
            vertices.append(vertex)
        if not isna:
            vol = tm.Trimesh(vertices=vertices,faces=faces)
            if not vol.is_watertight:
                raise BrokenPipeError(f"volume in frame: {frame} isn't watertight.")
            volumedf.append(vol.volume)
        else:
            volumedf.append(None)

    name = f"Volume_of"
    meta = {
        "function": "Volume",
        "faces" : faces
    }
    i = 1
    for point in points:
        name += "_" + point
        meta["point"+"_"+str(i)] = point
        i+=1
    
    return p3b.FeaturesResult(volumedf, feature, name, meta)

def standard_dev(feature, point:str) -> p3b.FeaturesResult:
    if point in feature.data.columns:
        data = feature.data[point]
    else:
        data = feature.tracking.data[point]
    window = 30
    roller = data.rolling(window, min_periods=5).std()
    name = f"standard_deviation_{point}"
    meta = {"function": "standard_deviation", "point": point}
    return p3b.FeaturesResult(roller, feature, name, meta)


def features(features_collection : p3b.FeaturesCollection, embedding_length : list = list(range(-15, 16, 3))):

    print("calculating distance...")
    distance=[  ("neck", "earl"),
                ("neck", "earr"),
                ("neck", "bcl"),
                ("neck", "bcr"),
                ("bcl", "hipl"),
                ("bcr", "hipr"),
                ("hipl", "tailbase"),
                ("hipr", "tailbase"),
                ("headcentre", "neck"),
                ("neck", "bodycentre"),
                ("bodycentre", "tailbase"),
                ("headcentre", "earl"),
                ("headcentre", "earr"),
                ("bodycentre", "bcl"),
                ("bodycentre", "bcr"),
                ("bodycentre", "hipl"),
                ("bodycentre", "hipr")
                ]
    for handle in distance:
        features_collection.each.distance_between(handle[0], handle[1], dims=("x", "y")).store()
        features_collection.each.distance_between(handle[0], handle[1], dims=("y", "z")).store()
    print("calculating heights...")
    height_p = [("headcentre"),
                ("earl"),
                ("earr"),
                ("neck"),
                ("bcl"),
                ("bcr"),
                ("bodycentre"),
                ("hipl"),
                ("hipr"),
                ("tailcentre")]
    
    for point in height_p:
        for handle in features_collection:
            feature : p3b.Features = features_collection[handle]
            height(feature, point).store()

    print("calculating angles...")
    angles=[("bodycentre", "neck", "neck", "headcentre"),
            ("bodycentre", "neck", "neck", "earl"),
            ("bodycentre", "neck", "neck", "earr"),
            ("tailbase", "bodycentre", "bodycentre", "neck"),
            ("tailbase", "bodycentre", "tailbase", "hipl"),
            ("tailbase", "bodycentre", "tailbase", "hipr"),
            ("tailbase", "bodycentre", "hipl", "bcl"),
            ("tailbase", "bodycentre", "hipr", "bcr"),
            ("bodycentre", "tailbase", "tailbase", "tailcentre"),
            ("bodycentre", "tailbase", "tailcentre", "tailtip")
    ]

    for plane in (("x", "y"), ("y", "z")):
        for points in angles:
            for handle in features_collection:
                feature : p3b.Features = features_collection[handle]
                angle(feature, points[0], points[1], points[2], points[3], plane).store()

    print("calculating speeds...")

    speeds=("headcentre",
        "earl",
        "earr",
        "neck",
        "bcl",
        "bcr",
        "bodycentre",
        "hipl",
        "hipr",
        "tailcentre"
        )

    for point in speeds:
        features_collection.each.speed(point, dims=("x", "y", "z")).store()

    print("calculating distance to boundary...")

    distance_to_boundary=("headcentre",
                            "earl",
                            "earr",
                            "neck",
                            "bcl",
                            "bcr",
                            "bodycentre",
                            "hipl",
                            "hipr",
                            "tailcentre"
                            )
    
    oft_boundary = p3b.features.boundary.DynamicBoundary(["tl", "tr", "bl", "br"])
    for point in distance_to_boundary:
        features_collection.each.distance_to_boundary(point, oft_boundary).store()

    print("calculating recognition...")
    is_point_recognized=(["nose"])
    for point in is_point_recognized:
        for handle in features_collection:
            feature : p3b.Features = features_collection[handle]
            is_recognized(feature, point).store()
    
    print("calculating volumes...")
    volumes={
        ("neck", "bodycentre", "bcl", "bcr"): ((0, 1, 2), (2, 1, 3), (0, 3, 1), (0, 2, 3)),
        ("bodycentre", "hipl", "tailbase", "hipr"): ((0, 3, 2), (3, 1, 2), (0, 2, 1),
                                                    (0, 1, 3)),
        ("neck", "bcl", "hipl", "bodycentre"): ((0, 1, 3), (1, 2, 3), (3, 2, 0), (0, 2, 1)),
        ("neck", "bcr", "hipr", "bodycentre"): ((0, 3, 1), (1, 3, 2), (3, 0, 2), (0, 1, 2))
        }
    
    for ba in volumes:
        for handle in features_collection:
            feature : p3b.Features = features_collection[handle]
            faces: tuple[tuple] = volumes[ba]
            volume(feature, points=ba, faces=faces).store()
    
        # Standard deviation
    print("calculating standard deviation...")

    standard_deviations=("headcentre.z",
                    "earl.z",
                    "earr.z",
                    "bodycentre.z",
                    "Volume_of_neck_bodycentre_bcl_bcr",
                    "Volume_of_bodycentre_hipl_tailbase_hipr",
                    "Volume_of_neck_bcl_hipl_bodycentre",
                    "Volume_of_neck_bcr_hipr_bodycentre"
                    )

    for thing in standard_deviations:
        for handle in features_collection:
            feature : p3b.Features = features_collection[handle]
            standard_dev(feature, thing).store()

    print("Missing data filling (forward/backward)...")
    for file in features_collection.keys():
        feature_obj = features_collection[file]
        df = feature_obj.data
        df = df.ffill().bfill()
        feature_obj.data = df

    # Embedding — embedding_df lives on Features (not FeaturesCollection) and returns a DataFrame
    print("Embedding...")
    embedding = {col: list(embedding_length) for col in features_collection[0].data.columns}

    feature_dict = {}
    for handle in natsorted(features_collection):
        feature_dict[handle] = features_collection[handle].embedding_df(embedding)

    return pd.concat(feature_dict.values(), keys=feature_dict.keys(), names=['video_id', 'frame'])

