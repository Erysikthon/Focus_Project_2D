import numpy as np
import pandas as pd
from py3r.behaviour.features.features_collection import FeaturesCollection
from py3r.behaviour.features.boundary import DynamicBoundary
from py3r.behaviour.tracking.tracking_collection import TrackingCollection
from py3r.behaviour.tracking.tracking_mv import TrackingMV
from natsort import natsorted


def triangulate(collection_path: str,
                fps: int,
                rescale_points: tuple[str],
                rescale_distance: float,
                filter_threshold: float = 0.9,
                construction_points: dict = None,
                smoothing: bool = True,
                smoothing_mouse: int = 3,
                smoothing_oft: int = 20):

    tracking_collection = TrackingCollection.from_yolo3r_folder(collection_path, fps=fps, tracking_cls=TrackingMV)

    for _, tracking_mv in tracking_collection._obj_dict.items():
        tracking_mv.filter_likelihood(filter_threshold)

    triangulated_tracking_collection = tracking_collection.stereo_triangulate()
    triangulated_tracking_collection.each.strip_column_names()
    triangulated_tracking_collection.each.rescale_by_known_distance(
        rescale_points[0], rescale_points[1], rescale_distance, dims=("x", "y", "z")
    )

    oft_added = []
    mouse_added = []
    if construction_points is not None:
        for handle, construction_infos in construction_points.items():
            between_points = construction_infos["between_points"]
            for video_id, tracking in triangulated_tracking_collection._obj_dict.items():
                df = tracking.data
                for dim in ('x', 'y', 'z'):
                    cols = [f'{pt}.{dim}' for pt in between_points if f'{pt}.{dim}' in df.columns]
                    if cols:
                        df[f'{handle}.{dim}'] = df[cols].mean(axis=1)
                tracking.data = df
            if construction_infos["mouse_or_oft"] == "mouse":
                mouse_added.append(handle)
            elif construction_infos["mouse_or_oft"] == "oft":
                oft_added.append(handle)
            else:
                raise ValueError(
                    f"{construction_infos['mouse_or_oft']} only accepts 'mouse' or 'oft' as values"
                )
            print(f"Created construction point {handle} between {between_points} as {construction_infos['mouse_or_oft']} point")

    if smoothing:
        oft_points = ["tr", "tl", "br", "bl", "top_tr", "top_tl", "top_br", "top_bl"] + oft_added
        triangulated_tracking_collection.each.smooth_all(
            window=smoothing_mouse,
            method="mean",
            overrides=[(oft_points, "median", smoothing_oft)],
            dims=("x", "y", "z"),
            strict=False
        )

    features_collection = FeaturesCollection.from_tracking_collection(triangulated_tracking_collection)

    return features_collection


def _angle_between_vectors(td, p0, p1, p2, p3, d0, d1):
    """Angle (radians, wrapped to [-pi, pi]) between vectors p0->p1 and p2->p3 in the d0-d1 plane."""
    az1 = np.arctan2(td[f'{p1}.{d1}'] - td[f'{p0}.{d1}'],
                     td[f'{p1}.{d0}'] - td[f'{p0}.{d0}'])
    az2 = np.arctan2(td[f'{p3}.{d1}'] - td[f'{p2}.{d1}'],
                     td[f'{p3}.{d0}'] - td[f'{p2}.{d0}'])
    return (az1 - az2 + np.pi) % (2 * np.pi) - np.pi


def _tetrahedron_volume(td, p0, p1, p2, p3):
    """Absolute volume of tetrahedron defined by 4 tracked points (per frame)."""
    def _col(p, dim):
        return td[f'{p}.{dim}'].values

    a = np.stack([_col(p1, 'x') - _col(p0, 'x'),
                  _col(p1, 'y') - _col(p0, 'y'),
                  _col(p1, 'z') - _col(p0, 'z')], axis=1)
    b = np.stack([_col(p2, 'x') - _col(p0, 'x'),
                  _col(p2, 'y') - _col(p0, 'y'),
                  _col(p2, 'z') - _col(p0, 'z')], axis=1)
    c = np.stack([_col(p3, 'x') - _col(p0, 'x'),
                  _col(p3, 'y') - _col(p0, 'y'),
                  _col(p3, 'z') - _col(p0, 'z')], axis=1)
    vol = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0
    return pd.Series(vol, index=td.index)


def features(features_collection: FeaturesCollection,
             distance: dict = {},
             angle: dict = {},
             speed: tuple = (),
             distance_to_boundary: tuple = (),
             is_point_recognized: tuple = (),
             volume: dict = {},
             standard_deviation: tuple = (),
             f_b_fill: bool = True,
             embedding_length=list(range(0, 1))):

    # Distance
    print("calculating distance...")
    for handle in distance:
        for dim in distance[handle]:
            features_collection.each.distance_between(handle[0], handle[1], dims=(dim,)).store()

    # Angles — computed manually to support arbitrary 4-point vectors in both xy and yz planes.
    # py3r's azimuth_deviation only handles 3-point (shared pivot) angles in the xy plane,
    # so we use arctan2 directly for full generality.
    print("calculating angles...")
    for handle in angle:
        radians_or_sincos: str = angle[handle]
        if radians_or_sincos not in ("radians", "sincos"):
            raise KeyError(f"only 'sincos' or 'radians' accepted for angles, got: {radians_or_sincos}")
        for file_key in features_collection.keys():
            feat = features_collection[file_key]
            td = feat.tracking.data
            for d0, d1 in [("x", "y"), ("y", "z")]:
                ang = _angle_between_vectors(td, handle[0], handle[1], handle[2], handle[3], d0, d1)
                name_base = f"angle_{handle[0]}_{handle[1]}_{handle[2]}_{handle[3]}_{d0}{d1}"
                if radians_or_sincos == "radians":
                    feat.store(ang, name_base)
                else:
                    feat.store(np.sin(ang).rename(f"sin_{name_base}"), f"sin_{name_base}")
                    feat.store(np.cos(ang).rename(f"cos_{name_base}"), f"cos_{name_base}")

    # Speed
    print("calculating speed...")
    for point in speed:
        features_collection.each.speed(point, dims=("x", "y", "z")).store()

    # Is point recognized (1 = tracked, 0 = missing)
    print("calculating recognition...")
    for point in is_point_recognized:
        for file_key in features_collection.keys():
            feat = features_collection[file_key]
            td = feat.tracking.data
            recognized = (~td[f'{point}.x'].isna()).astype(float)
            recognized.name = f"is_recognized_{point}"
            feat.store(recognized, f"is_recognized_{point}")

    # Distance to boundary — uses DynamicBoundary; replaces removed distance_to_boundary_dynamic
    print("calculating distance to boundary...")
    oft_boundary = DynamicBoundary(["tl", "tr", "bl", "br"])
    for point in distance_to_boundary:
        features_collection.each.distance_to_boundary(point, oft_boundary).store()

    # Volume of tetrahedra (faces parameter accepted for API compatibility but unused —
    # scalar triple product gives the exact same result for any consistent face ordering)
    print("calculating volume...")
    for handle in volume:
        col_name = f"Volume_of_{'_'.join(handle)}"
        for file_key in features_collection.keys():
            feat = features_collection[file_key]
            td = feat.tracking.data
            vol = _tetrahedron_volume(td, handle[0], handle[1], handle[2], handle[3])
            feat.store(vol, col_name)

    # Rolling standard deviation (window=15 frames); looks up column in stored features
    # first, then falls back to raw tracking coordinates (e.g. "headcentre.z")
    print("calculating standard deviation...")
    for col in standard_deviation:
        for file_key in features_collection.keys():
            feat = features_collection[file_key]
            if col in feat.data.columns:
                series = feat.data[col]
            elif col in feat.tracking.data.columns:
                series = feat.tracking.data[col]
            else:
                print(f"  Warning: column '{col}' not found, skipping std")
                continue
            rolled = series.rolling(window=15, min_periods=1, center=True).std().fillna(0)
            feat.store(rolled.rename(f"std_{col}"), f"std_{col}")

    # Missing data handling
    if f_b_fill:
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
