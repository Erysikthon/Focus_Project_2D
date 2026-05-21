reliable_mousepoints = ("nose", "bodycentre")

COLORS = [
    [136,34,51],
    [238, 204, 136],
    [153,170,68],
    [51, 119, 17],
    [51, 153, 153],
    [119, 204, 221],
    [119,102,204],
    [85,34,136],
    [153,68,170],
    [238,51,119],
    [170,221,204],
]

area_dict = {f"area_{i}" : {"fill_color" : COLORS[i]} for i in range(0, 10)}
text_dict = {f"area_{i}" : {"color" : COLORS[i]} for i in range(0, 10)}

style = {
    "boundaries" : {
        "default" : {
            "fill_color" : (68,119,170),
            "fill_alpha" : 0.8,
            "edge_color" : (123,123,123),
            "edge_width" : 1
        },
        **area_dict
    },
    "points" : {
        "default" : {
            "color" : (221, 221, 221)
        }
    },
    "lines" : {
        "default" : {
            "color" : (255, 255, 255)
        }
    },
    "text": {
        "default" : {
            "color": (255, 255, 255),
            "font_scale": 0.5,
            "thickness": 1,
            "outline_color": (0, 0, 0),
            "outline_thickness": 2,
            "format": ".0f",
            "origin" : (100, 100)
            },
        **text_dict
    }
}