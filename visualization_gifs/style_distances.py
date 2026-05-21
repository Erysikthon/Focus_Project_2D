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

line_dict = {
    f"distance_{i}" : {
        "color" : {
            "from" : f"distance_{i}", 
            "cmap" : "viridis"
        }
    } for i in range(0, 8)
}
text_dict = {f"area_{i}" : {"color" : COLORS[i]} for i in range(0, 10)}

style = {
    "points" : {
        "default" : {
            "color" : {"from" : "distance_0", "cmap" : "viridis"},
        }
    },
    "lines" : {
        "default" : {
            "color" : {"from" : "distance_0", "cmap" : "viridis"},
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