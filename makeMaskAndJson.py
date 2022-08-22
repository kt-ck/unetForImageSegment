import pathlib
from skimage.io import imread
import sys
import cv2
import numpy as np
import json
root = __file__ + "/../image"
area = 3506
image_name = "0004_0004_1367955722 (200).png"
block = "right"
ratio = 10
# col = True
col = False
lines = [
    [
242,158,240,159,238,161,236,162,235,162,234,164,233,164,232,165,230,166,229,166,226,167,226,169,226,170,225,170,223,172,222,173,220,173,219,174,218,174,216,175,215,175,213,176,212,176,211,176,210,177,210,178,210,179,209,180,208,182,208,183,207,183,206,184,205,186,204,187,204,188,203,190,202,191,202,192,202,193,201,195,199,196,199,197,199,198,198,198,197,199,196,200,194,202,194,203,192,205,190,205,189,206,189,206,187,207,186,207,186,207,184,208,182,209,181,210,178,212,178,213,178,213,177,214,177,215,176,216,175,217,174,218,172,220,171,221,171,222,171,222,170,223,169,224,169,225,168,225,167,226,167,227,167,228,167,229,167,230,167,230,167,231,166,232,166,233,166,234,166,235,166,236,166,237,166,238,166,238,166,239,166,240,166,241,166,243,166,244,166,245,166,246,166,246,166,247,166,248,166,249,166,250,166,251,166,252,166,254,166,254,166,256,166,257,166,259,167,260,167,262,168,262,168,263,168,264,169,265,169,266,169,267,170,269,170,270,170,270,170,270,170,271,171,272,172,273,172,275,173,277,174,277,174,278,175,278,175,279,176,280,177,281,178,282,178,283,179,284,180,285,181,285,181,286,182,286,183,287,185,288,186,288,186,289,187,289,188,289,190,289,191,289,192,289,193,289,194,289,194,289,197,289,198,289,200,290,201,290,202,290,203,291,205,292,206,292,207,292,208,292,209,292,210,292,210,292,211,292,212,292,213,292,214,292,215,292,216,292,217,292,218,292,218,292,219,292,220,292,221,292,222,292,223,292,224,292,226,292,226,291,226,290,226,289,226,288,226,287,226,286,226,286,226,285,226,284,226,283,226,282,226,281,226,280,226,279,226,278,226,277,226,276,226,275,226,274,226,273,226,272,226,271,226,270,226,269,227,267,228,266,228,265,229,263,230,262,230,261,231,259,231,258,231,257,231,256,231,255,231,254,231,254,231,253,231,252,231,251,231,250,231,249,231,248,231,247,231,246,231,246,231,245,231,243,231,242,231,241,231,240,231,239,231,238,231,238,231,236,232,235,232,233,233,231,233,230,233,230,233,229,233,228,233,227,233,226,233,225,233,224,233,223,233,222,233,221,233,220,233,218,233,217,233,216,234,215,234,214,234,214,234,213,234,212,234,211,234,210,234,209,234,208,234,208,235,207,236,206,237,205,237,204,238,202,240,202,241,200,241,199,242,198,242,198,243,198,245,198,246,198,247,198,248,198,249,198,252,196,253,195,254,194,255,194,256,193,257,192,257,191,258,190,258,189,258,189,259,187,259,186,260,185,261,184,262,183,262,182,263,180,263,179,263,178,263,177,263,176,263,175,263,174,263,174,263,173,263,172,263,171,263,170,263,169,262,168,261,166,261,166,260,165,259,164,258,164,258,163,258,162,257,162,257,161,256,160,255,159,253,159,252,159,251,158,250,158,249,158,248,158,247,158,246,158,245,158,244,158,243,158,242,158

]]


def line2Point(line):
    return [(line[i], line[i+1]) for i in range(0, len(line), 2)]


points = line2Point(lines[0])

if col:
    points.sort(key=lambda x: (x[0], x[1]))

    last = (-1, -1)

    ref = {"start": points[0], "end": points[len(
        points)-1], "filename": image_name, "area": area}

    for i in range(len(points)):
        if points[i][0] > last[0]:
            ref[points[i][0]] = {"begin": points[i][1], "end": points[i][1]}
        elif points[i][0] == last[0]:
            ref[points[i][0]]["end"] = points[i][1]

        last = points[i]

    img = cv2.imread(root + "/{}".format(image_name), 0)
    mask = np.zeros_like(img)

    for i in range(ref["start"][0], ref["end"][0] + 1):
        if i in ref and ref[i]["begin"] < ref[i]["end"] and (i - 1 not in ref or (1/ratio <= (ref[i]["end"] - ref[i]["begin"]) / (ref[i - 1]["end"] - ref[i - 1]["begin"]) <= ratio)):
            # print(ref[i]["begin"], ref[i]["end"]+1)
            mask[ref[i]["begin"]:ref[i]["end"]+1, i] = 255
        else:
            # print(ref[i-1])
            mask[ref[i-1]["begin"]:ref[i-1]["end"]+1, i] = 255
            ref[i] = ref[i-1]
else:
    points.sort(key=lambda x: (x[1], x[0]))
    # print(points)
    last = (-1, -1)

    ref = {"start": points[0], "end": points[len(
        points)-1], "filename": image_name, "area": area}

    for i in range(len(points)):
        if points[i][1] > last[1]:
            ref[points[i][1]] = {"begin": points[i][0], "end": points[i][0]}
        elif points[i][1] == last[1]:
            ref[points[i][1]]["end"] = points[i][0]

        last = points[i]
    # print(ref)
    img = cv2.imread(root + "/{}".format(image_name), 0)
    mask = np.zeros_like(img)

    for i in range(ref["start"][1], ref["end"][1] + 1):
        if i == ref["start"][1] and ref[i]["begin"] == ref[i]["end"]:
            ref[i]["end"] += 1
            mask[i, ref[i]["begin"]: ref[i]["end"]+1] = 255 
        elif i in ref and ref[i]["begin"] < ref[i]["end"] and (i - 1 not in ref or (1/ratio <= (ref[i]["end"] - ref[i]["begin"]) / (ref[i - 1]["end"] - ref[i - 1]["begin"]) <= ratio)):
            # print(i, ref[i]["begin"], ref[i]["end"]+1)
            mask[i,ref[i]["begin"]:ref[i]["end"]+1] = 255
        else:
            # print(ref[i-1])
            # print(i)
            mask[i, ref[i-1]["begin"]:ref[i-1]["end"]+1] = 255
            ref[i] = ref[i-1]
ref_json = json.dumps(ref)

with open("json/{}/{}.json".format(block,image_name.split(".")[0]), 'w') as f:
    f.write(ref_json)
    f.close()

cv2.imwrite("mask/{}/{}".format(block,image_name), mask)


# def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
#     """Returns a list of files in a directory/path. Uses pathlib."""
#     filenames = [file for file in path.glob(ext) if file.is_file()]
#     return filenames


# frame = cv2.namedWindow("frame")
# image_name = get_filenames_of_path(root / 'image')

# for each in image_name:
#     image = imread(each)

#     cv2.imshow("frame", image)
