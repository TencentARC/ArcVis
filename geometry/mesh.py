# -*- coding: utf-8 -*-

import numpy as np


def get_verts_by_faces(verts, faces, vert_colors=None):
    """Rearrange verts in (V, 3) into (F, 3, 3) by getting the faces
    Verts_colors are map to faces, each use the mean color of three

    Args:
        verts: (V, 3) np array, verts adjusted by volume offset
        faces: (F, 3) np array, faces
        vert_colors: (V, 3) np array, rgb color in (0~1). optional

    Returns:
        verts_by_faces: (F, 3, 3) np array
        get_verts_by_faces: (F, 3), color of each face
    """
    verts_by_faces = np.take(verts, faces, axis=0)

    mean_face_colors = None
    if vert_colors is not None:
        mean_face_colors = np.take(vert_colors, faces, axis=0).mean(1)

    return verts_by_faces, mean_face_colors
