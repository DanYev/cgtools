import os
import sys
import copy
import time
import numpy as np

    
def get_distance(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.linalg.norm(v1 - v2) / 10.0  
    

def get_angle(v1, v2, v3):
    v1, v2, v3 = map(np.array, (v1, v2, v3))
    v1 = v1 - v2
    v2 = v3 - v2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees    


def get_dihedral(v1, v2, v3, v4):
    v1, v2, v3, v4 = map(np.array, (v1, v2, v3, v4))
    b1, b2, b3 = v2 - v1, v3 - v2, v4 - v3
    b2n = b2 / np.linalg.norm(b2)
    n1 = np.cross(b1, b2); n1 /= np.linalg.norm(n1)
    n2 = np.cross(b2, b3); n2 /= np.linalg.norm(n2)
    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, b2n), n2)
    return np.degrees(np.arctan2(y, x))
    
            
if __name__ == "__main__":
    pass
    

  
