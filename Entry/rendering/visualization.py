"""EDEN-style translucent block volumes, outlines and interval nodes."""
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


def marker(namespace, identifier, kind, stamp):
    value = Marker(header=Header(frame_id='map', stamp=stamp), ns=namespace, id=identifier, type=kind, action=Marker.ADD)
    value.pose.orientation.w = 1.
    return value


def point(xyz):
    return Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))


def block_markers(blocks, size, stamp):
    result = MarkerArray()
    size = np.asarray(size)
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    edges = [(a, b) for a in range(8) for b in range(a+1, 8) if np.abs(corners[a]-corners[b]).sum() == 1]
    for row in blocks:
        lower = np.asarray(row['key']) * size
        volume = marker('blocks', 2*row['marker_id'], Marker.CUBE, stamp)
        volume.pose.position = point(lower + size/2)
        volume.scale.x, volume.scale.y, volume.scale.z = map(float, size)
        volume.color.r, volume.color.g, volume.color.b, volume.color.a = 0., 229/255., 238/255., .125
        result.markers.append(volume)
        outline = marker('blocks', 2*row['marker_id']+1, Marker.LINE_LIST, stamp)
        outline.scale.x = .02
        outline.color.r, outline.color.g, outline.color.b, outline.color.a = 0., 229/255., 238/255., .55
        xyz = lower + corners*size
        outline.points = [point(xyz[i]) for edge in edges for i in edge]
        result.markers.append(outline)
    return result


def node_markers(nodes, stamp):
    result = MarkerArray()
    lines = marker('traversal', 0, Marker.LINE_LIST, stamp)
    lines.scale.x = .035
    lines.color.r, lines.color.g, lines.color.b, lines.color.a = 1., .65, .1, 1.
    for i, node in enumerate(nodes):
        sphere = marker('nodes', i, Marker.SPHERE, stamp)
        sphere.pose.position = point(node['position'])
        sphere.scale.x = sphere.scale.y = sphere.scale.z = .18
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 1., .65, .1, 1.
        label = marker('labels', i, Marker.TEXT_VIEW_FACING, stamp)
        label.pose.position = point(np.asarray(node['position']) + [0, 0, .3])
        label.scale.z = .22
        label.color.r = label.color.g = label.color.b = label.color.a = 1.
        label.text = node['id']
        result.markers.extend((sphere, label))
        if i:
            lines.points.extend((point(nodes[i-1]['position']), point(node['position'])))
    result.markers.append(lines)
    return result
