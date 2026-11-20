"""Continuous route contracts and measured-arrival state, independent of ROS."""
from pathlib import Path
import hashlib
import numpy as np


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            result.update(chunk)
    return result.hexdigest()


def check_support_path(path, start, goal, floor_z, tolerance=.1):
    path = np.asarray(path)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError('Invalid GPMP support path')
    if np.any(np.abs(path[:, 2]-floor_z) > .15):
        raise ValueError('Path leaves the selected floor')
    # Native A* deliberately omits the start; its first neighbour can be diagonal.
    if np.linalg.norm(path[0]-start) > .2+1e-7:
        raise ValueError('Native first waypoint is too far from the base support')
    if np.linalg.norm(path[-1]-goal) > tolerance+1e-7:
        raise ValueError('GPMP goal error exceeds 0.1 m')


class FollowState:
    def __init__(self, segments, arrival=.2, dwell=.5):
        if not segments:
            raise ValueError('At least one goal is required')
        self.segments = segments
        self.arrival, self.dwell = arrival, dwell
        self.state = 'waiting'
        self.index = -1
        self.started = self.inside = self.last_stamp = None
        self.events = []
        self.error = None

    def fail(self, reason, stamp):
        self.state, self.error = 'failed', reason
        self.events.append(dict(event='failed', stamp=stamp, segment=self.index, reason=reason))

    def update(self, position, stamp):
        position = np.asarray(position, float)
        if not np.isfinite(position).all() or not np.isfinite(stamp):
            self.fail('Invalid measured pose', None)
            return None
        if self.last_stamp is not None and stamp < self.last_stamp:
            self.fail('Odometry time moved backwards', stamp)
        self.last_stamp = stamp
        if self.state in ('failed', 'complete'):
            return None
        if self.state == 'waiting':
            return self._begin(0, stamp)
        if self.state == 'planning':
            return None
        row = self.segments[self.index]
        if stamp-self.started > max(120., 3*row['length_m']/.5):
            self.fail('Segment timed out; use the keyboard to pause the robot', stamp)
            return None
        if np.linalg.norm(position-row['goal_body_xyz']) > self.arrival:
            self.inside = None
        else:
            if self.inside is None:
                self.inside = stamp
            if stamp-self.inside+1e-9 >= self.dwell:
                self.events.append(dict(event='arrived', stamp=stamp, segment=self.index,
                                        position=position.tolist()))
                if self.index+1 == len(self.segments):
                    self.state = 'complete'
                else:
                    return self._begin(self.index+1, stamp)
        return None

    def _begin(self, index, stamp):
        self.index, self.started, self.inside, self.state = index, stamp, None, 'planning'
        self.events.append(dict(event='planning_requested', stamp=stamp, segment=index))
        return index

    def published(self, stamp, base):
        if self.state != 'planning':
            raise ValueError('No segment awaiting a path')
        self.started, self.state = stamp, 'following'
        self.events.append(dict(event='published', stamp=stamp, segment=self.index, base=list(map(float, base))))

    def report(self):
        return dict(state=self.state, active_segment=self.index,
                    completed_segments=sum(e['event']=='arrived' for e in self.events),
                    total_segments=len(self.segments), error=self.error, events=self.events)


class ShortGoalState:
    """Plan only after a measured arrival followed by a stationary hold."""
    def __init__(self,total,hold_s=3.,arrival=.2,dwell=.5):
        self.total,self.hold_s,self.arrival,self.dwell=total,hold_s,arrival,dwell
        self.state='waiting';self.index=0;self.events=[];self.goals=[];self.error=None
        self.last_stamp=self.started=self.inside=self.hold_start=None
        self.hold_origin=None;self.settle_until=None

    def fail(self,reason,stamp):
        self.state='failed';self.error=reason
        self.events.append(dict(event='failed',segment=self.index,stamp=stamp,reason=reason))

    def request(self,stamp):
        self.state='planning'
        self.events.append(dict(event='planning_requested',segment=self.index,stamp=stamp))
        return self.index

    def published(self,row,stamp,base):
        if self.state!='planning':raise ValueError('No short goal awaiting publication')
        self.goals.append(row);self.started=stamp;self.inside=None;self.state='following'
        self.events.append(dict(event='published',segment=self.index,stamp=stamp,base=list(map(float,base))))

    def update(self,position,stamp):
        position=np.asarray(position,float)
        if not np.isfinite(position).all() or not np.isfinite(stamp):
            self.fail('Invalid measured pose',None);return None
        if self.last_stamp is not None and stamp<self.last_stamp:
            self.fail('Odometry time moved backwards',stamp);return None
        self.last_stamp=stamp
        if self.state=='waiting':return self.request(stamp)
        if self.state=='following':
            row=self.goals[-1]
            if stamp-self.started>max(90.,3*row['length_m']/.35):
                self.fail('Short goal timed out',stamp);return None
            if np.linalg.norm(position-row['goal_body_xyz'])>self.arrival:self.inside=None
            else:
                if self.inside is None:self.inside=stamp
                if stamp-self.inside+1e-9>=self.dwell:
                    self.events.append(dict(event='arrived',segment=self.index,stamp=stamp,position=position.tolist()))
                    self.state='holding';self.settle_until=stamp+.6;self.hold_start=None;self.hold_origin=None
        elif self.state=='holding' and stamp>=self.settle_until:
            if self.hold_origin is None or np.linalg.norm(position-self.hold_origin)>.1:
                self.hold_origin=position.copy();self.hold_start=stamp
                self.events.append(dict(event='hold_started',segment=self.index,stamp=stamp,position=position.tolist()))
            if stamp-self.hold_start+1e-9>=self.hold_s:
                self.events.append(dict(event='hold_complete',segment=self.index,stamp=stamp,
                    stationary_s=stamp-self.hold_start,position=position.tolist()))
                if self.index+1==self.total:self.state='complete'
                else:
                    self.index+=1
                    return self.request(stamp)
        return None

    def report(self):
        return dict(state=self.state,active_segment=self.index,total_segments=self.total,
            completed_segments=sum(e['event']=='arrived' for e in self.events),
            completed_holds=sum(e['event']=='hold_complete' for e in self.events),
            hold_seconds=self.hold_s,error=self.error,events=self.events)
