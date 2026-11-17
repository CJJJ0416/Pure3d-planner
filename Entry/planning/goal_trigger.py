"""Initial/arrival inference events, with no temporal point-cloud partitioning."""
import numpy as np


class GoalCostTrigger:
    def __init__(self,tolerance=.3,dwell=3.,initial_dwell=3.):
        if tolerance<=0 or dwell<0 or initial_dwell<0:raise ValueError('Invalid goal timing')
        self.tolerance=tolerance;self.dwell=dwell;self.initial_dwell=initial_dwell
        self.initial_ready=False;self.initial_position=None;self.initial_since=None
        self.goal=None;self.goal_sequence=0;self.completed_goals=0;self.inside_since=None
        self.last_stamp=None;self.requests=[];self.goals=[]

    def set_goal(self,position,stamp):
        position=np.asarray(position,dtype=float)
        if position.shape!=(3,) or not np.isfinite(position).all():raise ValueError('Invalid goal')
        if self.goal is not None and np.linalg.norm(position-self.goal['position'])<.01:return False
        self.goal_sequence+=1;self.inside_since=None
        self.goal=dict(id=self.goal_sequence,position=position.tolist(),stamp=float(stamp),reached=False)
        self.goals.append(self.goal)
        return True

    def observe(self,position,stamp,revision,points):
        if points<=0:return None
        position=np.asarray(position,dtype=float)
        if position.shape!=(3,) or not np.isfinite(position).all() or not np.isfinite(stamp):raise ValueError('Invalid observation')
        if self.last_stamp is not None and stamp<self.last_stamp:raise ValueError('Odometry time moved backwards; restart Entry')
        self.last_stamp=stamp;reason=None
        if not self.initial_ready:
            if self.initial_position is None or np.linalg.norm(position-self.initial_position)>.1:
                self.initial_position=position.copy();self.initial_since=stamp
            if stamp-self.initial_since+1e-9<self.initial_dwell:return None
            self.initial_ready=True;reason='initial'
        elif self.goal is not None and not self.goal['reached']:
            if np.linalg.norm(position-self.goal['position'])>self.tolerance:self.inside_since=None;return None
            if self.inside_since is None:self.inside_since=stamp
            if stamp-self.inside_since+1e-9<self.dwell:return None
            self.goal['reached']=True;self.goal['arrival_stamp']=float(stamp)
            self.completed_goals+=1;reason='goal'
        if reason is None:return None
        request=dict(prediction_id=len(self.requests)+1,trigger=reason,goal_id=self.goal['id'] if reason=='goal' else None,
            stamp=float(stamp),map_revision=revision,points=points,input_scope='whole_pcd')
        self.requests.append(request)
        return request

    def report(self):
        latest=self.requests[-1] if self.requests else None
        return dict(initial_ready=self.initial_ready,completed_goals=self.completed_goals,goal_sequence=self.goal_sequence,
            active_goal=self.goal,goals=self.goals,requests=self.requests,
            latest_request_id=latest['prediction_id'] if latest else None,
            last_request_revision=latest['map_revision'] if latest else None)
