#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import rospy
import numpy as np
import pandas as pd
from cv_bridge import CvBridge

""" Class that loads timestamped images and events from a rosbag """


class Bag2Images:
    def __init__(self, path_to_bag, topic, secs0=None, nsecs0=None, max_num_messages_to_read=-1, t_max=-1):
        self.secs0 = secs0
        self.nsecs0 = nsecs0

        self.times = []
        self.images = []

        self.bridge = CvBridge()

        with rosbag.Bag(path_to_bag) as bag:

            topics = bag.get_type_and_topic_info().topics
            if topic not in topics:
                print("Warning: The topic with name %s was not found in bag %s" % (topic, path_to_bag))
                self.df = pd.DataFrame({"image": [], "time": []})
                return

            total_num_msgs = bag.get_message_count(topic)
            num_msgs_to_read = max_num_messages_to_read if max_num_messages_to_read > 0 else total_num_msgs

            msg_idx = 1
            t0 = None
            for topic, msg, t in bag.read_messages(topics=[topic]):
                if t0 is None and t is not None:
                    t = t0

                self.addImage(msg)
                msg_idx += 1

                if msg_idx >= num_msgs_to_read:
                    break

                if t is not None and t0 is not None:
                    if (t - t0).to_sec() > t_max and t_max != -1:
                        break

        self.df = pd.DataFrame({"image": self.images, "time": self.times})

    def addImage(self, msg):
        if self.secs0 == None:
            self.secs0 = msg.header.stamp.secs
            self.nsecs0 = msg.header.stamp.nsecs

        time = msg.header.stamp.secs - self.secs0 + 1e-9 * (msg.header.stamp.nsecs - self.nsecs0)
        self.times.append(time)

        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.images.append(img)

    def to_torch_video_tensor(self):
        return np.stack([np.permute(i, [2,0,1]) for i in self.images])


class Bag2Events:
    """ Class that loads events from a ROS bag to a DataFrame """
    def __init__(self, path_to_bag, topic="", secs0=0, nsecs0=0, max_num_messages_to_read=-1, t_max=-1):
        self.secs0 = secs0
        self.nsecs0 = nsecs0
        self.times = []
        self.xs = []
        self.ys = []
        self.polarities = []
        self.df = None

        with rosbag.Bag(path_to_bag) as bag:

            topics = bag.get_type_and_topic_info().topics
            for topic_name, topic_info in topics.iteritems():
                if topic_name == topic or (topic =="" and topic_info.msg_type == "dvs_msgs/EventArray"):
                    event_topic = topic_name
                    break
            else:
                print("Warning: The topic with name %s was not found in bag %s" % (topic, path_to_bag))
                self.df = pd.DataFrame({"y": [], "polarity": [], "x": [], "time": []})
                return

            total_num_msgs = bag.get_message_count(event_topic)
            num_msgs_to_read = max_num_messages_to_read if max_num_messages_to_read > 0 else total_num_msgs

            msg_idx = 1
            t0 = None
            for i,(topic, msg, t) in enumerate(bag.read_messages(topics=[event_topic], end_time=rospy.Time(1.8))):
                print(i, topic, msg.header.stamp, msg.header.seq, msg.events[-1])
                if t0 is None:
                    t0 = t

                self.addEventArray(msg)
                msg_idx += 1

                if msg_idx >= num_msgs_to_read:
                    break

                if (t - t0).to_sec() > t_max and t_max != -1:
                    break
        print(bag.get_message_count(topic))
        self.times = np.array(self.times)
        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)
        self.polarities = np.array(self.polarities)

        self.df = pd.DataFrame({'time': self.times.astype(np.float32),
                                'x': self.xs.astype(np.uint8),
                                'y': self.ys.astype(np.uint8),
                                'polarity': (self.polarities == 1).astype(np.bool)})

    def addEventArray(self, msg):
        if not msg.events:
            return

        if self.secs0 == None:
            self.secs0 = msg.events[0].ts.secs
            self.nsecs0 = msg.events[0].ts.nsecs
        #print(msg.header.stamp)

        times_tmp = []
        xs_tmp = []
        ys_tmp = []
        pols_tmp = []

        for e in msg.events:
            times_tmp.append(e.ts.secs - self.secs0 + 1e-9 * (e.ts.nsecs - self.nsecs0))
            xs_tmp.append(e.x)
            ys_tmp.append(e.y)
            pols_tmp.append(1 if e.polarity else -1)

        self.times += times_tmp
        self.xs += xs_tmp
        self.ys += ys_tmp
        self.polarities += pols_tmp

    def to_xytp(self):
        return self.df.as_matrix(["x", "y", "time", "polarity"])

    def to_txyp(self):
        return self.df.as_matrix(["time", "y", "x", "polarity"])