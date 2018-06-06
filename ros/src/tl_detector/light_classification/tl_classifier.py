from styx_msgs.msg import TrafficLight
import numpy as np
import cv2
import os
import rospy
import yaml

from keras.models import load_model
from keras import backend as K

IMAGE_HEIGHT = 600
IMAGE_WIDTH = 800
IMAGE_CHANNEL = 3

class TLClassifier(object):
    def __init__(self):
        self.model = None
        self.graph = None
        self.model_path = None

        #Load configuration
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        #select proper location according to environment
        if not (self.config['tl']['is_carla']):
            self.model_path = './light_classification/models/sim_tl_classifier.h5'
        else:
            self.model_path = './light_classification/models/real_tl_classifier.h5'
        # rospy.loginfo(self.config['tl']['is_carla'])
        rospy.loginfo(self.model_path)

        if self.model_path != None and os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            self.model._make_predict_function()
            self.graph = K.tf.get_default_graph()
            rospy.loginfo('Model loaded successfully!!')
        else:
            # print('Searched for:', self.sim_model_path,'No saved model found!!')
            rospy.logerr('Searched for: sim_tl_classifier.h5. No saved model found!!')
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #resize the image as accepted by the model
        image = cv2.resize(image,(IMAGE_WIDTH, IMAGE_HEIGHT))

        try:
            with self.graph.as_default():
                if self.model != None:
                    image = np.reshape( image, (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
                    scores = self.model.predict(image)

                    if type(scores)!= None and len(scores) > 0:
                        image_class = np.argmax(scores)

                        if image_class == 0:
                            return TrafficLight.RED
                        elif image_class == 1:
                            return TrafficLight.GREEN
                        else:
                            return TrafficLight.UNKNOWN
                    else:
                        rospy.logwarn('Model prediction empty')
                        return TrafficLight.UNKNOWN

                else:
                    return TrafficLight.UNKNOWN

        except Exception as e:
            rospy.logerr('TL Classifier failed!! Exception raised!!')
            rospy.logerr(e)
            return TrafficLight.UNKNOWN
