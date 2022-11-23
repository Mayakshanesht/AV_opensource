; Auto-generated. Do not edit!


(cl:in-package obstacle_detection-msg)


;//! \htmlinclude BoundingBoxes.msg.html

(cl:defclass <BoundingBoxes> (roslisp-msg-protocol:ros-message)
  ((BoundingBoxes
    :reader BoundingBoxes
    :initarg :BoundingBoxes
    :type (cl:vector obstacle_detection-msg:BoundingBox)
   :initform (cl:make-array 0 :element-type 'obstacle_detection-msg:BoundingBox :initial-element (cl:make-instance 'obstacle_detection-msg:BoundingBox))))
)

(cl:defclass BoundingBoxes (<BoundingBoxes>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <BoundingBoxes>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'BoundingBoxes)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name obstacle_detection-msg:<BoundingBoxes> is deprecated: use obstacle_detection-msg:BoundingBoxes instead.")))

(cl:ensure-generic-function 'BoundingBoxes-val :lambda-list '(m))
(cl:defmethod BoundingBoxes-val ((m <BoundingBoxes>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader obstacle_detection-msg:BoundingBoxes-val is deprecated.  Use obstacle_detection-msg:BoundingBoxes instead.")
  (BoundingBoxes m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <BoundingBoxes>) ostream)
  "Serializes a message object of type '<BoundingBoxes>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'BoundingBoxes))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'BoundingBoxes))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <BoundingBoxes>) istream)
  "Deserializes a message object of type '<BoundingBoxes>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'BoundingBoxes) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'BoundingBoxes)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'obstacle_detection-msg:BoundingBox))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<BoundingBoxes>)))
  "Returns string type for a message object of type '<BoundingBoxes>"
  "obstacle_detection/BoundingBoxes")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'BoundingBoxes)))
  "Returns string type for a message object of type 'BoundingBoxes"
  "obstacle_detection/BoundingBoxes")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<BoundingBoxes>)))
  "Returns md5sum for a message object of type '<BoundingBoxes>"
  "c87179ebc264f92e8bbb6eaab4093103")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'BoundingBoxes)))
  "Returns md5sum for a message object of type 'BoundingBoxes"
  "c87179ebc264f92e8bbb6eaab4093103")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<BoundingBoxes>)))
  "Returns full string definition for message of type '<BoundingBoxes>"
  (cl:format cl:nil "BoundingBox[] BoundingBoxes~%================================================================================~%MSG: obstacle_detection/BoundingBox~%float32 center_x~%float32 center_y~%float32 w~%float32 h~%float32 confidence~%float32 classid~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'BoundingBoxes)))
  "Returns full string definition for message of type 'BoundingBoxes"
  (cl:format cl:nil "BoundingBox[] BoundingBoxes~%================================================================================~%MSG: obstacle_detection/BoundingBox~%float32 center_x~%float32 center_y~%float32 w~%float32 h~%float32 confidence~%float32 classid~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <BoundingBoxes>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'BoundingBoxes) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <BoundingBoxes>))
  "Converts a ROS message object to a list"
  (cl:list 'BoundingBoxes
    (cl:cons ':BoundingBoxes (BoundingBoxes msg))
))
