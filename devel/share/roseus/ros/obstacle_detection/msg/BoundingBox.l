;; Auto-generated. Do not edit!


(when (boundp 'obstacle_detection::BoundingBox)
  (if (not (find-package "OBSTACLE_DETECTION"))
    (make-package "OBSTACLE_DETECTION"))
  (shadow 'BoundingBox (find-package "OBSTACLE_DETECTION")))
(unless (find-package "OBSTACLE_DETECTION::BOUNDINGBOX")
  (make-package "OBSTACLE_DETECTION::BOUNDINGBOX"))

(in-package "ROS")
;;//! \htmlinclude BoundingBox.msg.html


(defclass obstacle_detection::BoundingBox
  :super ros::object
  :slots (_center_x _center_y _w _h _confidence _classid ))

(defmethod obstacle_detection::BoundingBox
  (:init
   (&key
    ((:center_x __center_x) 0.0)
    ((:center_y __center_y) 0.0)
    ((:w __w) 0.0)
    ((:h __h) 0.0)
    ((:confidence __confidence) 0.0)
    ((:classid __classid) 0.0)
    )
   (send-super :init)
   (setq _center_x (float __center_x))
   (setq _center_y (float __center_y))
   (setq _w (float __w))
   (setq _h (float __h))
   (setq _confidence (float __confidence))
   (setq _classid (float __classid))
   self)
  (:center_x
   (&optional __center_x)
   (if __center_x (setq _center_x __center_x)) _center_x)
  (:center_y
   (&optional __center_y)
   (if __center_y (setq _center_y __center_y)) _center_y)
  (:w
   (&optional __w)
   (if __w (setq _w __w)) _w)
  (:h
   (&optional __h)
   (if __h (setq _h __h)) _h)
  (:confidence
   (&optional __confidence)
   (if __confidence (setq _confidence __confidence)) _confidence)
  (:classid
   (&optional __classid)
   (if __classid (setq _classid __classid)) _classid)
  (:serialization-length
   ()
   (+
    ;; float32 _center_x
    4
    ;; float32 _center_y
    4
    ;; float32 _w
    4
    ;; float32 _h
    4
    ;; float32 _confidence
    4
    ;; float32 _classid
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _center_x
       (sys::poke _center_x (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _center_y
       (sys::poke _center_y (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _w
       (sys::poke _w (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _h
       (sys::poke _h (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _confidence
       (sys::poke _confidence (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _classid
       (sys::poke _classid (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _center_x
     (setq _center_x (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _center_y
     (setq _center_y (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _w
     (setq _w (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _h
     (setq _h (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _confidence
     (setq _confidence (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _classid
     (setq _classid (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get obstacle_detection::BoundingBox :md5sum-) "be156bfb41e8226bcfb5c78e0f4bbf55")
(setf (get obstacle_detection::BoundingBox :datatype-) "obstacle_detection/BoundingBox")
(setf (get obstacle_detection::BoundingBox :definition-)
      "float32 center_x
float32 center_y
float32 w
float32 h
float32 confidence
float32 classid
")



(provide :obstacle_detection/BoundingBox "be156bfb41e8226bcfb5c78e0f4bbf55")


