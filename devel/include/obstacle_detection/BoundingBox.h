// Generated by gencpp from file obstacle_detection/BoundingBox.msg
// DO NOT EDIT!


#ifndef OBSTACLE_DETECTION_MESSAGE_BOUNDINGBOX_H
#define OBSTACLE_DETECTION_MESSAGE_BOUNDINGBOX_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace obstacle_detection
{
template <class ContainerAllocator>
struct BoundingBox_
{
  typedef BoundingBox_<ContainerAllocator> Type;

  BoundingBox_()
    : center_x(0.0)
    , center_y(0.0)
    , w(0.0)
    , h(0.0)
    , confidence(0.0)
    , classid(0.0)  {
    }
  BoundingBox_(const ContainerAllocator& _alloc)
    : center_x(0.0)
    , center_y(0.0)
    , w(0.0)
    , h(0.0)
    , confidence(0.0)
    , classid(0.0)  {
  (void)_alloc;
    }



   typedef float _center_x_type;
  _center_x_type center_x;

   typedef float _center_y_type;
  _center_y_type center_y;

   typedef float _w_type;
  _w_type w;

   typedef float _h_type;
  _h_type h;

   typedef float _confidence_type;
  _confidence_type confidence;

   typedef float _classid_type;
  _classid_type classid;





  typedef boost::shared_ptr< ::obstacle_detection::BoundingBox_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::obstacle_detection::BoundingBox_<ContainerAllocator> const> ConstPtr;

}; // struct BoundingBox_

typedef ::obstacle_detection::BoundingBox_<std::allocator<void> > BoundingBox;

typedef boost::shared_ptr< ::obstacle_detection::BoundingBox > BoundingBoxPtr;
typedef boost::shared_ptr< ::obstacle_detection::BoundingBox const> BoundingBoxConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::obstacle_detection::BoundingBox_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::obstacle_detection::BoundingBox_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::obstacle_detection::BoundingBox_<ContainerAllocator1> & lhs, const ::obstacle_detection::BoundingBox_<ContainerAllocator2> & rhs)
{
  return lhs.center_x == rhs.center_x &&
    lhs.center_y == rhs.center_y &&
    lhs.w == rhs.w &&
    lhs.h == rhs.h &&
    lhs.confidence == rhs.confidence &&
    lhs.classid == rhs.classid;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::obstacle_detection::BoundingBox_<ContainerAllocator1> & lhs, const ::obstacle_detection::BoundingBox_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace obstacle_detection

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::obstacle_detection::BoundingBox_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::obstacle_detection::BoundingBox_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::obstacle_detection::BoundingBox_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::obstacle_detection::BoundingBox_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::obstacle_detection::BoundingBox_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::obstacle_detection::BoundingBox_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::obstacle_detection::BoundingBox_<ContainerAllocator> >
{
  static const char* value()
  {
    return "be156bfb41e8226bcfb5c78e0f4bbf55";
  }

  static const char* value(const ::obstacle_detection::BoundingBox_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xbe156bfb41e8226bULL;
  static const uint64_t static_value2 = 0xcfb5c78e0f4bbf55ULL;
};

template<class ContainerAllocator>
struct DataType< ::obstacle_detection::BoundingBox_<ContainerAllocator> >
{
  static const char* value()
  {
    return "obstacle_detection/BoundingBox";
  }

  static const char* value(const ::obstacle_detection::BoundingBox_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::obstacle_detection::BoundingBox_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float32 center_x\n"
"float32 center_y\n"
"float32 w\n"
"float32 h\n"
"float32 confidence\n"
"float32 classid\n"
;
  }

  static const char* value(const ::obstacle_detection::BoundingBox_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::obstacle_detection::BoundingBox_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.center_x);
      stream.next(m.center_y);
      stream.next(m.w);
      stream.next(m.h);
      stream.next(m.confidence);
      stream.next(m.classid);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct BoundingBox_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::obstacle_detection::BoundingBox_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::obstacle_detection::BoundingBox_<ContainerAllocator>& v)
  {
    s << indent << "center_x: ";
    Printer<float>::stream(s, indent + "  ", v.center_x);
    s << indent << "center_y: ";
    Printer<float>::stream(s, indent + "  ", v.center_y);
    s << indent << "w: ";
    Printer<float>::stream(s, indent + "  ", v.w);
    s << indent << "h: ";
    Printer<float>::stream(s, indent + "  ", v.h);
    s << indent << "confidence: ";
    Printer<float>::stream(s, indent + "  ", v.confidence);
    s << indent << "classid: ";
    Printer<float>::stream(s, indent + "  ", v.classid);
  }
};

} // namespace message_operations
} // namespace ros

#endif // OBSTACLE_DETECTION_MESSAGE_BOUNDINGBOX_H
