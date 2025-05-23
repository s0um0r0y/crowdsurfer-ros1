// Generated by gencpp from file pedsim_msgs/LineObstacles.msg
// DO NOT EDIT!


#ifndef PEDSIM_MSGS_MESSAGE_LINEOBSTACLES_H
#define PEDSIM_MSGS_MESSAGE_LINEOBSTACLES_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <pedsim_msgs/LineObstacle.h>

namespace pedsim_msgs
{
template <class ContainerAllocator>
struct LineObstacles_
{
  typedef LineObstacles_<ContainerAllocator> Type;

  LineObstacles_()
    : header()
    , obstacles()  {
    }
  LineObstacles_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , obstacles(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef std::vector< ::pedsim_msgs::LineObstacle_<ContainerAllocator> , typename std::allocator_traits<ContainerAllocator>::template rebind_alloc< ::pedsim_msgs::LineObstacle_<ContainerAllocator> >> _obstacles_type;
  _obstacles_type obstacles;





  typedef boost::shared_ptr< ::pedsim_msgs::LineObstacles_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::pedsim_msgs::LineObstacles_<ContainerAllocator> const> ConstPtr;

}; // struct LineObstacles_

typedef ::pedsim_msgs::LineObstacles_<std::allocator<void> > LineObstacles;

typedef boost::shared_ptr< ::pedsim_msgs::LineObstacles > LineObstaclesPtr;
typedef boost::shared_ptr< ::pedsim_msgs::LineObstacles const> LineObstaclesConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::pedsim_msgs::LineObstacles_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::pedsim_msgs::LineObstacles_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::pedsim_msgs::LineObstacles_<ContainerAllocator1> & lhs, const ::pedsim_msgs::LineObstacles_<ContainerAllocator2> & rhs)
{
  return lhs.header == rhs.header &&
    lhs.obstacles == rhs.obstacles;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::pedsim_msgs::LineObstacles_<ContainerAllocator1> & lhs, const ::pedsim_msgs::LineObstacles_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace pedsim_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::pedsim_msgs::LineObstacles_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pedsim_msgs::LineObstacles_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::pedsim_msgs::LineObstacles_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::pedsim_msgs::LineObstacles_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pedsim_msgs::LineObstacles_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pedsim_msgs::LineObstacles_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::pedsim_msgs::LineObstacles_<ContainerAllocator> >
{
  static const char* value()
  {
    return "4de3122fdaa1292012d39892365813ee";
  }

  static const char* value(const ::pedsim_msgs::LineObstacles_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x4de3122fdaa12920ULL;
  static const uint64_t static_value2 = 0x12d39892365813eeULL;
};

template<class ContainerAllocator>
struct DataType< ::pedsim_msgs::LineObstacles_<ContainerAllocator> >
{
  static const char* value()
  {
    return "pedsim_msgs/LineObstacles";
  }

  static const char* value(const ::pedsim_msgs::LineObstacles_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::pedsim_msgs::LineObstacles_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# A collection of line obstacles.\n"
"# No need to header since these are detemined at sim initiation time.\n"
"Header header\n"
"pedsim_msgs/LineObstacle[] obstacles\n"
"\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"# Standard metadata for higher-level stamped data types.\n"
"# This is generally used to communicate timestamped data \n"
"# in a particular coordinate frame.\n"
"# \n"
"# sequence ID: consecutively increasing ID \n"
"uint32 seq\n"
"#Two-integer timestamp that is expressed as:\n"
"# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
"# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
"# time-handling sugar is provided by the client library\n"
"time stamp\n"
"#Frame this data is associated with\n"
"string frame_id\n"
"\n"
"================================================================================\n"
"MSG: pedsim_msgs/LineObstacle\n"
"# A line obstacle in the simulator.\n"
"\n"
"geometry_msgs/Point start\n"
"geometry_msgs/Point end\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Point\n"
"# This contains the position of a point in free space\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
;
  }

  static const char* value(const ::pedsim_msgs::LineObstacles_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::pedsim_msgs::LineObstacles_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.obstacles);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct LineObstacles_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::pedsim_msgs::LineObstacles_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::pedsim_msgs::LineObstacles_<ContainerAllocator>& v)
  {
    if (false || !indent.empty())
      s << std::endl;
    s << indent << "header: ";
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    if (true || !indent.empty())
      s << std::endl;
    s << indent << "obstacles: ";
    if (v.obstacles.empty() || false)
      s << "[";
    for (size_t i = 0; i < v.obstacles.size(); ++i)
    {
      if (false && i > 0)
        s << ", ";
      else if (!false)
        s << std::endl << indent << "  -";
      Printer< ::pedsim_msgs::LineObstacle_<ContainerAllocator> >::stream(s, false ? std::string() : indent + "    ", v.obstacles[i]);
    }
    if (v.obstacles.empty() || false)
      s << "]";
  }
};

} // namespace message_operations
} // namespace ros

#endif // PEDSIM_MSGS_MESSAGE_LINEOBSTACLES_H
