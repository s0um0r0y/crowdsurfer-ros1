// Generated by gencpp from file pedsim_msgs/Waypoint.msg
// DO NOT EDIT!


#ifndef PEDSIM_MSGS_MESSAGE_WAYPOINT_H
#define PEDSIM_MSGS_MESSAGE_WAYPOINT_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <geometry_msgs/Point.h>

namespace pedsim_msgs
{
template <class ContainerAllocator>
struct Waypoint_
{
  typedef Waypoint_<ContainerAllocator> Type;

  Waypoint_()
    : name()
    , behavior(0)
    , position()
    , radius(0.0)  {
    }
  Waypoint_(const ContainerAllocator& _alloc)
    : name(_alloc)
    , behavior(0)
    , position(_alloc)
    , radius(0.0)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> _name_type;
  _name_type name;

   typedef int8_t _behavior_type;
  _behavior_type behavior;

   typedef  ::geometry_msgs::Point_<ContainerAllocator>  _position_type;
  _position_type position;

   typedef float _radius_type;
  _radius_type radius;



// reducing the odds to have name collisions with Windows.h 
#if defined(_WIN32) && defined(BHV_SIMPLE)
  #undef BHV_SIMPLE
#endif
#if defined(_WIN32) && defined(BHV_SOURCE)
  #undef BHV_SOURCE
#endif
#if defined(_WIN32) && defined(BHV_SINK)
  #undef BHV_SINK
#endif

  enum {
    BHV_SIMPLE = 0,
    BHV_SOURCE = 1,
    BHV_SINK = 2,
  };


  typedef boost::shared_ptr< ::pedsim_msgs::Waypoint_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::pedsim_msgs::Waypoint_<ContainerAllocator> const> ConstPtr;

}; // struct Waypoint_

typedef ::pedsim_msgs::Waypoint_<std::allocator<void> > Waypoint;

typedef boost::shared_ptr< ::pedsim_msgs::Waypoint > WaypointPtr;
typedef boost::shared_ptr< ::pedsim_msgs::Waypoint const> WaypointConstPtr;

// constants requiring out of line definition

   

   

   



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::pedsim_msgs::Waypoint_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::pedsim_msgs::Waypoint_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::pedsim_msgs::Waypoint_<ContainerAllocator1> & lhs, const ::pedsim_msgs::Waypoint_<ContainerAllocator2> & rhs)
{
  return lhs.name == rhs.name &&
    lhs.behavior == rhs.behavior &&
    lhs.position == rhs.position &&
    lhs.radius == rhs.radius;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::pedsim_msgs::Waypoint_<ContainerAllocator1> & lhs, const ::pedsim_msgs::Waypoint_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace pedsim_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::pedsim_msgs::Waypoint_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pedsim_msgs::Waypoint_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::pedsim_msgs::Waypoint_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::pedsim_msgs::Waypoint_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pedsim_msgs::Waypoint_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pedsim_msgs::Waypoint_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::pedsim_msgs::Waypoint_<ContainerAllocator> >
{
  static const char* value()
  {
    return "7fe6c11b241f6ddddc1c756dacf4a21f";
  }

  static const char* value(const ::pedsim_msgs::Waypoint_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x7fe6c11b241f6dddULL;
  static const uint64_t static_value2 = 0xdc1c756dacf4a21fULL;
};

template<class ContainerAllocator>
struct DataType< ::pedsim_msgs::Waypoint_<ContainerAllocator> >
{
  static const char* value()
  {
    return "pedsim_msgs/Waypoint";
  }

  static const char* value(const ::pedsim_msgs::Waypoint_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::pedsim_msgs::Waypoint_<ContainerAllocator> >
{
  static const char* value()
  {
    return "int8 BHV_SIMPLE = 0\n"
"int8 BHV_SOURCE = 1\n"
"int8 BHV_SINK = 2\n"
"\n"
"string name\n"
"int8 behavior\n"
"geometry_msgs/Point position\n"
"float32 radius\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Point\n"
"# This contains the position of a point in free space\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
;
  }

  static const char* value(const ::pedsim_msgs::Waypoint_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::pedsim_msgs::Waypoint_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.name);
      stream.next(m.behavior);
      stream.next(m.position);
      stream.next(m.radius);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Waypoint_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::pedsim_msgs::Waypoint_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::pedsim_msgs::Waypoint_<ContainerAllocator>& v)
  {
    if (false || !indent.empty())
      s << std::endl;
    s << indent << "name: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>::stream(s, indent + "  ", v.name);
    if (true || !indent.empty())
      s << std::endl;
    s << indent << "behavior: ";
    Printer<int8_t>::stream(s, indent + "  ", v.behavior);
    if (true || !indent.empty())
      s << std::endl;
    s << indent << "position: ";
    Printer< ::geometry_msgs::Point_<ContainerAllocator> >::stream(s, indent + "  ", v.position);
    if (true || !indent.empty())
      s << std::endl;
    s << indent << "radius: ";
    Printer<float>::stream(s, indent + "  ", v.radius);
  }
};

} // namespace message_operations
} // namespace ros

#endif // PEDSIM_MSGS_MESSAGE_WAYPOINT_H
