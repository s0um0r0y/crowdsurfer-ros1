// Generated by gencpp from file spencer_tracking_msgs/GetPersonTrajectoriesResponse.msg
// DO NOT EDIT!


#ifndef SPENCER_TRACKING_MSGS_MESSAGE_GETPERSONTRAJECTORIESRESPONSE_H
#define SPENCER_TRACKING_MSGS_MESSAGE_GETPERSONTRAJECTORIESRESPONSE_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <spencer_tracking_msgs/PersonTrajectory.h>

namespace spencer_tracking_msgs
{
template <class ContainerAllocator>
struct GetPersonTrajectoriesResponse_
{
  typedef GetPersonTrajectoriesResponse_<ContainerAllocator> Type;

  GetPersonTrajectoriesResponse_()
    : trajectories()  {
    }
  GetPersonTrajectoriesResponse_(const ContainerAllocator& _alloc)
    : trajectories(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector< ::spencer_tracking_msgs::PersonTrajectory_<ContainerAllocator> , typename std::allocator_traits<ContainerAllocator>::template rebind_alloc< ::spencer_tracking_msgs::PersonTrajectory_<ContainerAllocator> >> _trajectories_type;
  _trajectories_type trajectories;





  typedef boost::shared_ptr< ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator> const> ConstPtr;

}; // struct GetPersonTrajectoriesResponse_

typedef ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<std::allocator<void> > GetPersonTrajectoriesResponse;

typedef boost::shared_ptr< ::spencer_tracking_msgs::GetPersonTrajectoriesResponse > GetPersonTrajectoriesResponsePtr;
typedef boost::shared_ptr< ::spencer_tracking_msgs::GetPersonTrajectoriesResponse const> GetPersonTrajectoriesResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator1> & lhs, const ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator2> & rhs)
{
  return lhs.trajectories == rhs.trajectories;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator1> & lhs, const ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace spencer_tracking_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "cac69139f499658fd82ffbcabd799a3d";
  }

  static const char* value(const ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xcac69139f499658fULL;
  static const uint64_t static_value2 = 0xd82ffbcabd799a3dULL;
};

template<class ContainerAllocator>
struct DataType< ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "spencer_tracking_msgs/GetPersonTrajectoriesResponse";
  }

  static const char* value(const ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "PersonTrajectory[] trajectories  # The trajectories of the tracks that have been asked for in requested_ids, in the same order.\n"
"\n"
"\n"
"================================================================================\n"
"MSG: spencer_tracking_msgs/PersonTrajectory\n"
"# Message defining the trajectory of a tracked person.\n"
"#\n"
"# The distinction between track and trajectory is that, depending on the\n"
"# implementation of the tracker, a single track (i.e. tracked person) might\n"
"# change the trajectory if at some point a new trajectory \"fits\" that track (person)\n"
"# better.\n"
"#\n"
"\n"
"uint64                   track_id   # Unique identifier of the tracked person.\n"
"PersonTrajectoryEntry[]  trajectory # All states of the last T frames of the most likely trajectory of that tracked person.\n"
"\n"
"================================================================================\n"
"MSG: spencer_tracking_msgs/PersonTrajectoryEntry\n"
"# Message defining an entry of a person trajectory.\n"
"#\n"
"\n"
"time     stamp           # age of the track\n"
"bool     is_occluded     # if the track is currently not matched by a detection\n"
"uint64   detection_id    # id of the corresponding detection in the current cycle (undefined if occluded)\n"
"\n"
"# The following fields are extracted from the Kalman state x and its covariance C\n"
"\n"
"geometry_msgs/PoseWithCovariance    pose   # pose of the track (z value and orientation might not be set, check if corresponding variance on diagonal is > 99999)\n"
"geometry_msgs/TwistWithCovariance   twist  # velocity of the track (z value and rotational velocities might not be set, check if corresponding variance on diagonal is > 99999)\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/PoseWithCovariance\n"
"# This represents a pose in free space with uncertainty.\n"
"\n"
"Pose pose\n"
"\n"
"# Row-major representation of the 6x6 covariance matrix\n"
"# The orientation parameters use a fixed-axis representation.\n"
"# In order, the parameters are:\n"
"# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)\n"
"float64[36] covariance\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Pose\n"
"# A representation of pose in free space, composed of position and orientation. \n"
"Point position\n"
"Quaternion orientation\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Point\n"
"# This contains the position of a point in free space\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Quaternion\n"
"# This represents an orientation in free space in quaternion form.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"float64 w\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/TwistWithCovariance\n"
"# This expresses velocity in free space with uncertainty.\n"
"\n"
"Twist twist\n"
"\n"
"# Row-major representation of the 6x6 covariance matrix\n"
"# The orientation parameters use a fixed-axis representation.\n"
"# In order, the parameters are:\n"
"# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)\n"
"float64[36] covariance\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Twist\n"
"# This expresses velocity in free space broken into its linear and angular parts.\n"
"Vector3 linear\n"
"Vector3 angular\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Vector3\n"
"# This represents a vector in free space. \n"
"# It is only meant to represent a direction. Therefore, it does not\n"
"# make sense to apply a translation to it (e.g., when applying a \n"
"# generic rigid transformation to a Vector3, tf2 will only apply the\n"
"# rotation). If you want your data to be translatable too, use the\n"
"# geometry_msgs/Point message instead.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
;
  }

  static const char* value(const ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.trajectories);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct GetPersonTrajectoriesResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::spencer_tracking_msgs::GetPersonTrajectoriesResponse_<ContainerAllocator>& v)
  {
    if (false || !indent.empty())
      s << std::endl;
    s << indent << "trajectories: ";
    if (v.trajectories.empty() || false)
      s << "[";
    for (size_t i = 0; i < v.trajectories.size(); ++i)
    {
      if (false && i > 0)
        s << ", ";
      else if (!false)
        s << std::endl << indent << "  -";
      Printer< ::spencer_tracking_msgs::PersonTrajectory_<ContainerAllocator> >::stream(s, false ? std::string() : indent + "    ", v.trajectories[i]);
    }
    if (v.trajectories.empty() || false)
      s << "]";
  }
};

} // namespace message_operations
} // namespace ros

#endif // SPENCER_TRACKING_MSGS_MESSAGE_GETPERSONTRAJECTORIESRESPONSE_H
