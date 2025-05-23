#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/src/turtlebot2_noetic_packages/kobuki/kobuki_testsuite"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/install/lib/python3/dist-packages:/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/build" \
    "/usr/bin/python3" \
    "/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/src/turtlebot2_noetic_packages/kobuki/kobuki_testsuite/setup.py" \
     \
    build --build-base "/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/build/turtlebot2_noetic_packages/kobuki/kobuki_testsuite" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/install" --install-scripts="/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/install/bin"
