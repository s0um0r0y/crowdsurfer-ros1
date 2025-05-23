## *********************************************************
##
## File autogenerated for the pedsim_simulator package
## by the dynamic_reconfigure package.
## Please do not edit.
##
## ********************************************************/

from dynamic_reconfigure.encoding import extract_params

inf = float('inf')

config_description = {'name': 'Default', 'type': '', 'state': True, 'cstate': 'true', 'id': 0, 'parent': 0, 'parameters': [{'name': 'update_rate', 'type': 'double', 'default': 24.0, 'level': 0, 'description': 'Simulation frequency (Hz)', 'min': 1.0, 'max': 50.0, 'srcline': 292, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'simulation_factor', 'type': 'double', 'default': 1.0, 'level': 0, 'description': 'Simulation factor', 'min': 0.1, 'max': 10.0, 'srcline': 292, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'force_obstacle', 'type': 'double', 'default': 10.0, 'level': 0, 'description': 'Obstacle force weight', 'min': 0.0, 'max': 50.0, 'srcline': 292, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'sigma_obstacle', 'type': 'double', 'default': 0.2, 'level': 0, 'description': 'Sigma factor (obstacles)', 'min': 0.1, 'max': 1.0, 'srcline': 292, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'force_social', 'type': 'double', 'default': 5.1, 'level': 0, 'description': 'Social force weight', 'min': 0.0, 'max': 10.0, 'srcline': 292, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'force_group_gaze', 'type': 'double', 'default': 3.0, 'level': 0, 'description': 'Group gaze force weight', 'min': 0.0, 'max': 10.0, 'srcline': 292, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'force_group_coherence', 'type': 'double', 'default': 2.0, 'level': 0, 'description': 'Group coherence force weight', 'min': 0.0, 'max': 10.0, 'srcline': 292, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'force_group_repulsion', 'type': 'double', 'default': 1.0, 'level': 0, 'description': 'Group repulsion force weight', 'min': 0.0, 'max': 10.0, 'srcline': 292, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'force_random', 'type': 'double', 'default': 0.1, 'level': 0, 'description': 'Random force weight', 'min': 0.0, 'max': 1.0, 'srcline': 292, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'force_wall', 'type': 'double', 'default': 2.0, 'level': 0, 'description': 'Wall force weight', 'min': 0.0, 'max': 10.0, 'srcline': 292, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'double', 'cconsttype': 'const double'}, {'name': 'paused', 'type': 'bool', 'default': False, 'level': 0, 'description': 'Pause/unpause simulation', 'min': False, 'max': True, 'srcline': 292, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'edit_method': '', 'ctype': 'bool', 'cconsttype': 'const bool'}], 'groups': [], 'srcline': 247, 'srcfile': '/opt/ros/noetic/lib/python3/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'class': 'DEFAULT', 'parentclass': '', 'parentname': 'Default', 'field': 'default', 'upper': 'DEFAULT', 'lower': 'groups'}

min = {}
max = {}
defaults = {}
level = {}
type = {}
all_level = 0

#def extract_params(config):
#    params = []
#    params.extend(config['parameters'])
#    for group in config['groups']:
#        params.extend(extract_params(group))
#    return params

for param in extract_params(config_description):
    min[param['name']] = param['min']
    max[param['name']] = param['max']
    defaults[param['name']] = param['default']
    level[param['name']] = param['level']
    type[param['name']] = param['type']
    all_level = all_level | param['level']

