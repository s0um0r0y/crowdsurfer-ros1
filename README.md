# Implemention of generative AI with sampling optimizer for path planning for a mobile robot

- reimplemented with RVIZ visualiztion in closed loop with a rosbag
- implemented generative AI for path planning using VQ-VAE and PixelCNN

![teaser](./sampling_from_vqvae.png)
![teaser](./PixelCNN.png)

# RVIZ window

- blue colour line is VQ-VAE + PixelCNN generated trajectory
- green colour line is PRIEST optimized trajectory
- green colour marker is the dynamic obstacles
  
![comparison_vqvae_pixelcnn_PRIEST_optimzer](https://github.com/user-attachments/assets/8896391a-1b49-4353-86b8-a23a5c3fdb22)

# How to run ?
```
  conda env create -f environment.yml
  ./run_open_loop_rosbag.sh
```
