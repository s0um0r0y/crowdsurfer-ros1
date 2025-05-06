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

# Pedsim Gazebo simulator

- Validated results in a crowded environment simulation to avoid humans

![teaser](./gazebo_simulation.png)

# How to run ?
```
  # this command is for running open loop using a rosbag
  conda env create -f environment.yml
  ./run_open_loop_rosbag.sh

  # this command is for running closed loop in pedsim gazebo simulation
  ./run_closed_loop_simulation.sh
```

# Citation
**Bibtex** -
```
@misc{kumar2025crowdsurfersamplingoptimizationaugmented,
      title={CrowdSurfer: Sampling Optimization Augmented with Vector-Quantized Variational AutoEncoder for Dense Crowd Navigation}, 
      author={Naman Kumar and Antareep Singha and Laksh Nanwani and Dhruv Potdar and Tarun R and Fatemeh Rastgar and Simon Idoko and Arun Kumar Singh and K. Madhava Krishna},
      year={2025},
      eprint={2409.16011},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.16011}, 
}
```
