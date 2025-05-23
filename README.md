# Implemention of generative AI with sampling optimizer for path planning for a mobile robot

- reimplemented with RVIZ visualiztion in closed loop with a rosbag and closed loop in pedsim gazebo simulation
- implemented generative AI for path planning for mobile robot in a crowded environment using VQ-VAE and PixelCNN

![teaser](./sampling_from_vqvae.png)
flowchart made by [Naman Kumar](https://github.com/namanxkumar)

![teaser](./PixelCNN.png)
flowchart made by [Soumo Roy](https://github.com/s0um0r0y)

## Built With
- Pedsim Gazebo Simulation
- ROS 1 noetic
- CUDA 11.8
- Pytorch

## RVIZ window

- blue colour line is VQ-VAE + PixelCNN generated trajectory
- green colour line is PRIEST optimized trajectory
- green colour marker is the dynamic obstacles
- [Simulation Demo](https://drive.google.com/file/d/1nSyOIk4JmVDSuj6wRM4vHAArfcK2Fj91/view?usp=sharing)
  
![comparison_vqvae_pixelcnn_PRIEST_optimzer](https://github.com/user-attachments/assets/8896391a-1b49-4353-86b8-a23a5c3fdb22)

## Pedsim Gazebo simulator

- Validated results in a crowded environment simulation to avoid humans (dynamic obstacles)
- [Simulation Demo](https://drive.google.com/file/d/19sQzzvD0daZ0SYvZoFq8Gw9cFL1mPFsC/view?usp=sharing)

![teaser](./gazebo_simulation.png)

## How to run ?
```
  # this command is for running open loop using a rosbag
  conda env create -f environment.yml
  ./run_open_loop_rosbag.sh

  # this command is for running closed loop in pedsim gazebo simulation
  ./run_closed_loop_simulation.sh
```

## Author
- [Soumo Roy](https://github.com/s0um0r0y) - soumoroy09@gmail.com
- [Aadith Warrier](https://github.com/aadith-warrier) (for guidance)

## Future Work
- Docker setup for running with ease
- Devconatiner setup for running with VS Code extension 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Citation
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

## Remark 

This work is a re-implemtation of [Crowdsurfer paper](https://github.com/Smart-Wheelchair-RRC/CrowdSurfer) which was published at IEEE ICRA 2025 (A* star conference for robotics) at IIIT hyderabad under the guidance of [Naman Kumar](https://github.com/namanxkumar) and [Dr. Madhava Krishna](https://robotics.iiit.ac.in/faculty_mkrishna/.)
