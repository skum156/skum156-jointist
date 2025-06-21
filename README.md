# Jointist-Custom: Updated Jointist Model


Steps: Set up virtual environment and then download weights and then add input file into "songs" folder and weights into "weights" folder and then run the inference with these flags.   
Flags to Use: 

HYDRA_FULL_ERROR=1 python pred_jointist.py \
  audio_path=songs \
  audio_ext=mp3 \
  gpus=[0] \
  checkpoint.transcription=weights/transcription1000.ckpt


> A fork of [KinWaiCheuk/Jointist](https://github.com/KinWaiCheuk/Jointist), modified for our AMT research.

---

## 📖 Description

Jointist-Custom extends the original Jointist multi-instrument transcription model by:

- Adding a preprocessing pipeline for custom input files  
- Integrating pretrained Jointist weights  
- Install scripts and update requirements.txt for dependency management

Use this model in our research paper in purpose of benchmarking this updated model on our dataset and test model with AMT custom scoring pipeline. 
Link to paper: 


---


