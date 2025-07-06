import subprocess

def run_prediction(audio_path, audio_ext='mp3', accelerator='cpu', devices=1, checkpoint='weights/transcription1000.ckpt'):
    cmd = [
        'python', 'pred_jointist.py',
        'audio_path=songs',
        'audio_ext=mp3',
        'trainer.accelerator=cpu',    # override existing key (no +)
        '+trainer.devices=1',         # append new key (with +)
        'checkpoint.transcription=weights/transcription1000.ckpt'
    ]

    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    run_prediction(audio_path='songs')
