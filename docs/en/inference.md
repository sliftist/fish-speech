# Inference

Inference support command line, HTTP API and web UI.

!!! note
    Overall, reasoning consists of several parts:

    1. Encode a given ~10 seconds of voice using VQGAN.
    2. Input the encoded semantic tokens and the corresponding text into the language model as an example.
    3. Given a new piece of text, let the model generate the corresponding semantic tokens.
    4. Input the generated semantic tokens into VITS / VQGAN to decode and generate the corresponding voice.

## Command Line Inference

Download the required `vqgan` and `llama` models from our Hugging Face repository.

```bash
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5
```

### 1. Generate prompt from voice:

!!! note
    If you plan to let the model randomly choose a voice timbre, you can skip this step.

```bash
python tools/vqgan/inference.py -i "D:\repos\cli-reader\samples\margot\(very good) serious - 6abfc430b14f4f6a9d6a74b6c7c84622\serious_77s.mp3" --checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
```

You should get a `fake.npy` file.

### 2. Generate semantic tokens from text:

TEST
It's so evident, I mean when Stephen wrote the script, Trump wasn't present, there wasn't such clear divide in the country, that there was a divide and classism is and was an issue, but became even more relevant now, in the last year. And you know, abused women became so sadly relevant and brought into light in the last couple months. And obviously this deals with someone who endured abuse as well. And there's so many bigger issues. And the thing I gravitated towards was it was just a human story. You got people who fear loss, people who are frustrated, people who are disenfranchised. They're all coming up with a version of truth, their own truth, to try to warrant their behavior because there's a lot of pain in their past, perhaps. It's a very human experience that to begin with, you're like, I don't know anything about ice skating. I don't really care about that stuff. I'm not from that part of America. How is this relevant to me in my life? And then you kind of delve into the story and you start relating on levels you never expected you could. And you start thinking big picture. Is this something our society needs to identify and deal with? I think it's important to tell stories that are entertaining, but to entertain in a meaningful way. That makes the job worthwhile. 

python tools/llama/generate.py --text "Whoa, don't touch me!" --prompt-text "It's so evident, I mean when Stephen wrote the script, Trump wasn't present, there wasn't such clear divide in the country, that there was a divide and classism is and was an issue, but became even more relevant now, in the last year. And you know, abused women became so sadly relevant and brought into light in the last couple months. And obviously this deals with someone who endured abuse as well. And there's so many bigger issues. And the thing I gravitated towards was it was just a human story. You got people who fear loss, people who are frustrated, people who are disenfranchised. They're all coming up with a version of truth, their own truth, to try to warrant their behavior because there's a lot of pain in their past, perhaps. It's a very human experience that to begin with, you're like, I don't know anything about ice skating. I don't really care about that stuff. I'm not from that part of America. How is this relevant to me in my life? And then you kind of delve into the story and you start relating on levels you never expected you could. And you start thinking big picture. Is this something our society needs to identify and deal with? I think it's important to tell stories that are entertaining, but to entertain in a meaningful way. That makes the job worthwhile." --prompt-tokens "fake.npy" --checkpoint-path "checkpoints/fish-speech-1.5" --num-samples 100 --compile
python tools/vqgan/inference.py -i codes_0.npy --output-path fake0.wav --checkpoint-path checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth
python tools/vqgan/inference.py -i codes_1.npy --output-path fake1.wav --checkpoint-path checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth


python -c "__import__('playsound').playsound('fake0.wav')" && python -c "__import__('playsound').playsound('fake1.wav')" && python -c "__import__('playsound').playsound('fake2.wav')" && python -c "__import__('playsound').playsound('fake3.wav')" && python -c "__import__('playsound').playsound('fake4.wav')" && python -c "__import__('playsound').playsound('fake5.wav')" && python -c "__import__('playsound').playsound('fake6.wav')" && python -c "__import__('playsound').playsound('fake7.wav')" && python -c "__import__('playsound').playsound('fake8.wav')" && python -c "__import__('playsound').playsound('fake9.wav')"

```bash
python tools/llama/generate.py --text "What is the prompt text anyways?" --prompt-text "" --prompt-tokens "fake.npy" --checkpoint-path "checkpoints/fish-speech-1.5" --num-samples 1 --compile
```

This command will create a `codes_N` file in the working directory, where N is an integer starting from 0.

!!! note
    You may want to use `--compile` to fuse CUDA kernels for faster inference (~30 tokens/second -> ~500 tokens/second).
    Correspondingly, if you do not plan to use acceleration, you can comment out the `--compile` parameter.

!!! info
    For GPUs that do not support bf16, you may need to use the `--half` parameter.

### 3. Generate vocals from semantic tokens:

#### VQGAN Decoder

```bash
python tools/vqgan/inference.py -i "codes_0.npy" --checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
```

## HTTP API Inference

We provide a HTTP API for inference. You can use the following command to start the server:

```bash
python -m tools.api_server --listen 127.0.0.1:8080 --llama-checkpoint-path "checkpoints/fish-speech-1.5" --decoder-checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" --decoder-config-name firefly_gan_vq
```

> If you want to speed up inference, you can add the `--compile` parameter.

After that, you can view and test the API at http://127.0.0.1:8080/.

Below is an example of sending a request using `tools/api_client.py`.

```bash
python -m tools.api_client \
    --text "Text to be input" \
    --reference_audio "Path to reference audio" \
    --reference_text "Text content of the reference audio" \
    --streaming True
```

The above command indicates synthesizing the desired audio according to the reference audio information and returning it in a streaming manner.

The following example demonstrates that you can use **multiple** reference audio paths and reference audio texts at once. Separate them with spaces in the command.

```bash
python -m tools.api_client \
    --text "Text to input" \
    --reference_audio "reference audio path1" "reference audio path2" \
    --reference_text "reference audio text1" "reference audio text2"\
    --streaming False \
    --output "generated" \
    --format "mp3"
```

The above command synthesizes the desired `MP3` format audio based on the information from multiple reference audios and saves it as `generated.mp3` in the current directory.

You can also use `--reference_id` (only one can be used) instead of `--reference-audio` and `--reference_text`, provided that you create a `references/<your reference_id>` folder in the project root directory, which contains any audio and annotation text. 
The currently supported reference audio has a maximum total duration of 90 seconds.


!!! info 
    To learn more about available parameters, you can use the command `python -m tools.api_client -h`

## GUI Inference 
[Download client](https://github.com/AnyaCoder/fish-speech-gui/releases)

## WebUI Inference

You can start the WebUI using the following command:

```bash
python -m tools.run_webui --llama-checkpoint-path "checkpoints/fish-speech-1.5" --decoder-checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" --decoder-config-name firefly_gan_vq
```
> If you want to speed up inference, you can add the `--compile` parameter.

!!! note
    You can save the label file and reference audio file in advance to the `references` folder in the main directory (which you need to create yourself), so that you can directly call them in the WebUI.

!!! note
    You can use Gradio environment variables, such as `GRADIO_SHARE`, `GRADIO_SERVER_PORT`, `GRADIO_SERVER_NAME` to configure WebUI.

Enjoy!
