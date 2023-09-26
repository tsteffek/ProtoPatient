FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

WORKDIR experiments
COPY . .

RUN pip uninstall -y typing_extensions

RUN pip install --no-cache -r requirements.txt

CMD ["training.py", "--help"]
