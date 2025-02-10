FROM python:3.8.12-buster

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY Carbon_Dioxide_Levels_in_Atmosphere_package_folder Carbon_Dioxide_Levels_in_Atmosphere_package_folder
COPY setup.py setup.py
RUN pip install .

COPY data data

COPY Makefile Makefile

CMD uvicorn Carbon_Dioxide_Levels_in_Atmosphere_package_folder.api:app --host 0.0.0.0 --port $PORT
