FROM deepset/hayhooks:main

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/Rusteam/hayhooks@fix-typing
