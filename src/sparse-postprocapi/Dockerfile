# Using official python runtime base image
FROM python:3.7.4

# Set the app directory
WORKDIR /usr/src/app

# Update packages and install libraries
RUN apt-get update && apt-get install -y python3-pip \
 && apt-get install -y python3-numpy && apt-get install -y libatlas-base-dev

# Upgrade pip
RUN pip install --upgrade pip

# Install numpy using precompiled wheels
RUN pip install --index-url=https://www.piwheels.org/simple/ \
    NUMPY CCHARDET PYCARES CFFI MULTIDICT YARL HIREDIS PYYAML

# Install other requirements
RUN pip install AIOHTTP AIODNS AIOREDIS NEST_ASYNCIO KUBERNETES

# Copy all to current dir
COPY . .

# Make port 8082 available for webapi
EXPOSE 8082

# Define the command to be run when launching the container
CMD ["python","-u","./app.py"]