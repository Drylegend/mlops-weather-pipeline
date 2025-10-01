# 1. Use an official lightweight Python image as a base
FROM python:3.13-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install the Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code and artifacts into the container
COPY . .

# 6. Tell Docker that the container will listen on port 8000
EXPOSE 8000

# 7. Define the command to run your Uvicorn server when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]